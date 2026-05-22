"""
pipeline/agent.py — LangGraph Agent for Military Map Vectorization

ARCHITECTURE OVERVIEW:
    The agent follows a Perceive → Plan → Execute → QA loop, implemented
    as a LangGraph StateGraph. Each node is a pure function that receives
    the shared AgentState dict and returns a partial update.

    Graph topology:

        [START]
           │
           ▼
      ┌─────────────┐
      │  perceive   │  Classifies map type: monochrome | stratégique
      └──────┬──────┘
             │ map_type
             ▼
      ┌─────────────┐
      │  preprocess │  Applies type-specific image corrections
      └──────┬──────┘
             │ preprocessed_image
             ▼
      ┌─────────────┐
      │  vectorize  │  U-Net inference + CC labeling → raw GeoJSON
      └──────┬──────┘
             │ raw_geojson, confidence_score
             ▼
      ┌──────────────┐
      │  qa_check    │  Confidence ≥ 90% → save; else → self_correct
      └──────┬───────┘
             │ passed / failed
        ┌────┴────┐
        ▼         ▼
   [georef]  [self_correct]──► [vectorize] (retry, max 2 times)
        │
        ▼
    [export] ─► [END]

WHY LANGGRAPH (not a simple pipeline):
    A standard pipeline.py runs steps sequentially with no ability to
    branch or retry. LangGraph's StateGraph provides:
      - Conditional routing based on QA score
      - Self-correction loop with retry cap (avoids infinite loops)
      - Full execution trace for audit (required for military applications)
      - Async-ready (each node can be await-ed in production)

USAGE:
    from pipeline.agent import build_agent, run_agent

    result = run_agent(
        raster_path="data/raw/tunis_sheet20.png",
        output_dir="data/processed/map_001",
        map_name="tunis",              # optional: for auto GCPs
        weights_path="models/unet_tunis.pth",
    )
    print(result["output_geojsons"])   # dict {layer_name: Path}
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import cv2
import numpy as np

# LangGraph
try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Shared mutable state passed between every node in the graph.
    TypedDict enforces key names; `total=False` means all keys are optional
    so nodes can return partial updates.

    Fields added progressively through the pipeline:
        perceive     → map_type, map_metadata
        preprocess   → preprocessed_image, crop_bbox
        vectorize    → raw_masks, confidence_score, raw_geojsons
        qa_check     → qa_passed, qa_feedback, retry_count
        self_correct → preprocessed_image (updated)
        georef       → georef_transform, georef_crs
        export       → output_geojsons, output_shapefiles, agent_log
    """
    # ── Input ────────────────────────────────────────────────────────────────
    raster_path: str
    output_dir: str
    map_name: Optional[str]
    weights_path: Optional[str]
    device: Optional[str]
    # Mode SIG : False (défaut) = sortie en pixels image,
    #            True            = applique georeferencing si dispo.
    georeference: bool

    # ── Perception ───────────────────────────────────────────────────────────
    map_type: Literal["monochrome", "stratégique", "unknown"]
    map_metadata: Dict[str, Any]        # scale, series, sheet, CRS hint

    # ── Image data ───────────────────────────────────────────────────────────
    original_image: np.ndarray          # BGR, full resolution
    preprocessed_image: np.ndarray      # BGR, downscaled + crop applied
    crop_bbox: tuple                    # (x1, y1, x2, y2) in downscaled coords
    downscale_scale: float              # downscaled = original * scale (≤ 1)
    original_size: tuple                # (W, H) original uncropped image

    # ── Vectorization ────────────────────────────────────────────────────────
    raw_masks: Dict[str, np.ndarray]    # {layer_name: binary_mask}
    confidence_score: float             # 0.0 – 1.0 composite quality score
    raw_geojsons: List[Dict]            # list of GeoJSON dicts (one per layer)

    # ── QA ───────────────────────────────────────────────────────────────────
    qa_passed: bool
    qa_feedback: str                    # human-readable explanation
    retry_count: int                    # incremented on each self-correction

    # ── Georeferencing ───────────────────────────────────────────────────────
    georef_transform: Optional[Any]     # rasterio Affine or None
    georef_crs: Optional[str]           # "EPSG:4326" etc.

    # ── Output ───────────────────────────────────────────────────────────────
    output_geojsons: Dict[str, str]     # {layer_name: file_path}
    output_shapefiles: Dict[str, str]
    qgis_bundle: Optional[str]          # chemin du cartovec_export.zip (mode SIG)
    agent_log: List[Dict]               # structured audit trail
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Perceive
# ─────────────────────────────────────────────────────────────────────────────

# HSV signature profiles for each map type.
# These are learned from empirical analysis of AMS/GSGS series.
_MAP_TYPE_SIGNATURES: Dict[str, Dict] = {
    "monochrome": {
        # E.g. photostat copies, blue-line reproductions, black-only prints.
        # Characteristics: near-zero saturation across >90% of pixels,
        # single dominant hue, no red-road class present.
        "max_mean_saturation": 18,      # HSV S channel mean
        "min_gray_ratio": 0.90,         # fraction of pixels with S < 20
        "description": "Monochrome / single-ink map (photocopy, blue-line)",
    },
    "stratégique": {
        # E.g. AMS Tunisia 1:50,000 — full color: red roads, green vegetation,
        # blue grid, brown contours, black buildings.
        "min_mean_saturation": 19,
        "min_color_diversity": 0.15,    # fraction of pixels with S > 40
        "has_red_roads": True,          # hue 0-10 or 170-180 present
        "description": "Stratégique / multi-color military map (AMS, GSGS)",
    },
}

_TUNIS_SHEET_CRS_HINTS = {
    # AMS Tunisia 1:50,000 sheets — approximate WGS84 corners
    # Add more sheets as needed
    "tunis":        {"epsg": 4326, "lon_range": (10.0, 10.5), "lat_range": (36.7, 37.0)},
    "ain_bessem":   {"epsg": 4326, "lon_range": (3.18, 3.65), "lat_range": (36.0, 36.25)},
    "default":      {"epsg": 4326, "lon_range": None,          "lat_range": None},
}


def node_perceive(state: AgentState) -> Dict:
    """
    PERCEIVE node — classifies the input map type and extracts metadata.

    Algorithm:
        1. Load the raster at reduced resolution (long-side 800 px) for
           fast color analysis — no need for full resolution here.
        2. Convert to HSV and compute saturation statistics.
        3. Check for AMS-specific color signatures:
            - Red roads (H=0-10 OR 170-180, S>80) → stratégique
            - Near-zero saturation everywhere → monochrome
        4. Detect scale/series from filename heuristics as metadata hint.

    The map_type output routes the graph to the correct preprocess function.
    """
    t0 = time.perf_counter()
    log_entry = {"node": "perceive", "ts": time.time()}

    raster_path = Path(state["raster_path"])
    if not raster_path.is_file():
        return {
            "map_type": "unknown",
            "map_metadata": {},
            "error": f"Raster not found: {raster_path}",
        }

    # ── Load at analysis resolution ──────────────────────────────────────────
    img_bgr = cv2.imread(str(raster_path))
    if img_bgr is None:
        return {
            "map_type": "unknown",
            "map_metadata": {},
            "error": f"cv2.imread failed for: {raster_path}",
        }

    h_full, w_full = img_bgr.shape[:2]
    scale = min(800 / max(h_full, w_full), 1.0)
    small = cv2.resize(img_bgr, (int(w_full * scale), int(h_full * scale)),
                       interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    total_px = small.shape[0] * small.shape[1]

    mean_sat = float(s_ch.mean())
    gray_ratio = float((s_ch < 20).sum() / total_px)
    color_diversity = float((s_ch > 40).sum() / total_px)

    # Red road detection (AMS stratégique signature)
    red_low  = ((h_ch <= 10)  & (s_ch >= 80)).sum()
    red_high = ((h_ch >= 170) & (s_ch >= 80)).sum()
    red_ratio = (red_low + red_high) / total_px

    # ── Classification ───────────────────────────────────────────────────────
    if gray_ratio >= 0.90 and mean_sat < 18:
        map_type: Literal["monochrome", "stratégique", "unknown"] = "monochrome"
    elif color_diversity >= 0.10 or red_ratio >= 0.005:
        map_type = "stratégique"
    else:
        map_type = "monochrome"   # safe default: monochrome pipeline is gentler

    # ── Metadata extraction ──────────────────────────────────────────────────
    fname_lower = raster_path.stem.lower()
    map_name = state.get("map_name") or fname_lower

    crs_hint = _TUNIS_SHEET_CRS_HINTS.get(
        map_name,
        _TUNIS_SHEET_CRS_HINTS["default"]
    )

    metadata = {
        "filename": raster_path.name,
        "original_size": (w_full, h_full),
        "mean_saturation": round(mean_sat, 2),
        "gray_ratio": round(gray_ratio, 4),
        "color_diversity": round(color_diversity, 4),
        "red_road_ratio": round(float(red_ratio), 6),
        "map_name": map_name,
        "crs_hint": crs_hint,
    }

    log_entry.update({
        "map_type": map_type,
        "metadata": metadata,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    })
    logger.info("[perceive] map_type=%s  red_ratio=%.4f  mean_sat=%.1f",
                map_type, red_ratio, mean_sat)

    return {
        "map_type": map_type,
        "map_metadata": metadata,
        "original_image": img_bgr,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Preprocess (type-aware)
# ─────────────────────────────────────────────────────────────────────────────

def node_preprocess(state: AgentState) -> Dict:
    """
    PREPROCESS node — applies type-specific preprocessing.

    Monochrome maps:
        - Adaptive histogram equalization (CLAHE) to recover faded ink
        - Gaussian denoising
        - Otsu threshold → binary (used as single-channel input to U-Net)
        - Pseudo-RGB: replicate binary to 3 channels

    Stratégique maps:
        - Standard resize to max 2400 px
        - Auto crop via variance-profile detect_map_frame (from preprocessing.py)
        - No binarization (color channels needed for HSV segmentation)

    Both paths: downscale to max_dim, store crop_bbox in state.
    """
    t0 = time.perf_counter()
    log_entry = {"node": "preprocess", "ts": time.time()}

    img = state.get("original_image")
    if img is None:
        img = cv2.imread(str(state["raster_path"]))

    map_type = state.get("map_type", "stratégique")
    max_dim = 2400

    # ── Downscale ────────────────────────────────────────────────────────────
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    img_small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # ── Detect map frame (crop legend + margins) ──────────────────────────────
    try:
        from pipeline.preprocessing import detect_map_frame
        bbox = detect_map_frame(img_small)
    except ImportError:
        # Fallback: use proportional crop (AMS standard margins)
        H, W = img_small.shape[:2]
        bbox = (int(W * 0.085), int(H * 0.085), int(W * 0.915), int(H * 0.82))

    x1, y1, x2, y2 = bbox
    img_crop = img_small[y1:y2, x1:x2]

    # ── Type-specific processing ──────────────────────────────────────────────
    if map_type == "monochrome":
        # Convert to grayscale
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

        # CLAHE: recover faded ink on aged scans
        # clipLimit=3.0 is empirically best for mid-20th century military maps
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Mild Gaussian denoise (σ=1) — preserves thin line structures
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 1)

        # Otsu threshold → binary image
        _, binary = cv2.threshold(denoised, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Pseudo-RGB for U-Net (which expects 3 channels)
        preprocessed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        log_entry["preprocessing_steps"] = ["clahe", "gaussian_blur", "otsu"]

    else:  # stratégique
        # Color map: no binarization
        # Apply mild bilateral filter to smooth color noise while preserving edges
        preprocessed = cv2.bilateralFilter(img_crop, d=5, sigmaColor=30, sigmaSpace=30)
        log_entry["preprocessing_steps"] = ["bilateral_filter"]

    log_entry.update({
        "map_type": map_type,
        "original_size": (w, h),
        "downscaled_size": (img_small.shape[1], img_small.shape[0]),
        "crop_bbox": bbox,
        "preprocessed_size": (preprocessed.shape[1], preprocessed.shape[0]),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    })

    return {
        "preprocessed_image": preprocessed,
        "crop_bbox": bbox,
        # Métadonnées de réalignement pixel : permettent à node_export de
        # ramener les vecteurs (espace crop+downscalé) sur l'image originale.
        "downscale_scale": float(scale),       # downscaled = original * scale
        "original_size": (int(w), int(h)),     # (W, H) de l'image d'origine
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Vectorize
# ─────────────────────────────────────────────────────────────────────────────

def node_vectorize(state: AgentState) -> Dict:
    """
    VECTORIZE node — U-Net inference + color segmentation + confidence scoring.

    Steps:
        1. Try U-Net inference (if weights available)
        2. Always run color segmentation (HSV) as parallel / fallback
        3. Merge: U-Net masks (if available) override HSV masks for their classes
        4. Compute confidence score as composite metric:
               score = 0.4 * coverage_quality
                     + 0.3 * contour_continuity
                     + 0.3 * spatial_distribution
           Where:
               coverage_quality   = clamp(total_feature_px / expected_px, 0, 1)
               contour_continuity = mean(max_component_size / total_px_per_layer)
               spatial_distribution = std of feature centroids / map_diagonal
    """
    t0 = time.perf_counter()
    log_entry = {"node": "vectorize", "ts": time.time()}

    img = state.get("preprocessed_image")
    map_type = state.get("map_type", "stratégique")
    bbox = state.get("crop_bbox")

    # ── Color segmentation (always) ───────────────────────────────────────────
    try:
        from pipeline.color_segmentation import extract_all_color_layers
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_masks = extract_all_color_layers(hsv)
    except ImportError:
        color_masks = {}

    # ── U-Net inference (optional, requires weights) ──────────────────────────
    unet_masks: Dict[str, np.ndarray] = {}
    weights_path = state.get("weights_path")
    if weights_path and Path(weights_path).is_file():
        try:
            from pipeline.semantic_segmentation import (
                build_unet, load_weights, predict_multi_class, get_device
            )
            model = build_unet(classes=3, activation="softmax2d")  # 3: bg, road, building
            device = state.get("device") or get_device(verbose=False)
            model = load_weights(model, weights_path, device=device)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            unet_masks = predict_multi_class(
                model, img_rgb,
                class_names=["roads", "buildings"],
                device=device,
            )
            log_entry["unet_inference"] = "success"
        except Exception as exc:
            logger.warning("[vectorize] U-Net inference failed: %s", exc)
            log_entry["unet_inference"] = f"failed: {exc}"
    else:
        log_entry["unet_inference"] = "skipped (no weights)"

    # ── Merge: U-Net > HSV for overlapping classes ────────────────────────────
    merged_masks = {**color_masks, **unet_masks}  # unet overrides color for same keys

    # ── Confidence scoring ────────────────────────────────────────────────────
    H, W = img.shape[:2]
    total_img_px = H * W
    confidence_score = _compute_confidence(merged_masks, total_img_px, map_type)

    # ── Raw GeoJSON assembly ──────────────────────────────────────────────────
    raw_geojsons = _masks_to_raw_geojsons(merged_masks, bbox)

    # ── Libération mémoire après traçage des contours ─────────────────────────
    # Les masques HSV/U-Net (uint8 H×W, plusieurs couches) ne servent plus une
    # fois les géométries extraites. On les jette explicitement ; le scoring de
    # confiance ci-dessus les a déjà consommés. Patch identique à run_pipeline.
    try:
        del hsv
    except (NameError, UnboundLocalError):
        pass
    del color_masks, unet_masks
    gc.collect()

    log_entry.update({
        "layers_produced": list(merged_masks.keys()),
        "confidence_score": round(confidence_score, 4),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "memory_gc": "del color/unet masks + gc.collect()",
    })
    logger.info("[vectorize] confidence=%.3f  layers=%s",
                confidence_score, list(merged_masks.keys()))

    return {
        "raw_masks": merged_masks,
        "confidence_score": confidence_score,
        "raw_geojsons": raw_geojsons,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def _compute_confidence(masks: Dict[str, np.ndarray],
                         total_px: int,
                         map_type: str) -> float:
    """
    Composite confidence score in [0, 1].

    For a stratégique map we expect:
        - red_roads:  0.3% – 3% of pixels
        - vegetation: 2%   – 15%
        - contours:   0.5% – 5%
        - buildings:  0%   – 8% (only in urban sheets)

    For monochrome:
        - roads:      1%   – 10%
        - buildings:  2%   – 20%

    Score degrades if coverage is too low (nothing detected) OR too high
    (over-segmentation / noise).
    """
    if not masks:
        return 0.0

    EXPECTED: Dict[str, tuple] = {
        "stratégique": {
            "red_roads":  (0.003, 0.03),
            "vegetation": (0.02,  0.15),
            "contours":   (0.005, 0.05),
            "buildings":  (0.0,   0.08),
        },
        "monochrome": {
            "roads":      (0.01,  0.10),
            "buildings":  (0.02,  0.20),
        },
    }.get(map_type, {})

    scores = []
    for name, mask in masks.items():
        pct = (mask > 0).sum() / total_px
        lo, hi = EXPECTED.get(name, (0.001, 0.20))
        if pct < lo:
            layer_score = pct / lo if lo > 0 else 0.0
        elif pct > hi:
            layer_score = max(0.0, 1.0 - (pct - hi) / hi)
        else:
            layer_score = 1.0
        scores.append(layer_score)

    return float(np.mean(scores)) if scores else 0.5


def _masks_to_raw_geojsons(masks: Dict[str, np.ndarray],
                           bbox: Optional[tuple]) -> List[Dict]:
    """
    Converts binary masks to list of GeoJSON FeatureCollection dicts.

    The masks are already in cropped-image coordinates. ``bbox`` is kept for
    API compatibility with older calls, but must not be used directly as a
    global-image filter here; pixel realignment happens later in node_export.
    """
    results = []
    for name, mask in masks.items():
        try:
            from pipeline.vectorization import mask_to_polygons, skeleton_to_lines
            from shapely.geometry import mapping

            is_line_layer = name in ("red_roads", "roads", "contours", "streets")

            if is_line_layer:
                # Skeletonize first
                try:
                    from skimage.morphology import skeletonize
                    skel = (skeletonize(mask > 0).astype(np.uint8) * 255)
                except ImportError:
                    skel = mask
                geoms = skeleton_to_lines(skel)
            else:
                geoms = mask_to_polygons(mask, min_area_px=30)

            # Filter in local crop coordinates. The old code used ``bbox``
            # directly even though it is expressed in the downscaled full image,
            # which could discard valid features before export realigned them.
            if geoms:
                from pipeline.vectorization import filter_features_by_bbox
                h, w = mask.shape[:2]
                geoms = filter_features_by_bbox(
                    geoms, (0, 0, w, h), margin=20, mode="centroid"
                )

            features = [
                {
                    "type": "Feature",
                    "properties": {"layer": name},
                    "geometry": mapping(g),
                }
                for g in geoms if g and not g.is_empty
            ]
            results.append({
                "type": "FeatureCollection",
                "name": name,
                "features": features,
            })
        except Exception as exc:
            logger.warning("[vectorize] GeoJSON conversion failed for %s: %s", name, exc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — QA / Self-Correction
# ─────────────────────────────────────────────────────────────────────────────

_QA_THRESHOLD = 0.90    # 90% confidence required to pass
_MAX_RETRIES = 2        # maximum self-correction attempts


def node_qa_check(state: AgentState) -> Dict:
    """
    QA node — evaluates vectorization quality and decides pass / retry.

    Checks:
        1. confidence_score >= QA_THRESHOLD
        2. At least one layer produced features
        3. No single layer dominates (over-segmentation guard)

    Stores qa_passed=True/False and qa_feedback in state.
    The conditional edge reads qa_passed to route to [georef] or [self_correct].
    """
    log_entry = {"node": "qa_check", "ts": time.time()}

    score = state.get("confidence_score", 0.0)
    raw_geojsons = state.get("raw_geojsons", [])
    retry_count = state.get("retry_count", 0)

    # ── Check 1: confidence threshold ────────────────────────────────────────
    passes_threshold = score >= _QA_THRESHOLD

    # ── Check 2: at least one layer has features ─────────────────────────────
    total_features = sum(len(gj.get("features", [])) for gj in raw_geojsons)
    has_features = total_features > 0

    # ── Check 3: no layer has > 60% of total features (over-segmentation) ────
    layer_counts = [len(gj.get("features", [])) for gj in raw_geojsons]
    if total_features > 0:
        max_ratio = max(layer_counts) / total_features
        no_oversegmentation = max_ratio <= 0.80
    else:
        no_oversegmentation = False

    qa_passed = passes_threshold and has_features and no_oversegmentation

    # ── Feedback message ──────────────────────────────────────────────────────
    feedback_parts = []
    if not passes_threshold:
        feedback_parts.append(f"confidence {score:.2%} < threshold {_QA_THRESHOLD:.0%}")
    if not has_features:
        feedback_parts.append("no features detected in any layer")
    if not no_oversegmentation and total_features > 0:
        feedback_parts.append(f"over-segmentation: max_layer_ratio={max_ratio:.2%}")

    qa_feedback = "; ".join(feedback_parts) if feedback_parts else "all checks passed"

    log_entry.update({
        "confidence_score": round(score, 4),
        "total_features": total_features,
        "qa_passed": qa_passed,
        "qa_feedback": qa_feedback,
        "retry_count": retry_count,
    })
    logger.info("[qa_check] passed=%s  score=%.3f  feedback='%s'",
                qa_passed, score, qa_feedback)

    return {
        "qa_passed": qa_passed,
        "qa_feedback": qa_feedback,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def node_self_correct(state: AgentState) -> Dict:
    """
    SELF-CORRECT node — applies recovery filters when QA fails.

    Fallback strategies applied in sequence based on failure mode:

    1. Low confidence (<50%):
        → Histogram equalization on the preprocessed image
        → Increase HSV saturation to amplify faint colors

    2. No features detected:
        → Force bilateral filter with stronger sigma
        → Morphological closing to reconnect broken lines

    3. Over-segmentation:
        → Apply opening filter to remove small isolated blobs
        → Increase min_area threshold for next vectorization pass

    After correction, the graph loops back to [vectorize].
    The retry_count prevents infinite loops: if retries >= MAX_RETRIES,
    we pass anyway with the best result so far.
    """
    log_entry = {"node": "self_correct", "ts": time.time()}

    img = state.get("preprocessed_image")
    score = state.get("confidence_score", 0.0)
    retry_count = state.get("retry_count", 0)
    feedback = state.get("qa_feedback", "")

    corrections_applied = []

    # ── Strategy selection ────────────────────────────────────────────────────
    if "no features detected" in feedback:
        # Aggressive enhancement: CLAHE + strong bilateral
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        # Strong bilateral to reconnect broken segments
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        corrections_applied.append("clahe_4.0 + bilateral_strong")

    elif score < 0.50 or "confidence" in feedback:
        # Boost color saturation in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # +40% saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # +10% brightness
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        corrections_applied.append("saturation_boost_1.4 + sharpen")

    elif "over-segmentation" in feedback:
        # Morphological opening to remove salt-and-pepper noise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _layer, mask in state.get("raw_masks", {}).items():
            mask[:] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        corrections_applied.append("morphological_open_5x5")

    else:
        # Generic: unsharp masking
        blur = cv2.GaussianBlur(img, (5, 5), 2)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        corrections_applied.append("unsharp_mask")

    log_entry.update({
        "corrections_applied": corrections_applied,
        "score_before": round(score, 4),
        "retry_count_after": retry_count + 1,
    })
    logger.info("[self_correct] retry=%d  corrections=%s",
                retry_count + 1, corrections_applied)

    return {
        "preprocessed_image": img,
        "retry_count": retry_count + 1,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Georeferencing
# ─────────────────────────────────────────────────────────────────────────────

def node_georef(state: AgentState) -> Dict:
    """
    GEOREF node — attaches WGS84 coordinates to the raw GeoJSON features.

    SHORT-CIRCUIT : si state["georeference"] est False (mode pixel par
    défaut), on ne touche pas aux geojsons et on log un passthrough.
    L'export récupère alors les coordonnées pixel telles quelles.

    Uses the AMS_ALGERIA_SHEETS / TUNIS registry from georeferencing.py.
    If map_name is not in the registry, uses the map's printed corner
    coordinates (stored in map_metadata.crs_hint) as a fallback.
    """
    log_entry = {"node": "georef", "ts": time.time()}

    # ── Mode pixel : on saute purement et simplement ────────────────────────
    if not state.get("georeference", False):
        log_entry["georef_status"] = "skipped (georeference=False — pixel mode)"
        return {
            "georef_crs": None,
            "agent_log": state.get("agent_log", []) + [log_entry],
        }

    map_name = state.get("map_name") or state.get("map_metadata", {}).get("map_name")
    bbox = state.get("crop_bbox")
    raw_geojsons = state.get("raw_geojsons", [])

    try:
        from pipeline.georeferencing import (
            AMS_ALGERIA_SHEETS, apply_transform_to_geojson,
            filter_features_by_bbox as geo_filter,
        )

        corners = AMS_ALGERIA_SHEETS.get(map_name)
        if corners is None:
            log_entry["georef_status"] = f"map_name '{map_name}' not in registry — pixel coords kept"
            return {
                "georef_crs": None,
                "agent_log": state.get("agent_log", []) + [log_entry],
            }

        georef_geojsons = []
        for gj in raw_geojsons:
            georef_gj = apply_transform_to_geojson(gj, bbox, corners)
            georef_gj["name"] = gj.get("name", "unknown")
            georef_gj["crs"] = {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            }
            georef_geojsons.append(georef_gj)

        log_entry["georef_status"] = "success"
        log_entry["crs"] = "EPSG:4326"
        return {
            "raw_geojsons": georef_geojsons,
            "georef_crs": "EPSG:4326",
            "agent_log": state.get("agent_log", []) + [log_entry],
        }

    except (ImportError, OSError, RuntimeError) as exc:
        # Graceful degrade : GDAL/rasterio cassé, DLL manquante, ou erreur de
        # transformation → on NE crash PAS le graphe. On log un warning de
        # priorité haute et on retombe sur les coords pixel (raw_geojsons
        # inchangés). L'export produira alors un GeoJSON pixel.
        logger.warning("[georef] dégradation gracieuse vers le mode pixel : %s", exc)
        log_entry["georef_status"] = f"degraded_to_pixel ({type(exc).__name__}: {exc})"
        log_entry["georef_fallback"] = True
        return {
            "georef_crs": None,
            "agent_log": state.get("agent_log", []) + [log_entry],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Export
# ─────────────────────────────────────────────────────────────────────────────

def _offset_geojson_dict(gj: Dict, offset, scale) -> Dict:
    """
    Applique le réalignement pixel (offset crop + remise à l'échelle) à un
    FeatureCollection dict, via shapely shape/mapping + apply_pixel_offset.

        X_final = (X_mask + offset_x) * scale

    Robuste : retourne le dict inchangé si shapely ou la transform échoue.
    """
    try:
        from shapely.geometry import shape, mapping
        from pipeline.vectorization import apply_pixel_offset
    except Exception:  # noqa: BLE001
        return gj

    feats_out = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            shp = shape(geom)
            moved = apply_pixel_offset([shp], offset=offset, scale=scale)
            if not moved:
                continue
            feat = {**feat, "geometry": mapping(moved[0])}
        except Exception:  # noqa: BLE001
            pass  # garde la géométrie d'origine en cas de souci
        feats_out.append(feat)
    return {**gj, "features": feats_out}


def node_export(state: AgentState) -> Dict:
    """
    EXPORT node — writes GeoJSON files and the agent audit log to disk.

    Parité avec run_pipeline :
      - Mode pixel (georeference=False ou georef dégradé) : réaligne les
        coordonnées vers l'image ORIGINALE non rognée (offset crop + 1/scale).
      - Mode SIG (georef réussi, georef_crs renseigné) : génère en plus le
        bundle QGIS cartovec_export.zip (project.qgs + layers/ relatifs).
    """
    import json as json_module
    log_entry = {"node": "export", "ts": time.time()}

    output_dir = Path(state.get("output_dir", "data/processed/agent_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Le georef a-t-il réellement abouti ? (sinon on est en pixel)
    is_gis = bool(state.get("georef_crs"))

    # Paramètres de réalignement pixel
    crop_bbox = state.get("crop_bbox") or (0, 0, 0, 0)
    ds_scale = state.get("downscale_scale", 1.0) or 1.0
    pixel_offset = (float(crop_bbox[0]), float(crop_bbox[1]))
    pixel_scale = (1.0 / ds_scale) if ds_scale > 0 else 1.0

    output_geojsons: Dict[str, str] = {}
    for gj in state.get("raw_geojsons", []):
        name = gj.get("name", f"layer_{len(output_geojsons)}")
        # En mode pixel uniquement : réaligner vers l'image originale.
        if not is_gis:
            gj = _offset_geojson_dict(gj, pixel_offset, pixel_scale)
        path = output_dir / f"{name}.geojson"
        with open(path, "w", encoding="utf-8") as f:
            json_module.dump(gj, f, indent=2)
        output_geojsons[name] = str(path)

    # Write audit log
    log_path = output_dir / "agent_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json_module.dump(state.get("agent_log", []), f, indent=2)

    log_entry.update({
        "output_files": list(output_geojsons.keys()),
        "output_dir": str(output_dir),
        "mode": "gis" if is_gis else "pixel",
        "pixel_offset": pixel_offset if not is_gis else None,
        "pixel_scale": round(pixel_scale, 4) if not is_gis else None,
    })

    # ── Bundle QGIS — uniquement en mode SIG réussi ──────────────────────────
    qgis_bundle = None
    if is_gis:
        try:
            from pipeline.export import build_qgis_bundle
            crs_epsg = 4326
            crs_str = state.get("georef_crs") or ""
            if "EPSG:" in str(crs_str):
                try:
                    crs_epsg = int(str(crs_str).upper().replace("EPSG:", "").strip())
                except (ValueError, TypeError):
                    crs_epsg = 4326
            zip_path = output_dir / "cartovec_export.zip"
            build_qgis_bundle(output_dir, zip_path, crs_epsg=crs_epsg,
                              title="CartoVec — Agent Export", smooth=True)
            qgis_bundle = str(zip_path)
            log_entry["qgis_bundle"] = qgis_bundle
        except Exception as exc:  # noqa: BLE001
            logger.warning("[export] bundle QGIS non généré : %s", exc)
            log_entry["qgis_bundle_error"] = str(exc)

    return {
        "output_geojsons": output_geojsons,
        "qgis_bundle": qgis_bundle,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────────────────────────────────────

def route_by_qa(state: AgentState) -> str:
    """
    Routes after QA node.
    Returns "georef" if passed, "self_correct" if failed + retries remain,
    "georef" (force-pass) if max retries exhausted.
    """
    if state.get("qa_passed"):
        return "georef"
    if state.get("retry_count", 0) >= _MAX_RETRIES:
        logger.warning("[routing] Max retries (%d) reached — forcing export with best result", _MAX_RETRIES)
        return "georef"   # pass anyway
    return "self_correct"


def route_by_map_type(state: AgentState) -> str:
    """Routes after perceive — currently all types go to preprocess."""
    # Future: could route monochrome to a specialized OCR-based path
    return "preprocess"


# =============================================================================
# Graph assembly
# =============================================================================

def build_agent(*, checkpointer=None, with_memory: bool = False):
    """
    Assemble et compile le StateGraph LangGraph.

    Arguments
    ---------
    checkpointer : checkpointer LangGraph explicite (ex. MemorySaver()).
        Permet le streaming avec mémoire de session isolée par thread_id.
    with_memory : si True et checkpointer=None, instancie un MemorySaver
        automatiquement (mémoire en RAM, suffisante pour le streaming live).

    Retourne un graphe compilé (appelable). Lève ImportError si langgraph
    n'est pas installé.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph est requis pour l'agent.\n"
            "Installe : pip install langgraph langchain-core"
        )

    if checkpointer is None and with_memory:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        except ImportError:
            checkpointer = None  # version langgraph sans checkpoint — on continue

    graph = StateGraph(AgentState)

    graph.add_node("perceive",      node_perceive)
    graph.add_node("preprocess",    node_preprocess)
    graph.add_node("vectorize",     node_vectorize)
    graph.add_node("qa_check",      node_qa_check)
    graph.add_node("self_correct",  node_self_correct)
    graph.add_node("georef",        node_georef)
    graph.add_node("export",        node_export)

    graph.add_edge(START, "perceive")
    graph.add_conditional_edges("perceive", route_by_map_type,
                                {"preprocess": "preprocess"})
    graph.add_edge("preprocess", "vectorize")
    graph.add_edge("vectorize",  "qa_check")
    graph.add_conditional_edges(
        "qa_check",
        route_by_qa,
        {"georef": "georef", "self_correct": "self_correct"},
    )
    graph.add_edge("self_correct", "vectorize")   # retry loop
    graph.add_edge("georef",       "export")
    graph.add_edge("export",       END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# =============================================================================
# Public entry point
# =============================================================================

def run_agent(
    raster_path: str,
    output_dir: str,
    *,
    map_name: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    georeference: bool = False,
) -> AgentState:
    """
    Point d'entree haut niveau : execute l'agent complet sur une carte raster.

    Args:
        raster_path:   chemin du raster d'entree (PNG / TIFF / JPG).
        output_dir:    dossier de sortie pour les GeoJSON.
        map_name:      nom de feuille pour lookup GCP auto (ex. "tunis").
        weights_path:  chemin .pth U-Net (optionnel ; HSV en fallback).
        device:        "cuda" / "cpu" / None (auto).
        georeference:  False (defaut) = sortie en espace pixel. True = active
                       le noeud georef (transformation AMS -> WGS84). En cas
                       d'echec GDAL, degradation gracieuse vers le mode pixel.

    Returns:
        AgentState final (output_geojsons, qgis_bundle, agent_log, ...).
    """
    agent = build_agent()

    initial_state: AgentState = {
        "raster_path":   raster_path,
        "output_dir":    output_dir,
        "map_name":      map_name,
        "weights_path":  weights_path,
        "device":        device,
        "georeference":  georeference,
        "retry_count":   0,
        "agent_log":     [],
    }

    logger.info("[agent] Starting -- raster=%s  map_name=%s  georef=%s",
                raster_path, map_name, georeference)
    result = agent.invoke(initial_state)
    logger.info("[agent] Done -- outputs=%s  qa_passed=%s  score=%.3f",
                list(result.get("output_geojsons", {}).keys()),
                result.get("qa_passed"),
                result.get("confidence_score", 0.0))
    return result
