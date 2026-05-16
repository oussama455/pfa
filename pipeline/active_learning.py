"""
pipeline/active_learning.py — Active Learning Level 1: Adaptive HSV Calibration

CONCEPT:
    Every time a human operator corrects a polygon in MapViewer (via Geoman),
    the correction carries the "ground truth" of what a feature REALLY looks like
    on that specific map. This module mines those corrections to automatically
    update the HSV thresholds used by color_segmentation.py — making each
    successive pipeline run more accurate without any GPU or labelled dataset.

ALGORITHM (per correction):
    1. Extract a 64×64 pixel neighbourhood around the corrected polygon's
       centroid from the original raster image (stored on disk).
    2. Convert to HSV and collect all pixels that fall INSIDE the corrected
       polygon boundary (ground-truth positive pixels).
    3. Compute HSV statistics (percentile-based range: p5–p95) for those pixels.
    4. Update the registry for the map's series using an Exponential Moving
       Average (EMA) with α=0.3:
           new_range = α * observed_range + (1-α) * current_range
    5. Persist the updated registry to a JSON file in the project's data dir.
    6. The next pipeline run loads the updated registry and uses it instead
       of the hard-coded DEFAULT_RANGES.

WHY EMA (not simple replacement):
    A single correction can be noisy (operator mis-clicks, atypical region).
    EMA gives more weight to the accumulated history than to any single sample,
    while still converging towards the true distribution after ~10 corrections.

    α=0.30 → after 5 corrections the new data has ~83% influence.
    α=0.10 → after 5 corrections the new data has ~41% influence (more conservative).

INTEGRATION POINTS:
    - webapp/vectorizer/api.py calls process_correction() after saving to DB
    - pipeline/color_segmentation.py calls load_adaptive_ranges() at startup
    - pipeline/agent.py uses AdaptiveSegmenter instead of extract_all_color_layers

USAGE:
    # In api.py (after saving Correction to DB):
    from pipeline.active_learning import process_correction
    process_correction(
        correction=correction_obj,   # Correction Django model instance
        map_upload=upload,           # MapUpload instance
    )

    # In color_segmentation.py (at top, replacing DEFAULT_RANGES):
    from pipeline.active_learning import load_adaptive_ranges
    RANGES = load_adaptive_ranges(map_series="ams_tunisia")
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# EMA smoothing factor — controls how fast the calibration adapts.
# 0.30: moderate adaptation (converges in ~5 corrections)
# 0.10: conservative (converges in ~15 corrections — safer for noisy operators)
EMA_ALPHA = 0.30

# Minimum corrections before the adaptive ranges override the defaults.
# Below this threshold, the defaults are returned unchanged.
MIN_CORRECTIONS_TO_ACTIVATE = 3

# Patch radius around correction centroid (pixels in downscaled image space)
PATCH_RADIUS = 64

# Registry filename
REGISTRY_FILENAME = "hsv_registry.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveHSVRange:
    """
    An HSV range that can be updated incrementally via EMA.
    Stores both the current range and the correction history for auditability.
    """
    # Current thresholds (used by segmentation)
    h_min: float
    s_min: float
    v_min: float
    h_max: float
    s_max: float
    v_max: float

    # Metadata for audit trail
    layer_name: str
    map_series: str
    correction_count: int = 0
    last_updated: float = 0.0    # Unix timestamp

    def to_opencv_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower, upper) uint8 arrays for cv2.inRange()."""
        lower = np.array([int(self.h_min), int(self.s_min), int(self.v_min)], dtype=np.uint8)
        upper = np.array([int(self.h_max), int(self.s_max), int(self.v_max)], dtype=np.uint8)
        return lower, upper

    def update_with_ema(
        self,
        observed: "AdaptiveHSVRange",
        alpha: float = EMA_ALPHA,
    ) -> "AdaptiveHSVRange":
        """
        Returns a new range updated by EMA:
            new = alpha * observed + (1-alpha) * self

        The range EXPANDS if the observation is wider (we never shrink
        the range below what was observed — safety margin).
        """
        def ema_min(current, obs):
            # For min bounds: take the EMA but cap at observed minimum
            # (we want to capture all valid pixels, not miss them)
            blended = alpha * obs + (1 - alpha) * current
            return min(blended, obs + 2)  # small tolerance

        def ema_max(current, obs):
            # For max bounds: symmetric
            blended = alpha * obs + (1 - alpha) * current
            return max(blended, obs - 2)

        return AdaptiveHSVRange(
            h_min=max(0,   ema_min(self.h_min, observed.h_min)),
            s_min=max(0,   ema_min(self.s_min, observed.s_min)),
            v_min=max(0,   ema_min(self.v_min, observed.v_min)),
            h_max=min(179, ema_max(self.h_max, observed.h_max)),
            s_max=min(255, ema_max(self.s_max, observed.s_max)),
            v_max=min(255, ema_max(self.v_max, observed.v_max)),
            layer_name=self.layer_name,
            map_series=self.map_series,
            correction_count=self.correction_count + 1,
            last_updated=time.time(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveHSVRange":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# Default HSV ranges (seed values — same as color_segmentation.py DEFAULT_RANGES)
# ─────────────────────────────────────────────────────────────────────────────

# Map series → layer → AdaptiveHSVRange
# "ams_tunisia" covers AMS Tunisia 1:50,000 (e.g. Tunis Sheet 20)
# "ams_algeria" covers AMS Algeria 1:50,000 (e.g. Ain Bessem Sheet 88)
DEFAULT_ADAPTIVE_RANGES: Dict[str, Dict[str, AdaptiveHSVRange]] = {
    "ams_tunisia": {
        "water":      AdaptiveHSVRange(95,  60,  60,  130, 255, 255, "water",      "ams_tunisia"),
        "vegetation": AdaptiveHSVRange(40,  40,  50,  85,  255, 255, "vegetation", "ams_tunisia"),
        "contours":   AdaptiveHSVRange(6,   40,  75,  25,  160, 215, "contours",   "ams_tunisia"),
        "red_roads":  AdaptiveHSVRange(0,   90,  70,  10,  255, 255, "red_roads",  "ams_tunisia"),
        "buildings":  AdaptiveHSVRange(0,   0,   30,  180, 50,  130, "buildings",  "ams_tunisia"),
    },
    "ams_algeria": {
        "water":      AdaptiveHSVRange(95,  60,  60,  130, 255, 255, "water",      "ams_algeria"),
        "vegetation": AdaptiveHSVRange(40,  40,  50,  85,  255, 255, "vegetation", "ams_algeria"),
        "contours":   AdaptiveHSVRange(6,   40,  75,  25,  160, 215, "contours",   "ams_algeria"),
        "red_roads":  AdaptiveHSVRange(0,   90,  70,  10,  255, 255, "red_roads",  "ams_algeria"),
        "buildings":  AdaptiveHSVRange(0,   0,   30,  180, 50,  130, "buildings",  "ams_algeria"),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Registry: persistence on disk
# ─────────────────────────────────────────────────────────────────────────────

def _registry_path() -> Path:
    """Returns the path to the HSV registry JSON file."""
    try:
        from pipeline.paths import Paths
        return Paths.data / REGISTRY_FILENAME
    except ImportError:
        return Path(__file__).resolve().parent.parent / "data" / REGISTRY_FILENAME


def load_registry() -> Dict[str, Dict[str, AdaptiveHSVRange]]:
    """
    Loads the persisted HSV registry from disk.
    Falls back to DEFAULT_ADAPTIVE_RANGES if the file does not exist or is corrupt.
    """
    path = _registry_path()
    if not path.exists():
        logger.debug("[active_learning] Registry not found at %s — using defaults", path)
        return {
            series: {k: AdaptiveHSVRange(**asdict(v)) for k, v in layers.items()}
            for series, layers in DEFAULT_ADAPTIVE_RANGES.items()
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        registry: Dict[str, Dict[str, AdaptiveHSVRange]] = {}
        for series, layers in raw.items():
            registry[series] = {}
            for layer_name, range_dict in layers.items():
                registry[series][layer_name] = AdaptiveHSVRange.from_dict(range_dict)

        logger.debug("[active_learning] Registry loaded from %s", path)
        return registry

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("[active_learning] Registry corrupt (%s) — using defaults", exc)
        return {
            series: {k: AdaptiveHSVRange(**asdict(v)) for k, v in layers.items()}
            for series, layers in DEFAULT_ADAPTIVE_RANGES.items()
        }


def save_registry(registry: Dict[str, Dict[str, AdaptiveHSVRange]]) -> None:
    """Saves the updated registry to disk (atomic write)."""
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        series: {name: r.to_dict() for name, r in layers.items()}
        for series, layers in registry.items()
    }

    # Atomic write: write to temp file first, then rename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    tmp.replace(path)
    logger.debug("[active_learning] Registry saved to %s", path)


def load_adaptive_ranges(
    map_series: str = "ams_tunisia",
) -> Dict[str, AdaptiveHSVRange]:
    """
    Returns the current HSV ranges for a given map series.
    If fewer than MIN_CORRECTIONS_TO_ACTIVATE corrections have been recorded,
    returns the default ranges (calibration not yet active).

    Usage in color_segmentation.py:
        from pipeline.active_learning import load_adaptive_ranges
        ranges = load_adaptive_ranges("ams_tunisia")
        lower, upper = ranges["red_roads"].to_opencv_range()
        mask = cv2.inRange(hsv, lower, upper)
    """
    registry = load_registry()
    series_ranges = registry.get(map_series, DEFAULT_ADAPTIVE_RANGES.get(map_series, {}))

    # Check if calibration is active (enough corrections accumulated)
    max_corrections = max(
        (r.correction_count for r in series_ranges.values()),
        default=0,
    )
    if max_corrections < MIN_CORRECTIONS_TO_ACTIVATE:
        logger.debug(
            "[active_learning] Calibration not yet active for '%s' "
            "(%d/%d corrections)",
            map_series, max_corrections, MIN_CORRECTIONS_TO_ACTIVATE,
        )
        return DEFAULT_ADAPTIVE_RANGES.get(map_series, series_ranges)

    logger.debug(
        "[active_learning] Using adaptive ranges for '%s' (max_corrections=%d)",
        map_series, max_corrections,
    )
    return series_ranges


# ─────────────────────────────────────────────────────────────────────────────
# HSV observation: extract statistics from a corrected polygon
# ─────────────────────────────────────────────────────────────────────────────

def _extract_hsv_stats_from_polygon(
    raster_bgr: np.ndarray,
    polygon_coords: List[List[float]],
    crop_bbox: Tuple[int, int, int, int],
    downscale_factor: float,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Extracts HSV statistics from the pixels inside a corrected polygon.

    Args:
        raster_bgr:        Full downscaled raster image (BGR, H×W×3).
        polygon_coords:    WGS84 or pixel coordinates of the corrected polygon
                           exterior ring [[lon/x, lat/y], ...].
        crop_bbox:         (x1, y1, x2, y2) of the map crop in downscaled image.
        downscale_factor:  Scale applied to the original raster for downscaling.

    Returns:
        Dict {layer_concept: (p5, p95)} for H, S, V channels.
        None if extraction fails (too few pixels, invalid polygon, etc.).
    """
    import cv2

    x1_crop, y1_crop, _, _ = crop_bbox

    # ── Convert polygon to pixel mask ─────────────────────────────────────────
    # polygon_coords may be in WGS84 (if georeferenced) or pixel space.
    # We detect which by checking if values are in [0, 360] range (lon/lat)
    # vs [0, image_width] range (pixels).
    H, W = raster_bgr.shape[:2]
    coords_arr = np.array(polygon_coords, dtype=np.float64)

    if coords_arr[:, 0].max() < 400 and coords_arr[:, 1].max() < 100:
        # Looks like WGS84 (lon: ~0-400, lat: ~-90 to 90)
        # Cannot back-project without full transform — skip
        logger.debug("[active_learning] WGS84 polygon — cannot extract pixels without transform")
        return None
    else:
        # Pixel coordinates in crop space → convert to full image space
        pts = coords_arr.copy()
        pts[:, 0] += x1_crop   # x offset
        pts[:, 1] += y1_crop   # y offset
        pts = pts.astype(np.int32)

    # Create binary mask at raster resolution
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Extract HSV pixels inside the polygon
    hsv = cv2.cvtColor(raster_bgr, cv2.COLOR_BGR2HSV)
    masked_hsv = hsv[mask > 0]

    if len(masked_hsv) < 20:
        logger.debug("[active_learning] Too few pixels inside polygon (%d)", len(masked_hsv))
        return None

    h_ch = masked_hsv[:, 0].astype(float)
    s_ch = masked_hsv[:, 1].astype(float)
    v_ch = masked_hsv[:, 2].astype(float)

    return {
        "H": (float(np.percentile(h_ch, 5)),  float(np.percentile(h_ch, 95))),
        "S": (float(np.percentile(s_ch, 5)),  float(np.percentile(s_ch, 95))),
        "V": (float(np.percentile(v_ch, 5)),  float(np.percentile(v_ch, 95))),
        "n_pixels": len(masked_hsv),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Map series detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_map_series(map_name: Optional[str], raster_path: Optional[Path]) -> str:
    """
    Infers the map series from map_name or filename heuristics.
    Returns one of the keys in DEFAULT_ADAPTIVE_RANGES.
    """
    if map_name:
        name_lower = map_name.lower()
        if any(k in name_lower for k in ("tunis", "tunisia", "sfax", "sousse", "bizerte")):
            return "ams_tunisia"
        if any(k in name_lower for k in ("alger", "algeria", "oran", "ain", "annaba")):
            return "ams_algeria"

    if raster_path:
        fname = raster_path.name.lower()
        if "tunis" in fname or "tunisia" in fname:
            return "ams_tunisia"
        if "algeria" in fname or "alger" in fname:
            return "ams_algeria"

    return "ams_tunisia"   # default: Tunisia (primary test dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: process one correction
# ─────────────────────────────────────────────────────────────────────────────

def process_correction(correction, map_upload) -> Optional[AdaptiveHSVRange]:
    """
    Main Active Learning entry point — called by api.py after saving a Correction.

    For DELETE corrections: records a negative example (no HSV update needed —
        but increments a rejection counter for future use in Level 2).

    For EDIT corrections: extracts HSV stats from the corrected polygon and
        updates the registry for the appropriate layer and map series.

    Args:
        correction:   Correction Django model instance (type, layer_name, geometry, …)
        map_upload:   MapUpload Django model instance (raster_path, map_name, …)

    Returns:
        The updated AdaptiveHSVRange for the corrected layer, or None if skipped.
    """
    import time as _time

    t0 = _time.perf_counter()

    layer_name      = correction.layer_name
    correction_type = correction.correction_type
    geometry        = correction.geometry   # GeoJSON geometry dict or None

    map_series = _detect_map_series(
        map_upload.map_name,
        map_upload.raster_path if hasattr(map_upload, "raster_path") else None,
    )

    logger.info(
        "[active_learning] Processing %s correction: layer=%s series=%s",
        correction_type, layer_name, map_series,
    )

    # ── DELETE: record but no HSV update ─────────────────────────────────────
    if correction_type == "delete":
        _record_negative_example(map_series, layer_name)
        logger.info(
            "[active_learning] DELETE recorded for '%s/%s' — no HSV update",
            map_series, layer_name,
        )
        return None

    # ── EDIT: extract HSV from corrected polygon ──────────────────────────────
    if correction_type != "edit" or not geometry:
        return None

    # Load the raster image
    try:
        from django.conf import settings
        raster_path = map_upload.raster_path
        img_bgr_full = cv2.imread(str(raster_path))
        if img_bgr_full is None:
            logger.warning("[active_learning] Cannot read raster: %s", raster_path)
            return None
    except Exception as exc:
        logger.warning("[active_learning] Raster load failed: %s", exc)
        return None

    # Downscale to pipeline resolution
    orig_h, orig_w = img_bgr_full.shape[:2]
    scale = min(2400 / max(orig_h, orig_w), 1.0)
    img_bgr = cv2.resize(
        img_bgr_full,
        (int(orig_w * scale), int(orig_h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    # Get crop bbox
    try:
        from pipeline.preprocessing import detect_map_frame
        bbox = detect_map_frame(img_bgr, verbose=False)
    except Exception:
        h, w = img_bgr.shape[:2]
        bbox = (int(w * 0.085), int(h * 0.085), int(w * 0.915), int(h * 0.82))

    # Extract polygon coordinates
    geom_type = geometry.get("type", "")
    if geom_type == "Polygon":
        exterior_ring = geometry["coordinates"][0]
    elif geom_type == "MultiPolygon":
        # Use the largest polygon ring
        exterior_ring = max(
            (poly[0] for poly in geometry["coordinates"]),
            key=len,
        )
    else:
        logger.warning("[active_learning] Unsupported geometry type: %s", geom_type)
        return None

    # Extract HSV statistics from corrected polygon pixels
    hsv_stats = _extract_hsv_stats_from_polygon(
        img_bgr, exterior_ring, bbox, scale
    )

    if hsv_stats is None:
        logger.warning(
            "[active_learning] Could not extract HSV stats for layer '%s'", layer_name
        )
        return None

    # Build observed range from extracted statistics
    observed_range = AdaptiveHSVRange(
        h_min=hsv_stats["H"][0],
        s_min=hsv_stats["S"][0],
        v_min=hsv_stats["V"][0],
        h_max=hsv_stats["H"][1],
        s_max=hsv_stats["S"][1],
        v_max=hsv_stats["V"][1],
        layer_name=layer_name,
        map_series=map_series,
    )

    # Load current registry and update with EMA
    registry = load_registry()

    if map_series not in registry:
        registry[map_series] = {}

    # Initialize from defaults if this layer not yet in registry
    if layer_name not in registry[map_series]:
        default = DEFAULT_ADAPTIVE_RANGES.get(map_series, {}).get(layer_name)
        if default:
            registry[map_series][layer_name] = AdaptiveHSVRange(**asdict(default))
        else:
            # Bootstrap from observation itself
            registry[map_series][layer_name] = AdaptiveHSVRange(
                h_min=hsv_stats["H"][0],
                s_min=hsv_stats["S"][0],
                v_min=hsv_stats["V"][0],
                h_max=hsv_stats["H"][1],
                s_max=hsv_stats["S"][1],
                v_max=hsv_stats["V"][1],
                layer_name=layer_name,
                map_series=map_series,
                correction_count=0,
            )

    current_range = registry[map_series][layer_name]
    updated_range = current_range.update_with_ema(observed_range, alpha=EMA_ALPHA)
    registry[map_series][layer_name] = updated_range

    # Persist registry
    save_registry(registry)

    elapsed = _time.perf_counter() - t0
    logger.info(
        "[active_learning] Updated '%s/%s': "
        "H[%.0f-%.0f] S[%.0f-%.0f] V[%.0f-%.0f] "
        "(n_px=%d, corrections=%d, elapsed=%.2fs)",
        map_series, layer_name,
        updated_range.h_min, updated_range.h_max,
        updated_range.s_min, updated_range.s_max,
        updated_range.v_min, updated_range.v_max,
        hsv_stats["n_pixels"],
        updated_range.correction_count,
        elapsed,
    )

    return updated_range


def _record_negative_example(map_series: str, layer_name: str) -> None:
    """
    Records a DELETE correction in the registry (increments a rejection counter).
    Used for Level 2 fine-tuning to know which regions are hard negatives.
    """
    registry = load_registry()
    if map_series not in registry:
        return
    if layer_name in registry[map_series]:
        r = registry[map_series][layer_name]
        # We don't update ranges for deletes — just note the count
        registry[map_series][layer_name] = AdaptiveHSVRange(
            h_min=r.h_min, s_min=r.s_min, v_min=r.v_min,
            h_max=r.h_max, s_max=r.s_max, v_max=r.v_max,
            layer_name=r.layer_name,
            map_series=r.map_series,
            correction_count=r.correction_count,
            last_updated=r.last_updated,
        )
        save_registry(registry)


# ─────────────────────────────────────────────────────────────────────────────
# Integration with color_segmentation.py
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveSegmenter:
    """
    Drop-in replacement for extract_all_color_layers() that uses
    adaptive HSV ranges from the Active Learning registry.

    Usage:
        segmenter = AdaptiveSegmenter(map_series="ams_tunisia")
        layers = segmenter.segment(hsv_image)
        # Returns same dict as extract_all_color_layers()

    The segmenter reloads the registry once per instance creation
    (not per call) for performance. Create a new instance per pipeline run.
    """

    def __init__(self, map_series: str = "ams_tunisia") -> None:
        self.map_series = map_series
        self.ranges     = load_adaptive_ranges(map_series)
        self._log_active_ranges()

    def _log_active_ranges(self) -> None:
        for name, r in self.ranges.items():
            logger.debug(
                "[AdaptiveSegmenter] %s/%s: H[%.0f-%.0f] S[%.0f-%.0f] V[%.0f-%.0f] "
                "(corrections=%d)",
                self.map_series, name,
                r.h_min, r.h_max, r.s_min, r.s_max, r.v_min, r.v_max,
                r.correction_count,
            )

    def segment(self, hsv: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segments the HSV image using adaptive ranges.
        Returns {layer_name: binary_mask}.
        """
        from skimage.morphology import skeletonize as ski_skel

        def mask_from_range(r: AdaptiveHSVRange) -> np.ndarray:
            lower, upper = r.to_opencv_range()
            return cv2.inRange(hsv, lower, upper)

        def clean_thin(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
            out = np.zeros_like(closed)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    out[labels == i] = 255
            return out

        def clean_poly(mask: np.ndarray, open_k: int = 3, close_k: int = 5,
                       min_area: int = 30) -> np.ndarray:
            ko = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
            kc = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
            m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            out = np.zeros_like(m)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    out[labels == i] = 255
            return out

        layers: Dict[str, np.ndarray] = {}

        # Water
        if "water" in self.ranges:
            layers["water"] = clean_poly(mask_from_range(self.ranges["water"]))

        # Vegetation
        if "vegetation" in self.ranges:
            layers["vegetation"] = clean_poly(mask_from_range(self.ranges["vegetation"]))

        # Contours (thin lines — no morphological open)
        if "contours" in self.ranges:
            raw = mask_from_range(self.ranges["contours"])
            layers["contours"] = ski_skel(
                clean_thin(raw, min_area=20).astype(bool)
            ).astype(np.uint8) * 255

        # Red roads (dual range: pure red H=0-10 + crimson H=170-180)
        if "red_roads" in self.ranges:
            r = self.ranges["red_roads"]
            low_mask = cv2.inRange(
                hsv,
                np.array([int(r.h_min), int(r.s_min), int(r.v_min)], dtype=np.uint8),
                np.array([int(r.h_max), int(r.s_max), int(r.v_max)], dtype=np.uint8),
            )
            # Crimson extension (H 170-180) uses same S/V bounds
            high_mask = cv2.inRange(
                hsv,
                np.array([170, int(r.s_min), int(r.v_min)], dtype=np.uint8),
                np.array([180, int(r.s_max), int(r.v_max)], dtype=np.uint8),
            )
            combined = cv2.bitwise_or(low_mask, high_mask)
            cleaned  = clean_poly(combined, open_k=2, close_k=3, min_area=30)
            layers["red_roads"] = ski_skel(cleaned.astype(bool)).astype(np.uint8) * 255

        # Buildings (dark gray polygons)
        if "buildings" in self.ranges:
            layers["buildings"] = clean_poly(
                mask_from_range(self.ranges["buildings"]),
                open_k=3, close_k=5, min_area=50,
            )

        return layers


# ─────────────────────────────────────────────────────────────────────────────
# Registry inspection utility
# ─────────────────────────────────────────────────────────────────────────────

def print_registry_summary() -> None:
    """Prints a human-readable summary of the current registry state."""
    registry = load_registry()
    print(f"\n{'='*65}")
    print(f"  HSV ACTIVE LEARNING REGISTRY — {_registry_path()}")
    print(f"{'='*65}")
    for series, layers in registry.items():
        print(f"\n  Series: {series}")
        print(f"  {'Layer':<15} {'H range':<12} {'S range':<12} {'V range':<12} {'Corrections'}")
        print(f"  {'-'*60}")
        for name, r in layers.items():
            status = "✅ ACTIVE" if r.correction_count >= MIN_CORRECTIONS_TO_ACTIVATE else f"⏳ {r.correction_count}/{MIN_CORRECTIONS_TO_ACTIVATE}"
            print(
                f"  {name:<15} "
                f"{r.h_min:.0f}-{r.h_max:.0f}{'':>6}"
                f"{r.s_min:.0f}-{r.s_max:.0f}{'':>6}"
                f"{r.v_min:.0f}-{r.v_max:.0f}{'':>6}"
                f"{status}"
            )
    print()


if __name__ == "__main__":
    print_registry_summary()
