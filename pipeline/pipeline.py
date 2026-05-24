"""
Orchestrateur bout-en-bout du pipeline de vectorisation.

Nouveautés (branche feature/legend-detection-and-calibration) :
    - remove_legend=True  → suppression de la légende INTERNE (Stage 2)
      Testé sur 8 cartes réelles : Bizerte, Tunis, Aïn El Kseïba,
      Aïn Bessem, Alger, Terny, Warnier, Renault
    - Calibrated_hsv=True → plages HSV calibrées sur les 8 cartes réelles
      (voir dataset_config.json). Meilleure détection des routes rouges
      (S_min abaissé de 90→60) et de la végétation.

Usage Python :
    from pipeline.pipeline import run_pipeline
    result = run_pipeline("data/raw/carte.png", output_dir="data/processed")

Usage CLI :
    python -m pipeline.pipeline data/raw/carte.png -o data/processed
    python -m pipeline.pipeline data/raw/carte.png --no-remove-legend   # désactive le stage 2
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Sécurités CUDA & mémoire (DOIVENT être positionnées AVANT tout import torch)
# CUDA_LAUNCH_BLOCKING=1  : force la synchronisation pour que la stack trace
#                          pointe vraiment l'opération qui plante.
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 : limite la fragmentation
#                          du pool de mémoire CUDA — utile sur RTX 2050
#                          (4 Go VRAM) où les segments fragmentés finissent
#                          par lever OutOfMemoryError même quand il reste
#                          de la place "en théorie".
# Ces variables sont sans effet si CUDA n'est pas utilisé.
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import argparse
import gc
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from skimage.morphology import skeletonize

from . import preprocessing as prep
from . import color_segmentation as colseg
from . import vectorization as vec

# Le module georeferencing est importé paresseusement : sans GDAL il
# peut être indisponible, et en mode pixel-only on n'en a pas besoin.
try:
    from . import georeferencing as geo
    _GEOREFERENCING_AVAILABLE = True
except (ImportError, OSError) as _geo_exc:  # pragma: no cover
    geo = None  # type: ignore[assignment]
    _GEOREFERENCING_AVAILABLE = False
    _GEOREFERENCING_IMPORT_ERROR = _geo_exc


# ── HSV calibrées sur 8 cartes réelles (Tunisie + Algérie, 1:50 000) ─────────
# Testé sur : Bizerte (6.3% red), Tunis (9.2%), Aïn El Kseïba (6.5%),
#             Aïn Bessem (2.6%), Alger (2.7%), Terny (1.9%), Warnier (2.0%)
# Différences vs DEFAULT_RANGES d'origine :
#   red_roads   : S_min 90→60  (capture routes bordeaux plus pâles)
#   red_roads_h : S_min 90→60
#   vegetation  : S_min 40→20  (végétation verte délavée des cartes anciennes)
#                 v_min 50     → inchangé
CALIBRATED_RANGES: Dict[str, colseg.HSVRange] = {
    "water":      colseg.HSVRange(h_min=95,  s_min=60,  v_min=60,
                                   h_max=145, s_max=255, v_max=200),
    "vegetation": colseg.HSVRange(h_min=35,  s_min=20,  v_min=50,
                                   h_max=95,  s_max=255, v_max=220),
    "contours":   colseg.HSVRange(h_min=6,   s_min=40,  v_min=75,
                                   h_max=25,  s_max=160, v_max=215),
    "red_roads":  colseg.HSVRange(h_min=0,   s_min=60,  v_min=60,
                                   h_max=15,  s_max=255, v_max=255),
    "buildings":  colseg.HSVRange(h_min=0,   s_min=0,   v_min=30,
                                   h_max=180, s_max=50,  v_max=130),
}
_CALIBRATED_RED_HIGH = colseg.HSVRange(h_min=155, s_min=60, v_min=60,
                                        h_max=180, s_max=255, v_max=255)


# ════════════════════════════════════════════════════════════════════════════
# Adaptive QA thresholds based on detected map type
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveQAThresholds:
    """معايير QA ديناميكية حسب نوع الخريطة."""
    qa_threshold: float                    # دقة QA المستهدفة (%)
    max_allowed_layer_ratio: float         # أعلى نسبة تغطية طبقة واحدة (%)
    min_acceptable_confidence: float       # الحد الأدنى للثقة المقبولة (%)
    max_iterations: int                    # أقصى عدد تكرارات للتصحيح الذاتي


def get_adaptive_qa_thresholds(map_type: str) -> AdaptiveQAThresholds:
    """
    حساب معايير QA ديناميكية بناءً على نوع الخريطة المكتشف.
    
    منطق:
        - Monochrome/Faded: معايير مرنة (خرائط قديمة صعبة الفهم من الآلة)
        - Color/Rich: معايير صارمة (خرائط حديثة واضحة)
    """
    if map_type == "monochrome_faded":
        return AdaptiveQAThresholds(
            qa_threshold=74.0,              # الخرائط القديمة يصعب وصولها لـ 90%
            max_allowed_layer_ratio=91.0,   # تسامح أعلى مع عدم التوازن
            min_acceptable_confidence=45.0, # قبول ثقة منخفضة نسبياً
            max_iterations=3,               # تكرارات قليلة لتجنب الإفراط
        )
    else:  # color_rich
        return AdaptiveQAThresholds(
            qa_threshold=90.0,              # الخرائط الملونة الحديثة
            max_allowed_layer_ratio=85.0,   # معايير أصارم
            min_acceptable_confidence=55.0, # معايير أعلى
            max_iterations=5,               # تكرارات أكثر للوصول للجودة
        )


def calculate_layer_coverage_stats(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    حساب نسبة تغطية كل طبقة بالنسبة للصورة الكلية.
    
    Returns:
        dict: {layer_name: coverage_percent}
    """
    stats = {}
    for name, mask in masks.items():
        if mask is None:
            stats[name] = 0.0
        else:
            total_pixels = mask.size
            active_pixels = np.count_nonzero(mask)
            stats[name] = (active_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    return stats


def calculate_max_layer_ratio(coverage_stats: Dict[str, float]) -> float:
    """حساب أعلى نسبة تغطية بين الطبقات."""
    return max(coverage_stats.values()) if coverage_stats else 0.0


def calculate_confidence_score(coverage_stats: Dict[str, float], 
                              qa_threshold: float) -> float:
    """
    حساب درجة ثقة عامة بناءً على توازن الطبقات.
    
    منطق:
        - إذا كانت إحدى الطبقات تغطي >90% → ثقة منخفضة (عدم توازن = إفراط تقسيم)
        - إذا كانت الطبقات متوازنة → ثقة عالية
    """
    if not coverage_stats:
        return 0.0
    
    max_ratio = max(coverage_stats.values())
    avg_ratio = sum(coverage_stats.values()) / len(coverage_stats)
    
    # درجة توازن: كلما قل الفرق بين أعلى طبقة والمتوسط، كلما كانت الثقة أعلى
    balance_score = 100.0 - abs(max_ratio - avg_ratio)
    
    return max(0.0, min(100.0, balance_score))



def _apply_transform_to_geom(geom, transform):
    from shapely.ops import transform as shapely_transform
    def fn(xs, ys):
        new_xs, new_ys = [], []
        for x, y in zip(xs, ys):
            wx, wy = transform * (x, y)
            new_xs.append(wx); new_ys.append(wy)
        return new_xs, new_ys
    return shapely_transform(fn, geom)


@dataclass
class PipelineResult:
    input_path:    str
    output_dir:    str
    layers:        Dict[str, str]
    counts:        Dict[str, int]
    georeferenced: bool
    has_georeference: bool = False
    legend_removed: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _load_semap_config(project_root: Path) -> dict | None:
    """Load SEMAP config from unified data/config.json, with legacy fallback."""
    unified_cfg = project_root / "data" / "config.json"
    if unified_cfg.exists():
        cfg = json.loads(unified_cfg.read_text(encoding="utf-8"))
        semap_cfg = cfg.get("semap")
        if semap_cfg:
            return semap_cfg

    legacy_cfg = project_root / "data" / "semap_config.json"
    if legacy_cfg.exists():
        cfg = json.loads(legacy_cfg.read_text(encoding="utf-8"))
        if "num_classes" in cfg and "classes" in cfg:
            return cfg
    return None


def run_pipeline(input_path: str | Path,
                 output_dir: str | Path,
                 *,
                 georeference: bool = False,
                 gcps: Optional[List] = None,
                 crs: Optional[str] = "EPSG:4326",
                 with_semantic: bool = False,
                 unet_weights: Optional[str] = None,
                 auto_crop: bool = True,
                 remove_legend: bool = True,
                 manual_bbox: Optional[tuple] = None,
                 device: Optional[str] = None,
                 use_calibrated_hsv: bool = True,
                 filter_bbox_margin: int = 0,
                 filter_bbox_mode: str = "centroid",
                 verbose: bool = True) -> PipelineResult:
    """
    Pipeline complet : raster → GeoJSON par couche.

    Arguments :
        georeference       : MODE SIG. Défaut **False** — coordonnées en pixels
                             image, prêtes pour Leaflet.CRS.Simple + ImageOverlay.
                             True = applique la transformation affine (via GCPs
                             ou registre AMS) pour produire du WGS84/EPSG:4326.
        gcps               : Ground Control Points pour le géoréférencement.
                             Ignorés si georeference=False.
        remove_legend      : supprime la légende interne (Stage 2).
                             Défaut True — obligatoire sur cartes AMS/GSGS.
        use_calibrated_hsv : utilise les plages HSV calibrées sur 8 cartes réelles.
                             Défaut True — meilleure détection routes + végétation.
        filter_bbox_margin : pixels de marge aux bords du crop à ignorer.
                             Mis à 0 par défaut car remove_legend s'en occupe.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Garde-fou : si l'appelant demande le SIG mais que le module est KO,
    # on log un avertissement et on retombe en mode pixel.
    if georeference and not _GEOREFERENCING_AVAILABLE:
        if verbose:
            print("[!] georeference=True demandé mais georeferencing.py "
                  "indisponible — bascule en mode pixel.")
        georeference = False

    # 1) Prétraitement + recadrage (Stage 1 neatline + Stage 2 légende)
    if verbose:
        print(f"[1/5] Prétraitement : {input_path.name}")
    image_bgr, image_hsv, crop_bbox, detected_map_type = prep.preprocess_with_crop(
        input_path,
        auto_crop=auto_crop,
        remove_legend=remove_legend,
        manual_bbox=manual_bbox,
    )
    if verbose:
        x1, y1, x2, y2 = crop_bbox
        H_crop, W_crop = image_bgr.shape[:2]
        print(f"      Cadre cartographique : {W_crop}×{H_crop} px")
        print(f"      Légende supprimée    : {remove_legend}")
        print(f"      Type de carte détecté: {detected_map_type}")


    # ── Réalignement pixel : calcule offset (crop) + scale (downscale) ───────
    # En mode pixel, les masques sont en espace ROGNÉ + DOWNSCALÉ. Pour que
    # les vecteurs retombent pile sur l'image ORIGINALE affichée par Leaflet
    # (CRS.Simple), on applique : X_final = (X_mask + x1) * inv_scale.
    #   - offset = coin haut-gauche du crop (coords downscalées) = crop_bbox[:2]
    #   - inv_scale = 1 / facteur_downscale (≥ 1)
    pixel_offset = (float(crop_bbox[0]), float(crop_bbox[1]))
    pixel_scale = 1.0
    try:
        from PIL import Image
        with Image.open(input_path) as _im:
            _w_orig, _h_orig = _im.size
        _ds = prep.compute_downscale_scale(_w_orig, _h_orig)
        pixel_scale = (1.0 / _ds) if _ds > 0 else 1.0
        if verbose:
            print(f"      Réalignement pixel   : offset={pixel_offset}, "
                  f"scale={pixel_scale:.4f} (orig {_w_orig}×{_h_orig})")
    except Exception as _exc:  # noqa: BLE001
        # Pillow absent ou image illisible : on garde scale=1 (offset seul).
        # Correct pour les cartes non downscalées ; léger décalage sinon.
        if verbose:
            print(f"      Réalignement pixel   : offset={pixel_offset}, "
                  f"scale=1.0 (dims originales indisponibles : {_exc})")

    # 2) Segmentation couleur
    if verbose:
        print("[2/5] Segmentation par couleur")
    color_masks = colseg.extract_all_color_layers(
        image_hsv,
        ranges=CALIBRATED_RANGES if use_calibrated_hsv else None,
        red_roads_high=_CALIBRATED_RED_HIGH if use_calibrated_hsv else None,
    )
    if verbose:
        for name, mask in color_masks.items():
            print(f"      {name:12s} → {colseg.coverage_percent(mask):5.1f}%")
    
    # ── Calcul des statistiques QA adaptatives ──────────────────────────────
    # Basé sur le type de carte détecté, on charge les seuils appropriés
    qa_thresholds = get_adaptive_qa_thresholds(detected_map_type)
    layer_coverage = calculate_layer_coverage_stats(color_masks)
    max_layer_ratio = calculate_max_layer_ratio(layer_coverage)
    confidence_score = calculate_confidence_score(layer_coverage, qa_thresholds.qa_threshold)
    
    if verbose:
        print(f"[QA] Thresholds adaptés pour '{detected_map_type}':")
        print(f"      QA Target           : {qa_thresholds.qa_threshold:.1f}%")
        print(f"      Max Layer Ratio     : {qa_thresholds.max_allowed_layer_ratio:.1f}%")
        print(f"      Min Acceptable Conf : {qa_thresholds.min_acceptable_confidence:.1f}%")
        print(f"      Current Confidence  : {confidence_score:.1f}%")
        print(f"      Max Layer Dominance : {max_layer_ratio:.1f}%")

    # 3) Segmentation sémantique (optionnelle) — auto-detection des classes
    semantic_masks: Dict[str, np.ndarray] = {}
    if with_semantic:
        from . import semantic_segmentation as semseg
        chosen_device = device or semseg.get_device(verbose=verbose)
        if verbose:
            print(f"[3/5] Segmentation U-Net sur {chosen_device.upper()}")

        # Auto-detecte le nombre de classes : SEMAP (6) > historical_maps (5) > defaut (3)
        project_root = Path(__file__).resolve().parent.parent
        hm_cfg     = project_root / "data" / "historical_maps" / "classes.json"

        n_classes = 3
        class_names = ["background", "roads", "buildings"]
        semap_cfg = _load_semap_config(project_root)
        if semap_cfg and unet_weights and "semap" in str(unet_weights).lower():
            cfg = semap_cfg
            n_classes = cfg["num_classes"]
            class_names = [c["name"] for c in cfg["classes"]]
            if verbose:
                print(f"      Classes SEMAP : {class_names}")
        elif hm_cfg.exists():
            cfg = json.loads(hm_cfg.read_text(encoding="utf-8"))
            n_classes = cfg.get("num_classes", 5)
            class_names = [c["name"] for c in cfg["classes"]]
            if verbose:
                print(f"      Classes historical_maps : {class_names}")

        model = semseg.build_unet(encoder_name="resnet34", classes=n_classes,
                                   encoder_weights=None,
                                   device=chosen_device)
        if unet_weights:
            model = semseg.load_weights(model, unet_weights, device=chosen_device)
        all_masks = semseg.predict_multi_class(
            model, prep.to_rgb(image_bgr),
            class_names=class_names,
            device=chosen_device,
        )
        # Filtre les classes "fond" qui ne s'exportent pas en GeoJSON
        skip = {"background", "unknown", "non_built", "contours"}
        semantic_masks = {k: v for k, v in all_masks.items() if k not in skip}
    elif verbose:
        print("[3/5] Segmentation U-Net : sautée (with_semantic=False)")

    # 4) Géoréférencement — conditionnel
    transform = None
    if georeference and gcps and _GEOREFERENCING_AVAILABLE:
        if verbose:
            print(f"[4/5] Géoréférencement ({len(gcps)} GCPs)")
        transform = geo.compute_transform(gcps)
    elif verbose:
        if not georeference:
            print("[4/5] Géoréférencement : DÉSACTIVÉ (mode pixel — défaut)")
        elif not gcps:
            print("[4/5] Géoréférencement : sauté (pas de GCPs)")

    # 5) Vectorisation + export
    if verbose:
        mode = "WGS84" if transform is not None else "PIXEL"
        print(f"[5/5] Vectorisation et export GeoJSON ({mode})")

    H_crop, W_crop = image_hsv.shape[:2]
    crop_filter_bbox_px = (0, 0, W_crop, H_crop)

    def _maybe_filter(geoms, mode_override=None):
        if filter_bbox_margin <= 0 or not geoms or transform is not None:
            return geoms
        return vec.filter_features_by_bbox(
            geoms, crop_filter_bbox_px,
            margin=filter_bbox_margin,
            mode=mode_override or filter_bbox_mode,
        )

    def _save_layer(name: str, geoms: list, layer_type: str) -> None:
        """
        Sauvegarde une couche en GeoJSON. Deux chemins :
          - transform=None (mode pixel) → save_geojson_pixel(json.dump)
          - transform=Affine (mode SIG) → geopandas + save_geojson
        """
        if not geoms:
            return
        if transform is None:
            # ── Mode pixel : pas de geopandas, json direct ──────────────────
            # Réalignement crop+downscale -> image originale AVANT export.
            # Appliqué ICI (après _maybe_filter qui raisonne en espace crop),
            # pour ne pas fausser le filtrage par bbox.
            geoms = vec.apply_pixel_offset(
                geoms, offset=pixel_offset, scale=pixel_scale)
            features = []
            for idx, g in enumerate(geoms):
                if g is None or g.is_empty:
                    continue
                features.append({
                    "type": "Feature",
                    "properties": {"layer": name, "id": int(idx)},
                    "geometry": mapping(g),
                })
            gj = {"type": "FeatureCollection", "features": features}
            out = output_dir / f"{name}.geojson"
            vec.save_geojson_pixel(gj, out)
            layers_out[name] = str(out)
            counts[name] = len(features)
        else:
            # ── Mode SIG : geopandas requis ─────────────────────────────────
            transformed = [_apply_transform_to_geom(g, transform) for g in geoms]
            gdf = vec.to_geodataframe(transformed, layer_name=name, crs=crs)
            out = output_dir / f"{name}.geojson"
            vec.save_geojson(gdf, out)
            layers_out[name] = str(out)
            counts[name] = len(transformed)

    # Import shapely.geometry.mapping en local pour ne pas alourdir
    # le top-level si shapely venait à manquer.
    from shapely.geometry import mapping

    layers_out: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for name in ("water", "vegetation"):
        mask = color_masks.get(name)
        if mask is None: continue
        polys = vec.mask_to_polygons(mask, transform=None, georeference=False)
        polys = _maybe_filter(polys)
        _save_layer(name, polys, "polygon")

    for name in ("red_roads", "contours"):
        mask = color_masks.get(name)
        if mask is None or not mask.any(): continue
        skeleton = skeletonize(mask > 0)
        lines = vec.skeleton_to_lines(skeleton, transform=None, georeference=False)
        lines = _maybe_filter(lines, mode_override="centroid")
        _save_layer(name, lines, "linestring")

    for name, mask in semantic_masks.items():
        polys = vec.mask_to_polygons(mask, transform=None, georeference=False)
        polys = _maybe_filter(polys)
        _save_layer(name, polys, "polygon")

    # ── Libération mémoire après vectorisation ───────────────────────────────
    # Sur cartes lourdes, image_bgr + image_hsv + 5 masques color + semantic_masks
    # peuvent peser >500 Mo. On les jette explicitement avant de retourner.
    del color_masks, semantic_masks
    try:
        del image_bgr, image_hsv
    except NameError:
        pass
    gc.collect()

    result = PipelineResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        layers=layers_out,
        counts=counts,
        georeferenced=transform is not None,
        has_georeference=transform is not None,
        legend_removed=remove_legend,
    )
    if verbose:
        print("\n=== Terminé ===")
        print(result.to_json())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de vectorisation de cartes militaires")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="data/processed")
    parser.add_argument("--semantic",       action="store_true")
    parser.add_argument("--weights",        default=None)
    parser.add_argument("--no-auto-crop",   action="store_true")
    parser.add_argument("--no-remove-legend", action="store_true",
                        help="Désactive la suppression de la légende interne (Stage 2)")
    parser.add_argument("--no-calibrated-hsv", action="store_true",
                        help="Utilise les plages HSV génériques (non calibrées)")
    parser.add_argument("--georeference", action="store_true",
                        help="Active le géoréférencement (mode SIG). "
                             "Par défaut, sortie en coordonnées pixel image.")
    parser.add_argument("--bbox", type=int, nargs=4,
                        metavar=("X1","Y1","X2","Y2"), default=None)
    parser.add_argument("--device",
                        choices=["auto","cuda","cpu"], default="auto")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        args.input, args.output,
        georeference=args.georeference,
        with_semantic=args.semantic,
        unet_weights=args.weights,
        auto_crop=not args.no_auto_crop,
        remove_legend=not args.no_remove_legend,
        use_calibrated_hsv=not args.no_calibrated_hsv,
        manual_bbox=tuple(args.bbox) if args.bbox else None,
        device=None if args.device == "auto" else args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
