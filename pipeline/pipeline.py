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

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from skimage.morphology import skeletonize

from . import preprocessing as prep
from . import color_segmentation as colseg
from . import vectorization as vec
from . import georeferencing as geo


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
                 gcps: Optional[List[geo.GCP]] = None,
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

    # 1) Prétraitement + recadrage (Stage 1 neatline + Stage 2 légende)
    if verbose:
        print(f"[1/5] Prétraitement : {input_path.name}")
    image_bgr, image_hsv, crop_bbox = prep.preprocess_with_crop(
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

    # 4) Géoréférencement
    transform = None
    if gcps:
        if verbose:
            print(f"[4/5] Géoréférencement ({len(gcps)} GCPs)")
        transform = geo.compute_transform(gcps)
    elif verbose:
        print("[4/5] Géoréférencement : sautée (pas de GCPs)")

    # 5) Vectorisation + export
    if verbose:
        print("[5/5] Vectorisation et export GeoJSON")

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

    layers_out: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for name in ("water", "vegetation"):
        mask = color_masks.get(name)
        if mask is None: continue
        polys = vec.mask_to_polygons(mask, transform=None)
        polys = _maybe_filter(polys)
        if transform:
            polys = [_apply_transform_to_geom(p, transform) for p in polys]
        if not polys: continue
        gdf = vec.to_geodataframe(polys, layer_name=name,
                                   crs=crs if transform else None)
        out = output_dir / f"{name}.geojson"
        vec.save_geojson(gdf, out)
        layers_out[name] = str(out); counts[name] = len(polys)

    for name in ("red_roads", "contours"):
        mask = color_masks.get(name)
        if mask is None or not mask.any(): continue
        skeleton = skeletonize(mask > 0)
        lines = vec.skeleton_to_lines(skeleton, transform=None)
        lines = _maybe_filter(lines, mode_override="centroid")
        if transform:
            lines = [_apply_transform_to_geom(ln, transform) for ln in lines]
        if not lines: continue
        gdf = vec.to_geodataframe(lines, layer_name=name,
                                   crs=crs if transform else None)
        out = output_dir / f"{name}.geojson"
        vec.save_geojson(gdf, out)
        layers_out[name] = str(out); counts[name] = len(lines)

    for name, mask in semantic_masks.items():
        polys = vec.mask_to_polygons(mask, transform=None)
        polys = _maybe_filter(polys)
        if transform:
            polys = [_apply_transform_to_geom(p, transform) for p in polys]
        if not polys: continue
        gdf = vec.to_geodataframe(polys, layer_name=name,
                                   crs=crs if transform else None)
        out = output_dir / f"{name}.geojson"
        vec.save_geojson(gdf, out)
        layers_out[name] = str(out); counts[name] = len(polys)

    result = PipelineResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        layers=layers_out,
        counts=counts,
        georeferenced=transform is not None,
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
    parser.add_argument("--bbox", type=int, nargs=4,
                        metavar=("X1","Y1","X2","Y2"), default=None)
    parser.add_argument("--device",
                        choices=["auto","cuda","cpu"], default="auto")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        args.input, args.output,
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
