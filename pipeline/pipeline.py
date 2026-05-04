"""
Orchestrateur bout-en-bout du pipeline de vectorisation.

Modifications apportées :
    - Déplacement de 'red_roads' vers une vectorisation par squelettisation (lignes).
    - Optimisation de l'étape 5 pour séparer polygones et polylignes.

Usage typique (Python) :
    from pipeline.pipeline import run_pipeline
    result = run_pipeline("data/raw/carte.png", output_dir="data/processed")

Usage CLI :
    python -m pipeline.pipeline data/raw/carte.png -o data/processed
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


def _apply_transform_to_geom(geom, transform):
    """
    Applique une transformation affine rasterio aux coordonnées d'une
    géométrie Shapely. Utilisé après filter_features_by_bbox (qui doit
    travailler en pixels) pour reprojeter en coords monde.
    """
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
    """Résumé de l'exécution du pipeline."""
    input_path: str
    output_dir: str
    layers: Dict[str, str]       # {nom_couche: chemin_geojson}
    counts: Dict[str, int]       # {nom_couche: nb_features}
    georeferenced: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def run_pipeline(input_path: str | Path,
                 output_dir: str | Path,
                 *,
                 gcps: Optional[List[geo.GCP]] = None,
                 crs: Optional[str] = "EPSG:4326",
                 with_semantic: bool = False,
                 unet_weights: Optional[str] = None,
                 auto_crop: bool = True,
                 manual_bbox: Optional[tuple] = None,
                 device: Optional[str] = None,
                 filter_bbox_margin: int = 20,
                 filter_bbox_mode: str = "centroid",
                 verbose: bool = True) -> PipelineResult:
    """
    Pipeline complet : raster → GeoJSON par couche.

    Arguments :
        input_path : carte raster (PNG, TIFF, JPG).
        output_dir : dossier de sortie pour les GeoJSON.
        gcps       : points de contrôle pour géoréférencer (optionnel).
        crs        : code EPSG des coordonnées des GCPs.
        with_semantic : si True, ajoute la segmentation U-Net
                        (nécessite pipeline.semantic_segmentation + poids).
        unet_weights : chemin vers les poids U-Net (.pth).
        auto_crop  : détecte et recadre automatiquement la zone cartographique
                     (exclut la légende et les marges). Défaut : True.
        manual_bbox : (x1, y1, x2, y2) pour forcer un recadrage manuel.
                      Prime sur auto_crop si fourni.
        device     : 'cuda', 'cpu' ou None (auto). Si None, GPU si dispo
                     sinon CPU. Utilise device='cpu' pour comparer / debug.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prétraitement + recadrage cadre cartographique
    if verbose:
        print(f"[1/5] Prétraitement : {input_path}")
    image_bgr, image_hsv, crop_bbox = prep.preprocess_with_crop(
        input_path,
        auto_crop=auto_crop,
        manual_bbox=manual_bbox,
    )
    if verbose:
        x1, y1, x2, y2 = crop_bbox
        print(f"       Cadre cartographique : ({x1}, {y1}) → ({x2}, {y2}) "
              f"= {x2 - x1} × {y2 - y1} px")

    # 2) Segmentation par couleur
    if verbose:
        print("[2/5] Segmentation par couleur (OpenCV)")
    color_masks = colseg.extract_all_color_layers(image_hsv)
    if verbose:
        for name, mask in color_masks.items():
            print(f"       {name:12s} → {colseg.coverage_percent(mask):5.2f}% des pixels")

    # 3) Segmentation sémantique (optionnelle) — auto GPU sinon CPU
    semantic_masks: Dict[str, np.ndarray] = {}
    if with_semantic:
        from . import semantic_segmentation as semseg  # import lazy
        if device is None:
            chosen_device = semseg.get_device(verbose=verbose)
        else:
            chosen_device = device
            if verbose:
                print(f"  Device forcé : {chosen_device}")
        if verbose:
            print(f"[3/5] Segmentation sémantique (U-Net) sur {chosen_device.upper()}")
        model = semseg.build_unet(encoder_name="resnet34", classes=3,
                                  device=chosen_device)
        if unet_weights:
            model = semseg.load_weights(model, unet_weights, device=chosen_device)
        image_rgb = prep.to_rgb(image_bgr)
        semantic_masks = semseg.predict_multi_class(
            model, image_rgb,
            class_names=["roads", "buildings"],
            device=chosen_device,
        )
    elif verbose:
        print("[3/5] Segmentation sémantique : sautée (with_semantic=False)")

    # 4) Géoréférencement
    transform = None
    if gcps:
        if verbose:
            print(f"[4/5] Géoréférencement ({len(gcps)} GCPs)")
        transform = geo.compute_transform(gcps)
    elif verbose:
        print("[4/5] Géoréférencement : sautée (pas de GCPs)")

    # 5) Vectorisation et export
    if verbose:
        print("[5/5] Vectorisation et export GeoJSON")
    layers_out: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    # bbox de filtrage (en pixels DU CROP) : on rejette les features qui
    # touchent les `margin` premiers pixels près du bord (souvent légende
    # résiduelle après auto_crop imparfait).
    H_crop, W_crop = image_hsv.shape[:2]
    crop_filter_bbox_px = (0, 0, W_crop, H_crop)
    if filter_bbox_margin > 0 and verbose:
        print(f"       Filtre bord : margin={filter_bbox_margin}px, mode={filter_bbox_mode}")

    def _maybe_filter(geoms, mode_override=None):
        """Applique filter_features_by_bbox si margin > 0."""
        if filter_bbox_margin <= 0 or not geoms:
            return geoms
        # Si geoms sont en monde (transform fourni), il faut filter en monde.
        # Pour simplifier on filtre en pixels AVANT la transform — donc on
        # a re-fait mask_to_polygons sans transform plus haut. Ici geoms
        # peuvent être déjà en monde si transform != None ; on saute alors.
        if transform is not None:
            return geoms
        return vec.filter_features_by_bbox(
            geoms, crop_filter_bbox_px,
            margin=filter_bbox_margin,
            mode=mode_override or filter_bbox_mode,
        )

    # --- A. POLYGONES (Eau, Végétation) ---
    poly_layers = ["water", "vegetation"]
    for name in poly_layers:
        mask = color_masks.get(name)
        if mask is None:
            continue
        # On extrait d'abord en pixels (sans transform) pour pouvoir filtrer
        polys = vec.mask_to_polygons(mask, transform=None)
        polys = _maybe_filter(polys)
        if transform is not None:
            # Re-projeter en coords monde après filtrage
            polys = [_apply_transform_to_geom(p, transform) for p in polys]
        if not polys:
            continue
        gdf = vec.to_geodataframe(polys, layer_name=name, crs=crs if transform else None)
        out_path = output_dir / f"{name}.geojson"
        vec.save_geojson(gdf, out_path)
        layers_out[name] = str(out_path)
        counts[name] = len(polys)

    # --- B. POLYLIGNES par squelettisation (Routes rouges + Courbes de niveau) ---
    red_road_mask = color_masks.get("red_roads")
    if red_road_mask is not None and red_road_mask.any():
        skeleton = skeletonize(red_road_mask > 0)
        lines = vec.skeleton_to_lines(skeleton, transform=None)
        lines = _maybe_filter(lines, mode_override="centroid")
        if transform is not None:
            lines = [_apply_transform_to_geom(ln, transform) for ln in lines]
        if lines:
            gdf = vec.to_geodataframe(lines, layer_name="red_roads", crs=crs if transform else None)
            out_path = output_dir / "red_roads.geojson"
            vec.save_geojson(gdf, out_path)
            layers_out["red_roads"] = str(out_path)
            counts["red_roads"] = len(lines)

    contour_mask = color_masks.get("contours")
    if contour_mask is not None and contour_mask.any():
        lines = vec.skeleton_to_lines(contour_mask, transform=None)
        lines = _maybe_filter(lines, mode_override="centroid")
        if transform is not None:
            lines = [_apply_transform_to_geom(ln, transform) for ln in lines]
        if lines:
            gdf = vec.to_geodataframe(lines, layer_name="contours",
                                      crs=crs if transform else None)
            out_path = output_dir / "contours.geojson"
            vec.save_geojson(gdf, out_path)
            layers_out["contours"] = str(out_path)
            counts["contours"] = len(lines)

    # --- C. SÉMANTIQUE (Bâtiments, Routes IA) ---
    for name, mask in semantic_masks.items():
        polys = vec.mask_to_polygons(mask, transform=None)
        polys = _maybe_filter(polys)
        if transform is not None:
            polys = [_apply_transform_to_geom(p, transform) for p in polys]
        if not polys:
            continue
        gdf = vec.to_geodataframe(polys, layer_name=name, crs=crs if transform else None)
        out_path = output_dir / f"{name}.geojson"
        vec.save_geojson(gdf, out_path)
        layers_out[name] = str(out_path)
        counts[name] = len(polys)

    result = PipelineResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        layers=layers_out,
        counts=counts,
        georeferenced=transform is not None,
    )
    if verbose:
        print("\n=== Terminé ===")
        print(result.to_json())
    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pipeline de vectorisation de cartes")
    parser.add_argument("input", help="Chemin vers la carte raster (PNG/TIF/JPG)")
    parser.add_argument("-o", "--output", default="data/processed",
                        help="Dossier de sortie (défaut : data/processed)")
    parser.add_argument("--semantic", action="store_true",
                        help="Active la segmentation sémantique U-Net")
    parser.add_argument("--weights", default=None,
                        help="Chemin vers les poids U-Net (.pth)")
    parser.add_argument("--no-auto-crop", action="store_true",
                        help="Désactive la détection auto du cadre cartographique")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        default=None,
                        help="Recadrage manuel : x1 y1 x2 y2 (en pixels)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device pour la segmentation U-Net "
                             "(auto = GPU si dispo sinon CPU)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.output,
        with_semantic=args.semantic,
        unet_weights=args.weights,
        auto_crop=not args.no_auto_crop,
        manual_bbox=tuple(args.bbox) if args.bbox else None,
        device=None if args.device == "auto" else args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
