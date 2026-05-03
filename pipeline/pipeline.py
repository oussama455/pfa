"""
Orchestrateur bout-en-bout du pipeline de vectorisation.

Modifications apportées :
    - Déplacement de 'red_roads' vers une vectorisation par squelettisation (lignes).
    - Optimisation de l'étape 5 pour séparer polygones et polylignes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from skimage.morphology import skeletonize  # Requis pour les routes rouges

from . import preprocessing as prep
from . import color_segmentation as colseg
from . import vectorization as vec
from . import georeferencing as geo


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
                 verbose: bool = True) -> PipelineResult:
    """
    Pipeline complet : raster → GeoJSON par couche.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prétraitement + recadrage cadre cartographique[cite: 5]
    if verbose:
        print(f"[1/5] Prétraitement : {input_path}")
    image_bgr, image_hsv, crop_bbox = prep.preprocess_with_crop(
        input_path,
        auto_crop=auto_crop,
        manual_bbox=manual_bbox,
    )
    if verbose:
        x1, y1, x2, y2 = crop_bbox
        print(f"       Cadre cartographique : ({x1}, {y1}) → ({x2}, {y2})")

    # 2) Segmentation par couleur[cite: 5]
    if verbose:
        print("[2/5] Segmentation par couleur (OpenCV)")
    color_masks = colseg.extract_all_color_layers(image_hsv)

    # 3) Segmentation sémantique (optionnelle)[cite: 5]
    semantic_masks: Dict[str, np.ndarray] = {}
    if with_semantic:
        from . import semantic_segmentation as semseg
        chosen_device = device if device else semseg.get_device(verbose=verbose)
        model = semseg.build_unet(encoder_name="resnet34", classes=3, device=chosen_device)
        if unet_weights:
            model = semseg.load_weights(model, unet_weights, device=chosen_device)
        image_rgb = prep.to_rgb(image_bgr)
        semantic_masks = semseg.predict_multi_class(
            model, image_rgb,
            class_names=["roads", "buildings"],
            device=chosen_device,
        )

    # 4) Géoréférencement[cite: 5]
    transform = None
    if gcps:
        if verbose:
            print(f"[4/5] Géoréférencement ({len(gcps)} GCPs)")
        transform = geo.compute_transform(gcps)

    # 5) Vectorisation et export[cite: 5]
    if verbose:
        print("[5/5] Vectorisation et export GeoJSON")
    layers_out: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    # --- A. POLYGONES (Eau, Végétation) ---
    # On a retiré 'red_roads' de cette liste[cite: 5]
    poly_layers = ["water", "vegetation"] 
    for name in poly_layers:
        mask = color_masks.get(name)
        if mask is None:
            continue
        polys = vec.mask_to_polygons(mask, transform=transform)
        if not polys:
            continue
        gdf = vec.to_geodataframe(polys, layer_name=name, crs=crs if transform else None)
        out_path = output_dir / f"{name}.geojson"
        vec.save_geojson(gdf, out_path)
        layers_out[name] = str(out_path)
        counts[name] = len(polys)

    # --- B. POLYLIGNES (Squelettisation) ---
    # Traitement spécifique pour les routes rouges
    red_road_mask = color_masks.get("red_roads")
    if red_road_mask is not None and red_road_mask.any():
        skeleton = skeletonize(red_road_mask > 0)
        lines = vec.skeleton_to_lines(skeleton, transform=transform)
        if lines:
            gdf = vec.to_geodataframe(lines, layer_name="red_roads", crs=crs if transform else None)
            out_path = output_dir / "red_roads.geojson"
            vec.save_geojson(gdf, out_path)
            layers_out["red_roads"] = str(out_path)
            counts["red_roads"] = len(lines)

    # Courbes de niveau (déjà gérées en squelette par colseg)[cite: 4, 5]
    contour_mask = color_masks.get("contours")
    if contour_mask is not None and contour_mask.any():
        lines = vec.skeleton_to_lines(contour_mask, transform=transform)
        if lines:
            gdf = vec.to_geodataframe(lines, layer_name="contours", crs=crs if transform else None)
            out_path = output_dir / "contours.geojson"
            vec.save_geojson(gdf, out_path)
            layers_out["contours"] = str(out_path)
            counts["contours"] = len(lines)

    # --- C. SÉMANTIQUE (Bâtiments, etc.) ---[cite: 5]
    for name, mask in semantic_masks.items():
        polys = vec.mask_to_polygons(mask, transform=transform)
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


def main():
    parser = argparse.ArgumentParser(description="Pipeline de vectorisation de cartes")
    parser.add_argument("input", help="Chemin vers la carte raster")
    parser.add_argument("-o", "--output", default="data/processed", help="Dossier de sortie")
    parser.add_argument("--semantic", action="store_true", help="Active U-Net")
    parser.add_argument("--weights", default=None, help="Poids U-Net (.pth)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.output,
        with_semantic=args.semantic,
        unet_weights=args.weights,
        device=None if args.device == "auto" else args.device,
    )


if __name__ == "__main__":
    main()