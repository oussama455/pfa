"""
Pipeline simple CartoVec : version pédagogique pour démo PFA.

Objectif :
    carte scannée -> masques HSV -> GeoJSON

Cette version évite volontairement :
    - l'agent LangGraph ;
    - l'U-Net ;
    - l'active learning ;
    - le géoréférencement automatique.

Elle sert à expliquer clairement le cœur géomatique du projet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skimage.morphology import skeletonize

from . import color_segmentation as colseg
from . import preprocessing as prep
from . import vectorization as vec
from .pipeline import CALIBRATED_RANGES, _CALIBRATED_RED_HIGH, PipelineResult


ZONE_LAYERS = ("water", "vegetation", "buildings")
LINE_LAYERS = ("red_roads", "contours")


def _export_polygons(mask, layer_name: str, output_dir: Path) -> tuple[str | None, int]:
    """Convertit un masque de zone en polygones GeoJSON."""
    if mask is None:
        return None, 0
    polygons = vec.mask_to_polygons(mask, transform=None)
    if not polygons:
        return None, 0
    gdf = vec.to_geodataframe(polygons, layer_name=layer_name, crs=None)
    path = output_dir / f"{layer_name}.geojson"
    vec.save_geojson(gdf, path)
    return str(path), len(polygons)


def _export_lines(mask, layer_name: str, output_dir: Path) -> tuple[str | None, int]:
    """Convertit un masque linéaire en LineStrings GeoJSON."""
    if mask is None:
        return None, 0
    lines = vec.skeleton_to_lines(skeletonize(mask > 0), transform=None)
    if not lines:
        return None, 0
    gdf = vec.to_geodataframe(lines, layer_name=layer_name, crs=None)
    path = output_dir / f"{layer_name}.geojson"
    vec.save_geojson(gdf, path)
    return str(path), len(lines)


def run_simple_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    remove_legend: bool = True,
    verbose: bool = True,
) -> PipelineResult:
    """
    Version courte du pipeline : prétraitement, HSV, vectorisation.

    # Vérification obligatoire
    - image_bgr doit être H x W x 3 ;
    - image_hsv doit être H x W x 3 ;
    - chaque masque doit être uint8 avec valeurs 0/255.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[1/3] Prétraitement : {input_path.name}")
    image_bgr, image_hsv, _bbox = prep.preprocess_with_crop(
        input_path,
        auto_crop=True,
        remove_legend=remove_legend,
    )

    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3
    assert image_hsv.ndim == 3 and image_hsv.shape[2] == 3

    if verbose:
        h, w = image_bgr.shape[:2]
        print(f"      image utile : {w} x {h} px")

    if verbose:
        print("[2/3] Segmentation HSV")
    masks = colseg.extract_all_color_layers(
        image_hsv,
        ranges=CALIBRATED_RANGES,
        red_roads_high=_CALIBRATED_RED_HIGH,
    )

    if verbose:
        for name, mask in masks.items():
            assert mask.dtype.name == "uint8"
            print(f"      {name:12s}: {colseg.coverage_percent(mask):5.1f}%")

    if verbose:
        print("[3/3] Vectorisation GeoJSON")
    layers: dict[str, str] = {}
    counts: dict[str, int] = {}

    for name in ZONE_LAYERS:
        path, count = _export_polygons(masks.get(name), name, output_dir)
        if path:
            layers[name] = path
            counts[name] = count

    for name in LINE_LAYERS:
        path, count = _export_lines(masks.get(name), name, output_dir)
        if path:
            layers[name] = path
            counts[name] = count

    result = PipelineResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        layers=layers,
        counts=counts,
        georeferenced=False,
        legend_removed=remove_legend,
    )
    if verbose:
        print(json.dumps(counts, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline CartoVec simplifié")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="data/processed/simple")
    parser.add_argument("--keep-legend", action="store_true")
    args = parser.parse_args()

    run_simple_pipeline(
        args.input,
        args.output,
        remove_legend=not args.keep_legend,
    )


if __name__ == "__main__":
    main()
