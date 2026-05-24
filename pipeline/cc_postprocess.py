"""
pipeline/cc_postprocess.py — Connected Components post-processing pipeline.

PURPOSE:
    Takes the binary mask output from U-Net and converts it into valid
    geospatial polygons using Connected Components (CC) labeling.

    WHY CC NOT WATERSHED:
        Watershed requires a distance transform + marker definition, which
        is sensitive to noisy military map backgrounds (scan artifacts,
        faded ink). For building footprints on AMS maps:
          - CC is deterministic and interpretable
          - CC works directly on the binary U-Net output (no gradient needed)
          - Military users can audit exactly which connected region produced
            which polygon — crucial for chain-of-custody in GIS analysis

PIPELINE:
    binary_mask (H×W uint8)
         │
         ▼
    morphological_clean()     — remove isolated noise pixels
         │
         ▼
    cc_label()                — cv2.connectedComponentsWithStats (8-connectivity)
         │
         ▼
    filter_by_area()          — reject too-small / too-large components
         │
         ▼
    component_to_polygon()    — rasterio.features.shapes per component
         │
         ▼
    simplify_and_validate()   — Douglas-Peucker + topology repair
         │
         ▼
    GeoDataFrame              — with CRS, layer attribute, CC stats
         │
    ┌────┴────┐
    ▼         ▼
 GeoJSON   Shapefile

USAGE:
    from pipeline.cc_postprocess import vectorize_mask

    gdf = vectorize_mask(
        mask=unet_output_mask,           # uint8 binary mask from U-Net
        layer_name="buildings",
        crs="EPSG:32632",               # Tunisian UTM zone 32N
        transform=rasterio_affine,       # optional: georeference
        output_geojson="out/buildings.geojson",
        output_shapefile="out/buildings.shp",
    )
    print(f"Extracted {len(gdf)} building polygons")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, mapping, shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    from rasterio import features as rio_features
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive min_area based on map type and layer
# ─────────────────────────────────────────────────────────────────────────────

def get_adaptive_min_area(layer_name: str, map_type: str = "color_rich") -> int:
    """
    حساب الحد الأدنى للمساحة (بالبكسل) لتجاهل الشوائب والضوضاء بشكل ديناميكي.
    
    المنطق:
        - للخرائط الباهتة (monochrome_faded): نرفع الفلتر لتجاهل أوساخ المسح الضوئي
        - لكل طبقة: حد أدنى مختلف بناءً على خصائصها الطبيعية
    
    Args:
        layer_name: اسم الطبقة (buildings, water, contours, etc.)
        map_type: نوع الخريطة ("color_rich" أو "monochrome_faded")
    
    Returns:
        الحد الأدنى للمساحة بالبكسل
    """
    # معامل الضرب بناءً على نوع الخريطة
    base_modifier = 2.0 if map_type == "monochrome_faded" else 1.0
    
    # الحد الأدنى المخصص لكل طبقة (في حالة الخرائط الملونة)
    layer_min_areas = {
        "buildings":  40,      # تجاهل الكتل الأصغر من مبنى صغير (جدران رقيقة)
        "contours":   150,     # الخطوط الكنتورية تحتاج مساحة اتصال كبيرة نسبياً
        "water":      60,      # الأودية والمسطحات المائية
        "vegetation": 100,     # بقع الغابات والمناطق الخضراء
        "roads":      30,      # الطرق تكون أرق، لكن حتى الطرق الدقيقة مهمة
    }
    
    # احصل على الحد الأدنى الأساسي، مع fallback افتراضي
    base_min_area = layer_min_areas.get(layer_name, 50)
    
    # تطبيق معامل الخريطة
    adaptive_min_area = int(base_min_area * base_modifier)
    
    logger.debug(
        f"[apply_adaptive_min_area] Layer '{layer_name}': "
        f"base={base_min_area} px × {base_modifier} ({map_type}) → {adaptive_min_area} px"
    )
    
    return adaptive_min_area


def apply_adaptive_min_area(layer_name: str, map_type: str = "color_rich") -> int:
    """Backward-compatible alias for older callers."""
    return get_adaptive_min_area(layer_name, map_type)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Morphological cleaning
# ─────────────────────────────────────────────────────────────────────────────

def morphological_clean(
    mask: np.ndarray,
    *,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
) -> np.ndarray:
    """
    Cleans a binary U-Net output mask before CC labeling.

    Operations (order matters):
        1. Morphological opening (erosion → dilation):
           Removes isolated noise pixels and disconnects weakly connected
           components. Kernel size 3 is conservative — enough to remove
           1-pixel artifacts without eroding thin wall structures.

        2. Morphological closing (dilation → erosion):
           Fills small holes within building footprints and reconnects
           minor breaks caused by scan artifacts (ink voids, fold lines).
           Kernel size 5 fills gaps up to 2 pixels wide.

    Args:
        mask:              Binary uint8 mask (0 or 255).
        open_kernel_size:  Opening kernel (default 3px — removes noise).
        close_kernel_size: Closing kernel (default 5px — fills holes).

    Returns:
        Cleaned binary mask (uint8, values 0 / 255).
    """
    binary = (mask > 0).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (open_kernel_size, open_kernel_size)
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
    )

    opened  = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Connected Components labeling
# ─────────────────────────────────────────────────────────────────────────────

def cc_label(
    mask: np.ndarray,
    *,
    connectivity: int = 8,
    min_area_px: int = 50,
    max_area_px: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Labels connected components using OpenCV's connectedComponentsWithStats.

    WHY 8-CONNECTIVITY:
        8-connectivity treats diagonal neighbors as connected, which correctly
        handles diagonal walls in building footprints (real buildings are not
        always axis-aligned). 4-connectivity would split diagonal corners into
        separate components.

    Args:
        mask:           Binary uint8 mask (0/255).
        connectivity:   4 or 8 (default 8).
        min_area_px:    Minimum component area to keep (pixels²).
                        Default 50 px at 2400-px image scale ≈ 3×3 m at 1:50k.
        max_area_px:    Maximum component area (None = no upper limit).
                        Use to filter out the entire map background.

    Returns:
        labeled_mask:   uint32 array where each pixel = its component label (0 = bg).
        component_info: List of dicts with stats for each valid component.
    """
    binary = (mask > 0).astype(np.uint8)

    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )

    component_info: List[Dict] = []

    for label_id in range(1, num_labels):   # skip 0 = background
        area   = int(stats[label_id, cv2.CC_STAT_AREA])
        x      = int(stats[label_id, cv2.CC_STAT_LEFT])
        y      = int(stats[label_id, cv2.CC_STAT_TOP])
        width  = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])

        if area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue

        component_info.append({
            "label_id": label_id,
            "area_px":  area,
            "bbox_px":  (x, y, x + width, y + height),
            "centroid_px": (cx, cy),
            "aspect_ratio": width / max(height, 1),
        })

    logger.debug(
        "[cc_label] %d labels found, %d valid (min_area=%d px)",
        num_labels - 1, len(component_info), min_area_px,
    )

    return label_map, component_info


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Component → Polygon conversion
# ─────────────────────────────────────────────────────────────────────────────

def component_to_polygon(
    label_map: np.ndarray,
    label_id: int,
    *,
    transform: Optional["Affine"] = None,
    simplify_tolerance: float = 1.5,
) -> Optional[Polygon]:
    """
    Converts a single connected component to a Shapely Polygon.

    Method: rasterio.features.shapes on a single-component binary mask.
    This is more accurate than cv2.findContours for geo-applications because
    rasterio correctly handles the pixel-center vs. pixel-corner convention.

    Args:
        label_map:          CC label array from cc_label().
        label_id:           Which component to convert.
        transform:          rasterio Affine for georeferencing (None = pixel coords).
        simplify_tolerance: Douglas-Peucker tolerance in pixel units.
                            1.5 px at 2400-px scale ≈ 3 m at 1:50,000.

    Returns:
        A valid Shapely Polygon, or None if conversion fails.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required. conda install -c conda-forge rasterio")

    # Create single-component binary mask
    single = ((label_map == label_id)).astype(np.uint8)

    polygons: List[Polygon] = []

    # rasterio.features.shapes extracts the polygon boundary of the mask
    shape_kwargs: Dict = {}
    if transform is not None:
        shape_kwargs["transform"] = transform

    for geom_dict, value in rio_features.shapes(single, mask=single, **shape_kwargs):
        if value != 1:
            continue
        geom = shape(geom_dict)
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue
        polygons.append(geom)

    if not polygons:
        return None

    # Merge all sub-polygons (usually just one, but handles donut shapes)
    merged = unary_union(polygons)

    # Topology repair (handles self-intersections from noisy mask borders)
    if not merged.is_valid:
        merged = make_valid(merged)

    # Simplify boundary to reduce vertex count (important for WebGIS performance)
    simplified = merged.simplify(simplify_tolerance, preserve_topology=True)

    # Ensure we still have a polygon after simplification
    if simplified.is_empty or not isinstance(simplified, (Polygon, MultiPolygon)):
        return None

    return simplified


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Full pipeline: mask → GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_geodataframe(
    mask: np.ndarray,
    layer_name: str,
    *,
    crs: str = "EPSG:4326",
    transform: Optional["Affine"] = None,
    min_area_px: Optional[int] = None,
    max_area_px: Optional[int] = None,
    simplify_tolerance: float = 1.5,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
    map_type: str = "color_rich",
) -> "gpd.GeoDataFrame":
    """
    Complete CC-based pipeline: binary mask → GeoDataFrame.
    
    Adaptive filtering based on map_type (detected from preprocessing).
    
    Parameters:
        map_type: "color_rich" (default) أو "monochrome_faded"
                  يؤثر على حساب min_area_px التلقائي
    """
    # حساب min_area_px تلقائياً من map_type إذا لم يتم تحديده
    if min_area_px is None:
        min_area_px = get_adaptive_min_area(layer_name, map_type=map_type)

    """
    Args:
        mask:            Binary uint8 mask from U-Net (H×W, values 0/255).
        layer_name:      Semantic name: "buildings", "roads", etc.
        crs:             Target CRS. For Tunisia:
                           "EPSG:4326"  — WGS84 geographic (if transform provides lon/lat)
                           "EPSG:32632" — UTM Zone 32N (metric, recommended for analysis)
                           "EPSG:22332" — Tunisia Mapping Grid (legacy AMS)
        transform:       rasterio Affine matrix from map georeferencing.
                         If None, output coordinates are in pixels.
        min_area_px:     Minimum component area in pixels.
        max_area_px:     Maximum component area (None = unlimited).
        simplify_tolerance: Douglas-Peucker simplification tolerance (pixels or CRS units).
        open_kernel_size:  Morphological opening kernel size.
        close_kernel_size: Morphological closing kernel size.

    Returns:
        GeoDataFrame with columns:
            geometry    — Polygon or MultiPolygon
            layer       — layer_name
            area_px     — original area in pixels
            centroid_x  — centroid x in pixel space
            centroid_y  — centroid y in pixel space
            label_id    — CC label (for debugging / human review)
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError(
            "geopandas is required. conda install -c conda-forge geopandas"
        )

    # ── Step 1: Clean the mask ────────────────────────────────────────────────
    cleaned = morphological_clean(
        mask,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
    )

    # ── Step 2: CC labeling ───────────────────────────────────────────────────
    label_map, component_info = cc_label(
        cleaned,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
    )

    if not component_info:
        logger.warning("[cc_postprocess] No valid components found in '%s' layer", layer_name)
        return gpd.GeoDataFrame(
            {"layer": [], "area_px": [], "centroid_x": [], "centroid_y": [], "label_id": []},
            geometry=[],
            crs=crs,
        )

    # ── Step 3: Convert each component to a polygon ───────────────────────────
    rows = []
    for comp in component_info:
        poly = component_to_polygon(
            label_map,
            comp["label_id"],
            transform=transform,
            simplify_tolerance=simplify_tolerance,
        )
        if poly is None or poly.is_empty:
            continue

        rows.append({
            "geometry":   poly,
            "layer":      layer_name,
            "area_px":    comp["area_px"],
            "centroid_x": comp["centroid_px"][0],
            "centroid_y": comp["centroid_px"][1],
            "label_id":   comp["label_id"],
        })

    if not rows:
        logger.warning("[cc_postprocess] All components failed polygon conversion for '%s'", layer_name)
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(rows, crs=crs)

    logger.info(
        "[cc_postprocess] '%s': %d polygons from %d CC components (min_area=%d px)",
        layer_name, len(gdf), len(component_info), min_area_px,
    )

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Export helpers (DEDUPLICATED : on reutilise pipeline.vectorization)
# ─────────────────────────────────────────────────────────────────────────────

# Re-export depuis pipeline.vectorization pour eviter la duplication.
# Les fonctions sont identiques mais centralisees la-bas (avec fallback
# pyogrio -> fiona, gestion des erreurs DLL Windows, etc.).
from pipeline.vectorization import (  # noqa: F401
    _detect_io_engine,
    save_geojson,
    save_shapefile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def vectorize_mask(
    mask: np.ndarray,
    layer_name: str,
    *,
    crs: str = "EPSG:4326",
    transform: Optional["Affine"] = None,
    output_geojson: Optional[str | Path] = None,
    output_shapefile: Optional[str | Path] = None,
    min_area_px: Optional[int] = None,
    max_area_px: Optional[int] = None,
    simplify_tolerance: float = 1.5,
    map_type: str = "color_rich",
) -> "gpd.GeoDataFrame":
    """
    Public API: binary mask → GeoDataFrame + optional file export.

    This is the function to call from the agent or CLI:

        from pipeline.cc_postprocess import vectorize_mask

        gdf = vectorize_mask(
            mask=unet_binary_output,
            layer_name="buildings",
            crs="EPSG:32632",          # Tunisian UTM 32N for metric analysis
            transform=affine,          # from georeferencing.py
            output_geojson="out/buildings.geojson",
            output_shapefile="out/buildings.shp",
            map_type="monochrome_faded",  # auto-detected from preprocessing
        )

    Args:
        mask:             Binary uint8 mask (0/255) from U-Net or HSV segmentation.
        layer_name:       Semantic name ("buildings", "roads", "vegetation", …).
        crs:              CRS string for the GeoDataFrame and output files.
                          Use "EPSG:4326" if transform provides lon/lat coords.
                          Use "EPSG:32632" (UTM 32N) for metric Tunisia analysis.
        transform:        rasterio Affine matrix. If None, coords stay in pixels.
        output_geojson:   If set, write GeoJSON to this path.
        output_shapefile: If set, write Shapefile to this path.
        min_area_px:      Minimum CC area in pixels. If None, auto-calculated based
                          on map_type and layer_name.
        max_area_px:      Maximum CC area (None = no limit).
        simplify_tolerance: Simplification tolerance (pixels or map units).
        map_type:         "color_rich" (default) or "monochrome_faded" for adaptive filtering.

    Returns:
        GeoDataFrame with geometry, layer, area_px, centroid_x/y, label_id.
    """
    # حساب min_area_px تلقائياً إذا لم يتم تحديده
    if min_area_px is None:
        min_area_px = get_adaptive_min_area(layer_name, map_type=map_type)
    
    gdf = mask_to_geodataframe(
        mask,
        layer_name,
        crs=crs,
        transform=transform,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        simplify_tolerance=simplify_tolerance,
        map_type=map_type,
    )

    if output_geojson:
        save_geojson(gdf, output_geojson)

    if output_shapefile:
        save_shapefile(gdf, output_shapefile)

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example CLI usage:
        python -m pipeline.cc_postprocess \
            --mask data/processed/map_42/buildings_mask.png \
            --layer buildings \
            --crs EPSG:32632 \
            --output-geojson data/processed/map_42/buildings.geojson \
            --output-shp data/processed/map_42/buildings.shp
    """
    import argparse
    import cv2 as _cv2

    parser = argparse.ArgumentParser(description="CC-based mask vectorization")
    parser.add_argument("--mask",             required=True, help="Path to binary mask PNG")
    parser.add_argument("--layer",            default="features", help="Layer name")
    parser.add_argument("--crs",              default="EPSG:4326", help="Output CRS")
    parser.add_argument("--output-geojson",   default=None)
    parser.add_argument("--output-shp",       default=None)
    parser.add_argument("--min-area",         type=int, default=None)
    parser.add_argument("--map-type",         default="color_rich",
                        choices=["color_rich", "monochrome_faded"])
    args = parser.parse_args()

    mask_img = _cv2.imread(args.mask, _cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    result_gdf = vectorize_mask(
        mask=mask_img,
        layer_name=args.layer,
        crs=args.crs,
        output_geojson=args.output_geojson,
        output_shapefile=args.output_shp,
        min_area_px=args.min_area,
        map_type=args.map_type,
    )
    print(f"Extracted {len(result_gdf)} features from '{args.layer}'")
    if not result_gdf.empty:
        print(result_gdf[["layer", "area_px", "centroid_x", "centroid_y"]].head(10).to_string())
