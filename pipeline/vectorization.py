"""
Vectorisation : masque raster binaire → géométries vectorielles.

Utilise rasterio.features.shapes pour extraire les polygones,
puis Shapely pour simplifier et nettoyer.

Deux cibles de sortie :
    - polygones fermés (bâtiments, forêts, plans d'eau)
    - polylignes (routes, courbes de niveau) obtenues en squelettisant
      puis en traçant les segments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString, shape, mapping
from shapely.ops import unary_union, linemerge

try:
    from rasterio import features as rio_features
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def _require_rasterio() -> None:
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio est requis. pip install rasterio")


# ---------------------------------------------------------------------
# Polygones (bâtiments, zones végétation, plans d'eau)
# ---------------------------------------------------------------------
def mask_to_polygons(mask: np.ndarray, *,
                     transform: Optional["Affine"] = None,
                     min_area_px: int = 20,
                     simplify_tolerance_px: float = 1.5) -> List[Polygon]:
    """
    Convertit un masque binaire en liste de polygones Shapely.

    - transform : matrice affine rasterio pour géoréférencer (si None,
      les coordonnées restent en pixels).
    - min_area_px : rejette les polygones plus petits (unité = pixel²).
    - simplify_tolerance_px : simplification Douglas-Peucker.
    """
    _require_rasterio()

    binary = (mask > 0).astype(np.uint8)
    polygons: List[Polygon] = []

    # rasterio.features.shapes refuse transform=None — il faut soit ne pas
    # passer le paramètre, soit fournir Affine.identity() (1px = 1 unité monde).
    shapes_kwargs = {"mask": binary.astype(bool)}
    if transform is not None:
        shapes_kwargs["transform"] = transform

    for geom, val in rio_features.shapes(binary, **shapes_kwargs):
        if val != 1:
            continue
        poly = shape(geom)
        if not isinstance(poly, Polygon):
            continue
        if poly.area < min_area_px:
            continue
        poly = poly.simplify(simplify_tolerance_px, preserve_topology=True)
        polygons.append(poly)

    return polygons


# ---------------------------------------------------------------------
# Polylignes (routes, courbes) à partir d'un squelette
# ---------------------------------------------------------------------
def skeleton_to_lines(skeleton_mask: np.ndarray, *,
                      transform: Optional["Affine"] = None,
                      simplify_tolerance_px: float = 1.0) -> List[LineString]:
    """
    Convertit un masque squelettisé (lignes d'un pixel de large) en LineStrings.

    Méthode simple : on trace chaque segment entre pixels voisins, puis on
    fusionne avec shapely.ops.linemerge. Pour des résultats de production,
    remplacer par une traversée de graphe (ex. sknw, networkx).
    """
    ys, xs = np.where(skeleton_mask > 0)
    if len(xs) == 0:
        return []

    # Set de pixels actifs pour lookup rapide
    pixels = set(zip(xs.tolist(), ys.tolist()))
    segments = []
    # 8-voisinage
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def to_world(x, y):
        if transform is None:
            return (float(x), float(y))
        wx, wy = transform * (x + 0.5, y + 0.5)
        return (wx, wy)

    for (x, y) in pixels:
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (nx, ny) in pixels and (x, y) < (nx, ny):  # ordre → 1 seg par paire
                segments.append(LineString([to_world(x, y), to_world(nx, ny)]))

    if not segments:
        return []

    merged = linemerge(unary_union(segments))
    if isinstance(merged, LineString):
        lines = [merged]
    else:
        lines = list(merged.geoms)

    lines = [ln.simplify(simplify_tolerance_px, preserve_topology=False) for ln in lines]
    return lines


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
def to_geodataframe(geometries: Iterable, *,
                    layer_name: str,
                    crs: Optional[str] = None) -> gpd.GeoDataFrame:
    """Encapsule une liste de géométries dans un GeoDataFrame prêt à exporter."""
    geoms = list(geometries)
    gdf = gpd.GeoDataFrame(
        {"layer": [layer_name] * len(geoms)},
        geometry=geoms,
        crs=crs,
    )
    return gdf


# ---------------------------------------------------------------------
# Filtre par bbox : élimine les features qui touchent / chevauchent les
# bords du cadre (typiquement la légende, le titre, les notes en marge
# qui ont été détectés en couleur mais ne font pas partie de la carte).
# ---------------------------------------------------------------------
def filter_features_by_bbox(geometries,
                             bbox: tuple[float, float, float, float],
                             *,
                             margin: float = 0.0,
                             mode: str = "within"):
    """
    Filtre une liste de géométries (Polygon ou LineString) selon leur
    position par rapport à une bbox.

    Arguments :
        geometries : liste de Polygon / LineString.
        bbox       : (x1, y1, x2, y2) en pixels (ou unités monde si déjà
                     géoréférencé).
        margin     : retirer une marge intérieure (px). Une feature qui
                     touche la zone des `margin` premiers pixels près du
                     bord est considérée comme appartenant à la légende.
                     Pour le 1:50000 à 2400 px : margin=20-50 px.
        mode       : "within"     = ne garde que les géométries entièrement
                                     dans bbox+margin (strict).
                     "intersects" = garde dès qu'il y a intersection avec
                                     bbox+margin (laxiste).
                     "centroid"   = garde si le centroïde est dans bbox+margin
                                     (compromis recommandé pour les courbes
                                     de niveau qui peuvent dépasser un peu).

    Retourne la liste filtrée.
    """
    from shapely.geometry import box as shapely_box
    x1, y1, x2, y2 = bbox
    inner = shapely_box(x1 + margin, y1 + margin, x2 - margin, y2 - margin)

    out = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if mode == "within":
            keep = geom.within(inner)
        elif mode == "intersects":
            keep = geom.intersects(inner)
        elif mode == "centroid":
            keep = inner.contains(geom.centroid)
        else:
            raise ValueError(f"mode invalide : {mode}")
        if keep:
            out.append(geom)
    return out


def clip_features_to_bbox(geometries,
                           bbox: tuple[float, float, float, float]):
    """
    Coupe (clip) chaque géométrie à la bbox. Différent de
    `filter_features_by_bbox` qui rejette/accepte des géométries entières :
    ici on conserve les morceaux qui sont DANS la bbox et on jette le reste.

    Utile pour nettoyer les courbes de niveau qui dépassent légèrement le
    cadre, ou les polygones d'eau coupés par le neatline.
    """
    from shapely.geometry import box as shapely_box
    from shapely.geometry.base import BaseGeometry
    x1, y1, x2, y2 = bbox
    clip_box = shapely_box(x1, y1, x2, y2)

    out = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(clip_box)
        if clipped.is_empty:
            continue
        # Si l'intersection produit une MultiGeometry, on l'éclate
        if hasattr(clipped, "geoms"):
            for sub in clipped.geoms:
                if not sub.is_empty:
                    out.append(sub)
        elif isinstance(clipped, BaseGeometry):
            out.append(clipped)
    return out


def _detect_io_engine() -> str:
    """
    Choisit le backend I/O disponible pour geopandas.to_file().

    Sur Windows, pyogrio (le backend par défaut de geopandas >= 0.14) plante
    parfois avec "DLL load failed while importing lib" à cause de conflits
    de DLLs entre conda-forge et les paquets pip. Dans ce cas on bascule
    sur fiona qui est plus stable.

    Retourne "pyogrio" ou "fiona". Lève RuntimeError si aucun ne marche.
    """
    try:
        import pyogrio  # noqa: F401
        # Force le chargement de la C-extension pour detecter les DLL errors
        from pyogrio import _io  # noqa: F401
        return "pyogrio"
    except (ImportError, OSError):
        pass
    try:
        import fiona  # noqa: F401
        return "fiona"
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Aucun backend I/O disponible pour geopandas. "
            "Installe pyogrio OU fiona :\n"
            "  pip install pyogrio    (rapide, parfois cassé sur Windows)\n"
            "  pip install fiona      (plus lent mais fiable sur Windows)\n"
            f"Cause originale : {exc}"
        )


def save_geojson(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    """
    Sauvegarde en GeoJSON, avec fallback automatique pyogrio -> fiona si la
    DLL pyogrio est cassée (cas Windows fréquent).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = _detect_io_engine()
    try:
        gdf.to_file(path, driver="GeoJSON", engine=engine)
    except (OSError, ImportError) as exc:
        # Fallback : si on avait choisi pyogrio mais qu'il echoue à l'écriture
        if engine == "pyogrio":
            try:
                import fiona  # noqa: F401
                gdf.to_file(path, driver="GeoJSON", engine="fiona")
            except Exception:
                raise RuntimeError(
                    f"Echec ecriture GeoJSON (backend {engine}). "
                    f"Installe fiona en fallback : pip install fiona\n"
                    f"Cause : {exc}"
                ) from exc
        else:
            raise
    return path


def save_shapefile(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Sauvegarde en Shapefile (crée aussi les .shx, .dbf, .prj)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = _detect_io_engine()
    try:
        gdf.to_file(path, driver="ESRI Shapefile", engine=engine)
    except (OSError, ImportError) as exc:
        if engine == "pyogrio":
            try:
                import fiona  # noqa: F401
                gdf.to_file(path, driver="ESRI Shapefile", engine="fiona")
            except Exception:
                raise RuntimeError(
                    f"Echec ecriture Shapefile (backend {engine}). "
                    f"Cause : {exc}"
                ) from exc
        else:
            raise
    return path
