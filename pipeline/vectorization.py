"""
Vectorisation : masque raster binaire → géométries vectorielles.

Deux modes :
    - PIXEL (défaut, georeference=False) : les coordonnées GeoJSON restent
      en pixels image (X ∈ [0, W], Y ∈ [0, H]). C'est le mode obligatoire
      quand on veut superposer les vecteurs sur le raster d'origine dans
      Leaflet (CRS.Simple + ImageOverlay).
    - SIG   (georeference=True)         : on applique une transformation
      affine via pipeline/georeferencing.py pour passer en WGS84/EPSG:4326.
      Dépend de rasterio + GDAL — peut être indisponible sur poste Windows
      non configuré.

Deux cibles de sortie :
    - polygones fermés (bâtiments, forêts, plans d'eau)
    - polylignes (routes, courbes de niveau) obtenues en squelettisant
      puis en traçant les segments.

Robustesse :
    Les modules SIG (rasterio, fiona, pyogrio, geopandas) sont importés
    paresseusement — un poste sans GDAL peut toujours produire du GeoJSON
    pixel via json.dump direct.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Imports SIG paresseux — on les tente une seule fois et on note la dispo.
# Le mode pixel-only n'a besoin QUE de numpy + shapely (shapely est pur Python
# côté API publique, donc fiable même sans GDAL).
# ─────────────────────────────────────────────────────────────────────────────
try:
    from shapely.geometry import Polygon, LineString, shape, mapping  # noqa: F401
    from shapely.ops import unary_union, linemerge
    SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover — shapely est une dépendance dure
    SHAPELY_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except (ImportError, OSError):
    gpd = None  # type: ignore[assignment]
    GEOPANDAS_AVAILABLE = False

try:
    from rasterio import features as rio_features
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except (ImportError, OSError):
    rio_features = None  # type: ignore[assignment]
    Affine = None        # type: ignore[assignment]
    RASTERIO_AVAILABLE = False


def _require_rasterio() -> None:
    if not RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio est requis pour la vectorisation polygonale. "
            "Installe avec `pip install rasterio` (ou `conda install -c "
            "conda-forge rasterio` sur Windows)."
        )


def _require_geopandas() -> None:
    if not GEOPANDAS_AVAILABLE:
        raise ImportError(
            "geopandas est requis pour ce chemin d'export. "
            "En mode pixel pur (georeference=False), préfère "
            "masks_to_geojson() qui écrit du GeoJSON sans geopandas."
        )


# ---------------------------------------------------------------------
# Polygones (bâtiments, zones végétation, plans d'eau)
# ---------------------------------------------------------------------
def mask_to_polygons(mask: np.ndarray, *,
                     transform: Optional["Affine"] = None,
                     georeference: bool = False,
                     min_area_px: int = 20,
                     simplify_tolerance_px: float = 1.5) -> List[Polygon]:
    """
    Convertit un masque binaire en liste de polygones Shapely.

    Arguments
    ---------
    transform : matrice affine rasterio pour géoréférencer. Ignorée si
        georeference=False (mode pixel par défaut).
    georeference : True = applique la transformation SIG si fournie.
        False (défaut) = coordonnées en pixels image, X∈[0,W], Y∈[0,H].
    min_area_px : rejette les polygones plus petits que ce seuil (px²).
    simplify_tolerance_px : tolérance Douglas-Peucker (px).

    Implémentation : utilise rasterio.features.shapes si dispo, sinon
    bascule sur cv2.findContours + approxPolyDP. Le fallback OpenCV
    permet de produire du GeoJSON pixel même sur poste sans GDAL.

    Court-circuit : si le masque est None, vide ou intégralement zéro,
    retourne [] immédiatement — pas d'IndexError ni de crash rasterio.
    """
    if mask is None or mask.size == 0:
        return []
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return []   # masque entièrement vide (ex. carte du désert, pas d'eau)
    use_transform = transform if georeference else None

    if RASTERIO_AVAILABLE:
        return _mask_to_polygons_rasterio(
            binary, transform=use_transform,
            min_area_px=min_area_px,
            simplify_tolerance_px=simplify_tolerance_px,
        )

    # Fallback OpenCV — uniquement en pixels (pas de support transform ici,
    # car le mode SIG nécessite rasterio de toute façon).
    if use_transform is not None:
        raise RuntimeError(
            "Mode SIG demandé (transform fourni) mais rasterio est indisponible. "
            "Installe rasterio ou repasse en mode pixel (georeference=False)."
        )
    return _mask_to_polygons_cv2(
        binary,
        min_area_px=min_area_px,
        simplify_tolerance_px=simplify_tolerance_px,
    )


def _mask_to_polygons_rasterio(binary: np.ndarray, *,
                                transform: Optional["Affine"],
                                min_area_px: int,
                                simplify_tolerance_px: float) -> List["Polygon"]:
    """Implémentation rasterio.features.shapes."""
    polygons: List[Polygon] = []
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


def _mask_to_polygons_cv2(binary: np.ndarray, *,
                           min_area_px: int,
                           simplify_tolerance_px: float) -> List["Polygon"]:
    """
    Fallback OpenCV : cv2.findContours + Douglas-Peucker.
    Produit des polygones en coordonnées pixel uniquement.

    Robustesse : findContours peut renvoyer (None, _) sur certaines versions
    d'OpenCV anciennes, ou crasher en cv2.error sur masque corrompu.
    """
    import cv2
    try:
        result = cv2.findContours(
            (binary * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    except cv2.error:
        return []
    # OpenCV 3.x retourne (img, contours, hierarchy), 4.x retourne (contours, hierarchy)
    contours = result[-2] if len(result) >= 2 else []
    if contours is None:
        return []
    polygons: List[Polygon] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_tolerance_px, True)
        if len(approx) < 3:
            continue
        coords = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygons.append(Polygon(coords))
    return polygons


# ---------------------------------------------------------------------
# Polylignes (routes, courbes) à partir d'un squelette
# ---------------------------------------------------------------------
def skeleton_to_lines(skeleton_mask: np.ndarray, *,
                      transform: Optional["Affine"] = None,
                      georeference: bool = False,
                      simplify_tolerance_px: float = 1.0) -> List[LineString]:
    """
    Convertit un masque squelettisé (lignes d'un pixel de large) en LineStrings.

    Arguments
    ---------
    transform : matrice affine. Ignorée si georeference=False.
    georeference : True = applique la transform. False (défaut) = coords pixel.
    simplify_tolerance_px : tolérance Douglas-Peucker (px).

    Méthode simple : on trace chaque segment entre pixels voisins, puis on
    fusionne avec shapely.ops.linemerge. Pour des résultats de production,
    remplacer par une traversée de graphe (ex. sknw, networkx).

    Court-circuit : masque None / vide → liste vide sans crash.
    """
    if skeleton_mask is None or skeleton_mask.size == 0:
        return []
    ys, xs = np.where(skeleton_mask > 0)
    if len(xs) == 0:
        return []

    # Set de pixels actifs pour lookup rapide
    pixels = set(zip(xs.tolist(), ys.tolist()))
    segments = []
    # 8-voisinage
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    use_transform = transform if georeference else None

    def to_world(x, y):
        if use_transform is None:
            return (float(x), float(y))
        wx, wy = use_transform * (x + 0.5, y + 0.5)
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
                    crs: Optional[str] = None):
    """
    Encapsule une liste de géométries dans un GeoDataFrame prêt à exporter.

    Requiert geopandas. Pour le chemin "pixel seul" (sans GDAL),
    utilise plutôt masks_to_geojson() qui produit du JSON natif sans GDF.
    """
    _require_geopandas()
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

    Note : ce détecteur n'est utilisé QUE quand on passe par geopandas
    (chemin SIG). En mode pixel-only on écrit le JSON à la main et on
    n'appelle jamais cette fonction.

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


# =====================================================================
# Helper haut niveau : dict de masks -> dict de GeoJSON
# =====================================================================
def masks_to_geojson(masks: dict, *,
                      transform=None,
                      georeference: bool = False,
                      min_area_px: int = 30,
                      simplify_tolerance_px: float = 1.0) -> dict:
    """
    Convertit un dict {nom_couche: mask_uint8} en dict {nom_couche: geojson}.

    Arguments
    ---------
    transform : matrice affine. Ignorée si georeference=False.
    georeference : False (défaut) = coordonnées GeoJSON en pixels image.
        True = applique la transform pour passer en coords monde.
    min_area_px : seuil sur les polygones (px²).

    - Pour les masques de zones (water, vegetation, buildings) : polygones.
    - Pour les masques de lignes (contours, red_roads) : polylignes (squelette).

    Sortie : dict { layer_name: FeatureCollection } où chaque feature porte
    `properties.layer` et `properties.id`. Pas de dépendance geopandas.
    """
    line_layers = {"contours", "red_roads", "roads"}
    out = {}
    if not masks:
        return out
    for name, mask in masks.items():
        # On accepte que le pipeline upstream ait laissé une couche à None
        # ou à zéro pixels (ex. carte du désert : aucun pixel bleu détecté).
        if mask is None or getattr(mask, "size", 0) == 0:
            out[name] = {"type": "FeatureCollection", "features": []}
            continue
        try:
            if not mask.any():
                out[name] = {"type": "FeatureCollection", "features": []}
                continue
        except AttributeError:
            # mask n'est pas un ndarray — on log et on saute proprement
            out[name] = {"type": "FeatureCollection", "features": []}
            continue
        is_line = name in line_layers
        if is_line:
            geoms = skeleton_to_lines(
                mask,
                transform=transform,
                georeference=georeference,
                simplify_tolerance_px=simplify_tolerance_px,
            )
        else:
            geoms = mask_to_polygons(
                mask,
                transform=transform,
                georeference=georeference,
                min_area_px=min_area_px,
                simplify_tolerance_px=simplify_tolerance_px,
            )
        if not geoms:
            out[name] = {"type": "FeatureCollection", "features": []}
            continue
        try:
            features = []
            for idx, geom in enumerate(geoms):
                if geom is None or geom.is_empty:
                    continue
                features.append({
                    "type": "Feature",
                    "properties": {"layer": name, "id": int(idx)},
                    "geometry": mapping(geom),
                })
            out[name] = {"type": "FeatureCollection", "features": features}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"type": "FeatureCollection", "features": [], "error": str(exc)}
    return out


# =====================================================================
# Sauvegarde pixel-only — utilisée en mode georeference=False
# =====================================================================
def save_geojson_pixel(geojson_dict: dict, path: str | Path) -> Path:
    """
    Écrit un FeatureCollection (dict Python) en .geojson, sans geopandas.

    Conçue pour le mode pixel : pas besoin de GDAL, juste json.dump.
    Sert aussi de filet de sécurité quand pyogrio/fiona sont cassés.
    """
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2, ensure_ascii=False)
    return path
