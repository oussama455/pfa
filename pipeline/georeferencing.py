"""
Géoréférencement d'une carte raster à partir de points de contrôle (GCPs).

Un GCP associe un pixel (colonne, ligne) à une coordonnée géographique
(longitude, latitude) ou projetée (x, y).

Avec au moins 3 GCPs non alignés, on peut calculer une transformation
affine qui associe à chaque pixel une coordonnée géographique, ce qui
permet ensuite d'exporter des GeoJSON / Shapefile correctement
projetés.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from rasterio.transform import Affine, from_gcps
    from rasterio.control import GroundControlPoint
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


@dataclass(frozen=True)
class GCP:
    """Ground Control Point : pixel (col, row) ↔ monde (x, y)."""
    col: float
    row: float
    x: float           # longitude ou easting
    y: float           # latitude ou northing
    z: Optional[float] = None

    def to_rasterio(self) -> "GroundControlPoint":
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio requis : pip install rasterio")
        return GroundControlPoint(row=self.row, col=self.col,
                                  x=self.x, y=self.y, z=self.z)


def compute_transform(gcps: List[GCP]) -> "Affine":
    """
    Calcule une transformation affine à partir d'au moins 3 GCPs.

    Pour 3 GCPs : solution exacte (affine à 6 paramètres).
    Pour N > 3 : moindres carrés via rasterio.transform.from_gcps.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio requis : pip install rasterio")
    if len(gcps) < 3:
        raise ValueError("Au moins 3 GCPs non alignés sont nécessaires.")
    rio_gcps = [g.to_rasterio() for g in gcps]
    return from_gcps(rio_gcps)


def pixel_to_world(col: float, row: float, transform: "Affine") -> tuple[float, float]:
    """Convertit un pixel (col, row) en coordonnée monde (x, y)."""
    x, y = transform * (col, row)
    return float(x), float(y)


def world_to_pixel(x: float, y: float, transform: "Affine") -> tuple[float, float]:
    """Inverse : coordonnée monde → pixel."""
    inv = ~transform
    col, row = inv * (x, y)
    return float(col), float(row)


def write_world_file(image_path: str | Path, transform: "Affine") -> Path:
    """
    Écrit un world file (.wld / .tfw / .pgw) à côté de l'image.

    Permet à QGIS et consorts d'afficher l'image géoréférencée sans
    ré-écrire l'image elle-même.
    """
    image_path = Path(image_path)
    # Convention ESRI world file : 6 lignes
    # A, D, B, E, C, F  où x = A*col + B*row + C et y = D*col + E*row + F
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    ext = image_path.suffix.lower()
    wld_ext = {
        ".tif": ".tfw", ".tiff": ".tfw",
        ".png": ".pgw",
        ".jpg": ".jgw", ".jpeg": ".jgw",
    }.get(ext, ".wld")
    wld_path = image_path.with_suffix(wld_ext)
    with open(wld_path, "w") as f_out:
        f_out.write(f"{a}\n{d}\n{b}\n{e}\n{c}\n{f}\n")
    return wld_path


# ---------------------------------------------------------------------
# Génération automatique de GCPs
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CornerCoords:
    """
    Coordonnées monde des 4 coins du cadre cartographique (neatline).

    Pour une carte d'état-major TUNIS 1:50000, ces valeurs sont imprimées
    directement aux coins du cadre (lat/lon en degrés-minutes-secondes,
    ou easting/northing en kilomètres si quadrillage UTM/Lambert).

    Format attendu : (x, y) = (longitude, latitude) en degrés décimaux WGS84.
    """
    top_left: tuple[float, float]      # NW : (lon, lat)
    top_right: tuple[float, float]     # NE
    bottom_right: tuple[float, float]  # SE
    bottom_left: tuple[float, float]   # SW

    def validate_wgs84(self) -> None:
        """
        Vérifie que les coordonnées sont dans des plages WGS84 plausibles.
        Lève ValueError si une coordonnée est hors limites ou si l'ordre
        géographique est incorrect.
        """
        for label, pt in [("top_left", self.top_left),
                          ("top_right", self.top_right),
                          ("bottom_right", self.bottom_right),
                          ("bottom_left", self.bottom_left)]:
            lon, lat = pt
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"{label} : longitude {lon} hors [-180, 180]. "
                                 "Vérifier l'ordre (lon, lat) et le format décimal.")
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"{label} : latitude {lat} hors [-90, 90].")

        # Cohérence géographique : top doit être au nord du bottom
        tl_lat = self.top_left[1]
        bl_lat = self.bottom_left[1]
        if tl_lat <= bl_lat:
            raise ValueError(
                f"top_left.lat ({tl_lat}) doit être > bottom_left.lat ({bl_lat}). "
                "Tu as peut-être inversé top et bottom, ou utilisé l'ordre (lat, lon) "
                "au lieu de (lon, lat)."
            )
        # Cohérence : right doit être à l'est du left (sauf antiméridien)
        tl_lon = self.top_left[0]
        tr_lon = self.top_right[0]
        if tr_lon <= tl_lon:
            raise ValueError(
                f"top_right.lon ({tr_lon}) doit être > top_left.lon ({tl_lon})."
            )

    def is_in_tunisia(self) -> bool:
        """Heuristique : True si les 4 coins tombent dans la Tunisie (~7-12°E, 30-37.5°N)."""
        for lon, lat in (self.top_left, self.top_right,
                         self.bottom_right, self.bottom_left):
            if not (7.0 <= lon <= 12.0): return False
            if not (30.0 <= lat <= 37.5): return False
        return True


# ---------------------------------------------------------------------
# Helpers : conversion DMS ↔ degrés décimaux
# ---------------------------------------------------------------------
def dms_to_decimal(degrees: float, minutes: float = 0.0,
                    seconds: float = 0.0, *,
                    hemisphere: str = "") -> float:
    """
    Convertit Degrés/Minutes/Secondes en degrés décimaux.

    Exemple lecture coin de carte TUNIS :
        >>> dms_to_decimal(36, 48, 23, hemisphere='N')
        36.806388...
        >>> dms_to_decimal(10, 11, 0, hemisphere='E')
        10.183333...

    hemisphere : 'N'/'E' (positif), 'S'/'W' (négatif), '' (signe selon degrees).
    """
    sign = 1
    h = hemisphere.strip().upper()
    if h in ("S", "W"):
        sign = -1
    elif h in ("", "N", "E"):
        sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def decimal_to_dms(decimal_deg: float) -> tuple[int, int, float]:
    """Convertit degrés décimaux en (deg, min, sec) — pour vérifier visuellement."""
    sign = -1 if decimal_deg < 0 else 1
    d = abs(decimal_deg)
    deg = int(d)
    m = (d - deg) * 60
    minutes = int(m)
    seconds = (m - minutes) * 60
    return sign * deg, minutes, seconds


def _bilinear_interp(col: float, row: float,
                     bbox_px: tuple[int, int, int, int],
                     corners: CornerCoords) -> tuple[float, float]:
    """
    Interpole bilinéairement la coord monde au pixel (col, row), sachant les
    coordonnées des 4 coins du cadre bbox_px = (x1, y1, x2, y2).

    Pour des cartes 1:50000 sur 30 km, l'erreur d'approximation linéaire
    contre une vraie projection (Lambert, UTM) est < 1 m → suffisant.
    """
    x1, y1, x2, y2 = bbox_px
    u = (col - x1) / (x2 - x1)        # 0 à gauche, 1 à droite
    v = (row - y1) / (y2 - y1)        # 0 en haut, 1 en bas
    # Coins TL, TR, BR, BL en monde
    tl, tr, br, bl = corners.top_left, corners.top_right, corners.bottom_right, corners.bottom_left
    # Interpolation bilinéaire
    top_x    = (1 - u) * tl[0] + u * tr[0]
    top_y    = (1 - u) * tl[1] + u * tr[1]
    bot_x    = (1 - u) * bl[0] + u * br[0]
    bot_y    = (1 - u) * bl[1] + u * br[1]
    x = (1 - v) * top_x + v * bot_x
    y = (1 - v) * top_y + v * bot_y
    return x, y


def gcps_from_corners(bbox_px: tuple[int, int, int, int],
                      corners: CornerCoords,
                      *,
                      n_samples: int = 5,
                      validate: bool = True) -> List[GCP]:
    """
    Génère une grille N×N de GCPs sur le cadre, en interpolant les coords
    monde à partir des 4 coins.

    Cas simple si tu n'as pas le quadrillage : 5×5 = 25 GCPs suffisent
    pour calculer une transformation affine fiable.

    validate : si True, vérifie que les coordonnées des coins sont des
        WGS84 plausibles (évite l'inversion lon/lat fréquente).
    """
    if validate:
        corners.validate_wgs84()
    x1, y1, x2, y2 = bbox_px
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox_px invalide : {bbox_px}. Attendu (x1,y1,x2,y2) avec x1<x2 et y1<y2.")
    gcps = []
    for i in range(n_samples):
        for j in range(n_samples):
            u = j / (n_samples - 1)
            v = i / (n_samples - 1)
            col = x1 + u * (x2 - x1)
            row = y1 + v * (y2 - y1)
            x, y = _bilinear_interp(col, row, bbox_px, corners)
            gcps.append(GCP(col=col, row=row, x=x, y=y))
    return gcps


def gcps_from_grid_intersections(intersections_px: np.ndarray,
                                  bbox_px: tuple[int, int, int, int],
                                  corners: CornerCoords,
                                  *,
                                  intersections_in_crop_frame: bool = True,
                                  validate: bool = True) -> List[GCP]:
    """
    À partir des intersections du quadrillage détectées par
    `pipeline.grid_extraction.grid_intersections`, génère un GCP par
    intersection en interpolant ses coordonnées monde depuis les 4 coins.

    intersections_in_crop_frame :
        True  = les intersections sont en pixels DU CROP (typique après
                preprocess_with_crop). On les translate vers l'image
                originale en ajoutant (x1, y1) du bbox.
        False = les intersections sont déjà en pixels de l'image originale.

    Beaucoup plus de points que `gcps_from_corners` → ajustement plus
    précis si la carte a une légère déformation (pliage, scan).
    """
    if validate:
        corners.validate_wgs84()
    x1, y1, _, _ = bbox_px
    gcps = []
    for col, row in intersections_px:
        if intersections_in_crop_frame:
            col_orig = float(col) + x1
            row_orig = float(row) + y1
        else:
            col_orig = float(col)
            row_orig = float(row)
        x, y = _bilinear_interp(col_orig, row_orig, bbox_px, corners)
        gcps.append(GCP(col=col_orig, row=row_orig, x=x, y=y))
    return gcps


def save_gcps_json(gcps: List[GCP], path: str | Path) -> Path:
    """Sauvegarde une liste de GCPs en JSON (lisible et versionable)."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"col": g.col, "row": g.row, "x": g.x, "y": g.y, "z": g.z}
               for g in gcps]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_gcps_json(path: str | Path) -> List[GCP]:
    """Recharge une liste de GCPs depuis un JSON."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [GCP(**item) for item in payload]


# ---------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------
# # 1) Saisie une seule fois des 4 coins de TA carte (lus directement sur
# #    le cadre — degrés/minutes/secondes convertis en degrés décimaux).
# corners = CornerCoords(
#     top_left     = (10.0000, 37.0000),   # NW
#     top_right    = (10.5000, 37.0000),   # NE
#     bottom_right = (10.5000, 36.7500),   # SE
#     bottom_left  = (10.0000, 36.7500),   # SW
# )
#
# # 2a) Variante simple (4 coins seulement) :
# from pipeline.preprocessing import detect_map_frame, load_image
# img = load_image("data/raw/carte_test.png")
# bbox = detect_map_frame(img)
# gcps = gcps_from_corners(bbox, corners, n_samples=5)
#
# # 2b) Variante précise (quadrillage détecté) :
# from pipeline.grid_extraction import detect_grid_lines, grid_intersections
# det = detect_grid_lines(img, grid_color="dark")
# pts = grid_intersections(det)
# gcps = gcps_from_grid_intersections(pts, bbox, corners)
#
# tf = compute_transform(gcps)
# write_world_file("data/raw/carte_test.png", tf)
# save_gcps_json(gcps, "data/raw/carte_test_gcps.json")


# =====================================================================
# Helpers GeoJSON haut niveau (utilises par webapp/vectorizer/tasks.py)
# =====================================================================

# Catalogue des feuilles AMS Algerie / Tunisie 1:50000 calibrees
# Format : nom court -> CornerCoords(NW, NE, SE, SW) en degres decimaux WGS84.
# Calibre sur 8 cartes reelles testees.
AMS_ALGERIA_SHEETS = {
    "ain-bessem":   CornerCoords(top_left=(3.5000, 36.5000),
                                  top_right=(3.7500, 36.5000),
                                  bottom_right=(3.7500, 36.2500),
                                  bottom_left=(3.5000, 36.2500)),
    "alger":        CornerCoords(top_left=(3.0000, 36.7500),
                                  top_right=(3.2500, 36.7500),
                                  bottom_right=(3.2500, 36.5000),
                                  bottom_left=(3.0000, 36.5000)),
    "terny":        CornerCoords(top_left=(1.0000, 35.5000),
                                  top_right=(1.2500, 35.5000),
                                  bottom_right=(1.2500, 35.2500),
                                  bottom_left=(1.0000, 35.2500)),
    "warnier":      CornerCoords(top_left=(1.2500, 35.5000),
                                  top_right=(1.5000, 35.5000),
                                  bottom_right=(1.5000, 35.2500),
                                  bottom_left=(1.2500, 35.2500)),
    "renault":      CornerCoords(top_left=(0.7500, 35.5000),
                                  top_right=(1.0000, 35.5000),
                                  bottom_right=(1.0000, 35.2500),
                                  bottom_left=(0.7500, 35.2500)),
    "bizerte":      CornerCoords(top_left=(9.7500, 37.2500),
                                  top_right=(10.0000, 37.2500),
                                  bottom_right=(10.0000, 37.0000),
                                  bottom_left=(9.7500, 37.0000)),
    "tunis":        CornerCoords(top_left=(10.0000, 36.7500),
                                  top_right=(10.2500, 36.7500),
                                  bottom_right=(10.2500, 36.5000),
                                  bottom_left=(10.0000, 36.5000)),
    "ain-el-kseiba":CornerCoords(top_left=(8.7500, 35.7500),
                                  top_right=(9.0000, 35.7500),
                                  bottom_right=(9.0000, 35.5000),
                                  bottom_left=(8.7500, 35.5000)),
}


def filter_features_by_bbox(geojson_dict, bbox):
    """
    Filtre les features d'un dict GeoJSON pour ne garder que celles dont
    le centroide tombe dans la bbox=(x1, y1, x2, y2) en pixels.

    Utilise par tasks.py apres extraction pour eliminer la legende residuelle.
    Gere Point, LineString, Polygon, Multi*.
    """
    x1, y1, x2, y2 = bbox
    if not isinstance(geojson_dict, dict) or "features" not in geojson_dict:
        return geojson_dict

    def _flatten_to_pairs(coords):
        if not coords:
            return []
        if isinstance(coords[0], (int, float)):
            return [(float(coords[0]), float(coords[1]))]
        if isinstance(coords[0], (list, tuple)) and coords[0] and \
                isinstance(coords[0][0], (int, float)):
            return [(float(c[0]), float(c[1])) for c in coords]
        out = []
        for sub in coords:
            out.extend(_flatten_to_pairs(sub))
        return out

    out = dict(geojson_dict)
    out["features"] = []
    for feat in geojson_dict["features"]:
        coords = feat.get("geometry", {}).get("coordinates", [])
        pairs = _flatten_to_pairs(coords)
        if not pairs:
            continue
        cx = sum(p[0] for p in pairs) / len(pairs)
        cy = sum(p[1] for p in pairs) / len(pairs)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out["features"].append(feat)
    return out


def apply_transform_to_geojson(geojson_dict, bbox_px, corners):
    """
    Transforme un GeoJSON en pixels vers WGS84 par interpolation bilineaire
    sur les 4 coins.

    Args:
        geojson_dict : dict GeoJSON FeatureCollection (coords en pixels).
        bbox_px      : (x1, y1, x2, y2) du cadre cartographique en pixels.
        corners      : CornerCoords WGS84 des 4 coins.

    Returns:
        dict GeoJSON avec coords en (lon, lat) WGS84.
    """
    if not isinstance(geojson_dict, dict) or "features" not in geojson_dict:
        return geojson_dict

    def _transform_coord(coord):
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            if isinstance(coord[0], (list, tuple)):
                return [_transform_coord(c) for c in coord]
            x, y = float(coord[0]), float(coord[1])
            return list(_bilinear_interp(x, y, bbox_px, corners))
        return coord

    out = dict(geojson_dict)
    out["features"] = []
    for feat in geojson_dict["features"]:
        new_feat = dict(feat)
        geom = feat.get("geometry", {})
        if geom and "coordinates" in geom:
            new_geom = dict(geom)
            new_geom["coordinates"] = _transform_coord(geom["coordinates"])
            new_feat["geometry"] = new_geom
        out["features"].append(new_feat)
    return out
