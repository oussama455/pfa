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
    """
    top_left: tuple[float, float]      # (x, y) ou (lon, lat)
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]


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
                      n_samples: int = 5) -> List[GCP]:
    """
    Génère une grille N×N de GCPs sur le cadre, en interpolant les coords
    monde à partir des 4 coins.

    Cas simple si tu n'as pas le quadrillage : 5×5 = 25 GCPs suffisent
    pour calculer une transformation affine fiable.
    """
    x1, y1, x2, y2 = bbox_px
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
                                  intersections_in_crop_frame: bool = True) -> List[GCP]:
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
