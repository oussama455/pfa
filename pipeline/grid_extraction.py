"""
Extraction automatique du quadrillage cartographique pour le géoréférencement.

Sur une carte d'état-major (TUNIS 1:50000 par exemple), le quadrillage
kilométrique est imprimé sur la carte. Si on connaît :
    - les coordonnées des 4 coins du cadre (neatline) en monde réel
    - la position des intersections du quadrillage en pixels

… alors on peut générer automatiquement DES DIZAINES de GCPs au lieu d'en
saisir 3 ou 4 à la main → meilleur ajustement, moins d'erreur.

Stratégie :
    1. detect_grid_lines() : trouve les lignes horizontales et verticales
       du quadrillage (méthode "projection" robuste aux scans bruités).
    2. grid_intersections() : croise les listes pour obtenir les pixels
       d'intersection.
    3. (côté georeferencing.py) gcps_from_grid_intersections() : interpole
       les coordonnées monde de chaque intersection à partir des 4 coins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class GridDetection:
    """Résultat de la détection du quadrillage."""
    h_lines: List[int]                    # positions y des lignes horizontales
    v_lines: List[int]                    # positions x des lignes verticales
    h_spacing_px: float                   # pas moyen entre lignes horizontales
    v_spacing_px: float                   # pas moyen entre lignes verticales

    def __repr__(self) -> str:
        return (f"GridDetection(h_lines={len(self.h_lines)}, "
                f"v_lines={len(self.v_lines)}, "
                f"spacing_px=({self.h_spacing_px:.1f}, {self.v_spacing_px:.1f}))")


# -----------------------------------------------------------
# Binarisation pour la couleur du quadrillage
# -----------------------------------------------------------
def _binarize_for_grid(image_bgr: np.ndarray, *,
                       grid_color: str = "dark",
                       hsv_range: Optional[Tuple[Tuple[int, int, int],
                                                  Tuple[int, int, int]]] = None,
                       ) -> np.ndarray:
    """
    Crée un masque binaire des pixels candidats à appartenir au quadrillage.

    grid_color :
        "dark" : seuil sur le canal gris (quadrillage noir).
        "blue" : seuil HSV bleu (quadrillage bleu type IGN).
        "custom" : utilise hsv_range = ((Hmin,Smin,Vmin), (Hmax,Smax,Vmax)).
    """
    if grid_color == "dark":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    elif grid_color == "blue":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array([95, 40, 40], dtype=np.uint8)
        hi = np.array([130, 255, 255], dtype=np.uint8)
        binary = cv2.inRange(hsv, lo, hi)
    elif grid_color == "custom" and hsv_range is not None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lo, hi = hsv_range
        binary = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                              np.array(hi, dtype=np.uint8))
    else:
        raise ValueError(f"grid_color invalide : {grid_color}")
    return binary


# -----------------------------------------------------------
# Détection des lignes du quadrillage
# -----------------------------------------------------------
def detect_grid_lines(image_bgr: np.ndarray, *,
                      grid_color: str = "dark",
                      min_line_length_ratio: float = 0.4,
                      cluster_tolerance_px: int = 8,
                      hsv_range: Optional[Tuple[Tuple[int, int, int],
                                                 Tuple[int, int, int]]] = None,
                      method: str = "projection",
                      peak_prominence: float = 1.5,
                      ) -> GridDetection:
    """
    Détecte les lignes horizontales et verticales du quadrillage.

    Deux méthodes :
        method="projection" (défaut, recommandé) : binarise, isole les
            structures linéaires par ouverture courte, puis projette
            horizontalement et verticalement. Les pics de projection
            donnent les positions des lignes du quadrillage. Robuste aux
            lignes coupées par le contenu cartographique.
        method="morpho" : ouverture morphologique avec noyau allongé,
            n'extrait que les lignes continues. Moins robuste.

    Paramètres :
        grid_color : "dark" (noir), "blue" (bleu IGN), "custom" + hsv_range.
        min_line_length_ratio : longueur minimale d'une ligne (ratio de la
            dimension de l'image), pour la méthode morpho.
        cluster_tolerance_px : fusionne les pics à moins de N px.
        peak_prominence : pic > prominence × moyenne pour être détecté.
    """
    H, W = image_bgr.shape[:2]
    binary = _binarize_for_grid(image_bgr, grid_color=grid_color,
                                  hsv_range=hsv_range)

    if method == "morpho":
        h_kernel_len = max(int(W * min_line_length_ratio), 30)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_kernel_len = max(int(H * min_line_length_ratio), 30)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        h_lines = _extract_line_positions(h_mask, axis="horizontal",
                                            cluster_tol=cluster_tolerance_px)
        v_lines = _extract_line_positions(v_mask, axis="vertical",
                                            cluster_tol=cluster_tolerance_px)

    elif method == "projection":
        # Pré-isolation : ouverture courte pour ne garder que des
        # structures qui ressemblent à des lignes (pas du bruit ponctuel).
        h_open = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 100, 15), 1)))
        v_open = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 100, 15))))

        proj_y = (h_open > 0).sum(axis=1).astype(np.float64)
        proj_x = (v_open > 0).sum(axis=0).astype(np.float64)

        h_lines = _peaks_above_threshold(proj_y,
                                           prominence=peak_prominence,
                                           min_distance=cluster_tolerance_px)
        v_lines = _peaks_above_threshold(proj_x,
                                           prominence=peak_prominence,
                                           min_distance=cluster_tolerance_px)
    else:
        raise ValueError(f"method invalide : {method}")

    h_spacing = _median_spacing(h_lines)
    v_spacing = _median_spacing(v_lines)

    return GridDetection(h_lines=h_lines, v_lines=v_lines,
                          h_spacing_px=h_spacing, v_spacing_px=v_spacing)


def _peaks_above_threshold(signal: np.ndarray, *,
                            prominence: float = 1.5,
                            min_distance: int = 8) -> List[int]:
    """
    Détecteur de pics simple : un pic est un maximum local strictement
    supérieur à `prominence × moyenne`.

    Évite la dépendance à scipy.signal.find_peaks.
    """
    if signal.size == 0:
        return []
    threshold = float(signal.mean()) * prominence
    peaks: List[int] = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] < threshold:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if peaks and i - peaks[-1] < min_distance:
                if signal[i] > signal[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return peaks


def _extract_line_positions(mask: np.ndarray, *,
                              axis: str,
                              cluster_tol: int = 8) -> List[int]:
    """
    Sur un masque ne contenant que des lignes, extrait la position
    centrale de chaque ligne distincte.
    """
    if axis == "horizontal":
        projection = (mask > 0).sum(axis=1)
    else:
        projection = (mask > 0).sum(axis=0)

    if projection.max() == 0:
        return []

    threshold = projection.max() * 0.3
    active = projection > threshold

    positions: List[int] = []
    in_run = False
    run_start = 0
    for i, val in enumerate(active):
        if val and not in_run:
            in_run = True
            run_start = i
        elif not val and in_run:
            in_run = False
            positions.append((run_start + i - 1) // 2)
    if in_run:
        positions.append((run_start + len(active) - 1) // 2)

    if cluster_tol > 0 and len(positions) > 1:
        merged: List[int] = [positions[0]]
        for p in positions[1:]:
            if p - merged[-1] < cluster_tol:
                merged[-1] = (merged[-1] + p) // 2
            else:
                merged.append(p)
        positions = merged

    return positions


def _median_spacing(positions: List[int]) -> float:
    """Pas médian entre positions triées (0 si moins de 2 points)."""
    if len(positions) < 2:
        return 0.0
    diffs = np.diff(sorted(positions))
    return float(np.median(diffs))


def filter_regular_lines(positions: List[int], *,
                          tolerance_ratio: float = 0.15) -> List[int]:
    """
    Garde uniquement les lignes dont l'espacement est cohérent avec un
    multiple entier du pas médian (élimine les faux positifs).

    tolerance_ratio : une ligne est gardée si son écart au voisin est à
                      moins de r près d'un multiple entier du pas médian.
    """
    if len(positions) < 3:
        return list(positions)
    sorted_pos = sorted(positions)
    median_step = _median_spacing(sorted_pos)
    if median_step == 0:
        return sorted_pos

    kept = [sorted_pos[0]]
    for p in sorted_pos[1:]:
        diff = p - kept[-1]
        ratio = diff / median_step
        nearest_int = round(ratio)
        if nearest_int >= 1 and abs(ratio - nearest_int) <= tolerance_ratio:
            kept.append(p)
    return kept


# -----------------------------------------------------------
# Intersections du quadrillage
# -----------------------------------------------------------
def grid_intersections(detection: GridDetection) -> np.ndarray:
    """
    Retourne un tableau Nx2 avec les coordonnées (x, y) en pixels de chaque
    intersection ligne H × ligne V.
    """
    if not detection.h_lines or not detection.v_lines:
        return np.zeros((0, 2), dtype=np.float64)
    pts = []
    for y in detection.h_lines:
        for x in detection.v_lines:
            pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)


# -----------------------------------------------------------
# Visualisation (pour notebooks)
# -----------------------------------------------------------
def draw_grid_overlay(image_bgr: np.ndarray, detection: GridDetection, *,
                       line_color: Tuple[int, int, int] = (0, 255, 0),
                       intersection_color: Tuple[int, int, int] = (255, 0, 255),
                       thickness: int = 2,
                       radius: int = 6) -> np.ndarray:
    """Dessine les lignes détectées et leurs intersections sur l'image."""
    out = image_bgr.copy()
    H, W = out.shape[:2]
    for y in detection.h_lines:
        cv2.line(out, (0, y), (W - 1, y), line_color, thickness)
    for x in detection.v_lines:
        cv2.line(out, (x, 0), (x, H - 1), line_color, thickness)
    for x in detection.v_lines:
        for y in detection.h_lines:
            cv2.circle(out, (x, y), radius, intersection_color, -1)
    return out
