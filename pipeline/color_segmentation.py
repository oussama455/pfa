"""
Segmentation par couleur (OpenCV, HSV).

Exploite la convention cartographique :
    bleu   → hydrographie (rivières, lacs)  ─ ou quadrillage : voir density_filter
    vert   → végétation, forêts
    marron → courbes de niveau
    rouge  → routes principales
    noir   → routes, bâtiments, texte (traité par le module IA)

Important : les plages HSV par défaut sont génériques. Pour de bons résultats,
calibrer sur ta carte avec notebooks/02_hsv_calibration.ipynb.

Pour les cartes avec un quadrillage marqué (cartes militaires d'état-major),
utiliser density_filter() pour distinguer les zones denses (eau, végétation)
des lignes fines (quadrillage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


# -----------------------------------------------------------
# Plages HSV (à AJUSTER sur ta carte de test via 02_hsv_calibration.ipynb)
# OpenCV : H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255]
# -----------------------------------------------------------
@dataclass(frozen=True)
class HSVRange:
    """Plage HSV (min / max) pour seuillage cv2.inRange."""
    h_min: int
    s_min: int
    v_min: int
    h_max: int
    s_max: int
    v_max: int

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)

    def __repr__(self) -> str:
        return (f"HSVRange(H {self.h_min}-{self.h_max}, "
                f"S {self.s_min}-{self.s_max}, V {self.v_min}-{self.v_max})")


# Defaults plus stricts (saturation min plus élevée pour éviter de prendre
# le papier crème / jauni des cartes anciennes).
#
# IMPORTANT : la range "contours" est volontairement RESTRICTIVE pour éviter
# l'explosion de features (typiquement 44 → 7000+ si on prend trop large).
# Sur la carte TUNIS 1:50000, le brun des courbes a S>=70, V<=210.
# Le papier jauni a S~30 et V~250 → exclu par S_min=70 et V_max=210.
DEFAULT_RANGES: Dict[str, HSVRange] = {
    "water":      HSVRange(h_min=95,  s_min=60, v_min=60,  h_max=130, s_max=255, v_max=255),
    "vegetation": HSVRange(h_min=40,  s_min=40, v_min=50,  h_max=85,  s_max=255, v_max=255),
    "contours":   HSVRange(h_min=8,   s_min=70, v_min=60,  h_max=22,  s_max=220, v_max=210),
}


# Le rouge en HSV traverse la jonction 0/180 → deux plages à combiner
RED_RANGE_LOW  = HSVRange(h_min=0,   s_min=100, v_min=70, h_max=8,   s_max=255, v_max=255)
RED_RANGE_HIGH = HSVRange(h_min=170, s_min=100, v_min=70, h_max=179, s_max=255, v_max=255)


# -----------------------------------------------------------
# Fonctions bas niveau
# -----------------------------------------------------------
def mask_from_range(hsv: np.ndarray, hsv_range: HSVRange) -> np.ndarray:
    """Retourne un masque binaire uint8 (0 / 255) pour la plage HSV."""
    return cv2.inRange(hsv, hsv_range.lower, hsv_range.upper)


def mask_red(hsv: np.ndarray) -> np.ndarray:
    """Masque rouge (combinaison des deux plages H basses et hautes)."""
    return cv2.bitwise_or(mask_from_range(hsv, RED_RANGE_LOW),
                          mask_from_range(hsv, RED_RANGE_HIGH))


def clean_mask(mask: np.ndarray, *,
               open_kernel: int = 3,
               close_kernel: int = 5,
               min_area: int = 50) -> np.ndarray:
    """
    Nettoie un masque binaire :
        - ouverture : supprime le bruit poivre et sel
        - fermeture : comble les trous
        - suppression des composantes trop petites
    """
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    out = np.zeros_like(cleaned)
    for i in range(1, num):  # 0 = fond
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def density_filter(mask: np.ndarray, *,
                   window: int = 25,
                   min_density: float = 0.25) -> np.ndarray:
    """
    Garde uniquement les pixels qui se trouvent dans une zone DENSE.

    Idée : un quadrillage est constitué de fines lignes parallèles, donc la
    densité locale (fraction de pixels actifs dans une fenêtre) est faible.
    Au contraire une zone d'eau hachurée a une densité locale élevée.

    Paramètres :
        window      : taille de la fenêtre de densité (impair, en pixels).
        min_density : densité minimum pour conserver le pixel (0–1).

    Pour cartes 1:50000 scannées vers 2000-3000 px de large, window=25-35 marche
    bien pour distinguer eau (dense) du quadrillage (épars).
    """
    if window % 2 == 0:
        window += 1
    binary = (mask > 0).astype(np.float32)
    # Densité locale = moyenne dans une fenêtre carrée (filtre uniforme rapide)
    density = cv2.boxFilter(binary, ddepth=-1, ksize=(window, window),
                             normalize=True)
    keep = (density >= min_density).astype(np.uint8) * 255
    # Combine avec le masque original : on ne garde que les pixels du masque
    # qui sont dans une zone dense.
    return cv2.bitwise_and(mask, keep)


def remove_thin_lines(mask: np.ndarray, *,
                      min_thickness: int = 5) -> np.ndarray:
    """
    Supprime les structures plus fines que `min_thickness` pixels (lignes
    de quadrillage, traits d'écriture).

    Conserve les zones de largeur >= min_thickness (zones denses, taches).
    Combine bien avec density_filter pour des cartes très "imprimées".
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (min_thickness, min_thickness))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Squelettise un masque binaire (utile pour courbes de niveau)."""
    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary).astype(np.uint8) * 255
    return skel


def filter_skeleton_by_length(skeleton: np.ndarray, *,
                               min_length_px: int = 30) -> np.ndarray:
    """
    Élimine les composantes connexes du squelette plus courtes que
    `min_length_px` pixels. Évite les milliers de petits fragments (qui
    transforment 44 vraies courbes en 7000+ features).

    À utiliser APRÈS skeletonize_mask().
    """
    binary = (skeleton > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, num):  # 0 = fond
        # Pour un squelette, l'AIRE = nombre de pixels = longueur approchée
        if stats[i, cv2.CC_STAT_AREA] >= min_length_px:
            out[labels == i] = 255
    return out


# -----------------------------------------------------------
# API haut niveau
# -----------------------------------------------------------
def extract_all_color_layers(hsv: np.ndarray, *,
                              clean: bool = True,
                              skeletonize_contours: bool = True,
                              dense_filter_zones: bool = True,
                              density_window: int = 25,
                              density_threshold: float = 0.25,
                              contour_min_area: int = 200,
                              contour_min_skeleton_length: int = 50,
                              red_min_area: int = 80,
                              ) -> Dict[str, np.ndarray]:
    """
    Extrait toutes les couches colorimétriques standard d'une carte.

    Arguments :
        clean : applique morphologie + filtrage par taille (défaut : oui).
        skeletonize_contours : transforme les courbes en squelette 1px.
        dense_filter_zones : applique density_filter sur eau + végétation.
        density_window, density_threshold : voir density_filter().
        contour_min_area    : aire mini avant squelettisation des courbes
            (200 px² élimine les fragments de bruit ; 30 explose en 7000+
            features).
        contour_min_skeleton_length : longueur minimale (px) d'une courbe
            APRÈS squelettisation. 50 garde les vraies courbes de niveau et
            élimine les petits fragments.
        red_min_area : aire mini pour les routes rouges.
    """
    layers: Dict[str, np.ndarray] = {}

    # Plages "zones" (eau, végétation) : on veut des régions denses
    for name in ("water", "vegetation"):
        rng = DEFAULT_RANGES[name]
        m = mask_from_range(hsv, rng)
        if clean:
            m = clean_mask(m, open_kernel=3, close_kernel=7, min_area=100)
        if dense_filter_zones:
            m = density_filter(m, window=density_window,
                                min_density=density_threshold)
            m = clean_mask(m, open_kernel=3, close_kernel=5, min_area=200)
        layers[name] = m

    # Courbes de niveau : on veut des lignes, pas de density_filter
    contour_raw = mask_from_range(hsv, DEFAULT_RANGES["contours"])
    if clean:
        # min_area élevé AVANT squelettisation : élimine le bruit en taches
        contour_raw = clean_mask(contour_raw, open_kernel=2,
                                  close_kernel=3, min_area=contour_min_area)
    if skeletonize_contours:
        contour_raw = skeletonize_mask(contour_raw)
        # CRITIQUE : filtre par longueur APRÈS squelettisation, sinon on
        # se retrouve avec des milliers de mini-fragments (44 → 7434 problème).
        contour_raw = filter_skeleton_by_length(
            contour_raw, min_length_px=contour_min_skeleton_length)
    layers["contours"] = contour_raw

    # Routes rouges : lignes fines aussi → pas de density_filter
    red_raw = mask_red(hsv)
    if clean:
        red_raw = clean_mask(red_raw, open_kernel=2, close_kernel=3,
                              min_area=red_min_area)
    layers["red_roads"] = red_raw

    return layers


def coverage_percent(mask: np.ndarray) -> float:
    """Pourcentage de pixels actifs dans un masque binaire (diagnostic)."""
    return 100.0 * float((mask > 0).sum()) / float(mask.size)


# -----------------------------------------------------------
# Helpers de calibration
# -----------------------------------------------------------
def sample_hsv_at(hsv: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """
    Retourne les valeurs HSV à des coordonnées (x, y) données.

    Utile pour calibrer : clique sur 3-5 pixels d'eau dans ton image,
    note leurs coordonnées, puis appelle cette fonction pour voir leur HSV.
    """
    samples = np.array([hsv[y, x] for (x, y) in points], dtype=np.int32)
    return samples


def suggest_range_from_samples(hsv_samples: np.ndarray, *,
                                margin_h: int = 10,
                                margin_s: int = 30,
                                margin_v: int = 40) -> HSVRange:
    """
    Construit une HSVRange autour d'un nuage d'échantillons HSV.

    On élargit chaque borne d'une marge pour tolérer les variations.
    """
    h_min = max(0,   int(hsv_samples[:, 0].min()) - margin_h)
    h_max = min(179, int(hsv_samples[:, 0].max()) + margin_h)
    s_min = max(0,   int(hsv_samples[:, 1].min()) - margin_s)
    s_max = min(255, int(hsv_samples[:, 1].max()) + margin_s)
    v_min = max(0,   int(hsv_samples[:, 2].min()) - margin_v)
    v_max = min(255, int(hsv_samples[:, 2].max()) + margin_v)
    return HSVRange(h_min=h_min, s_min=s_min, v_min=v_min,
                    h_max=h_max, s_max=s_max, v_max=v_max)
