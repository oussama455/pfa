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

import gc
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


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
# Plages calibrées sur cartes AMS/GSGS Algeria 1:50,000 (Aïn Bessem, GSGS 4072)
# Validé empiriquement sur carte.png (6719×5319 px, downscalée à 2400px).
#
# red_roads : Le rouge cartographique se situe dans H=0-10 (rouge pur) et H=170-180
#             (cramoisi). S_min=90 élimine les tons pastels tout en gardant les routes.
#             Ces masques sont SQUELETTISÉS pour produire des LineStrings (non des polygones).
#
# buildings : Bâtiments = gris foncé / anthracite, faible saturation, faible luminosité.
#             S_max=50, V_max=130. Produit des polygones fermés.
#
# contours  : Brun pâle AMS, S_min abaissé à 40 (au lieu de 70 dans la v1).
DEFAULT_RANGES: Dict[str, HSVRange] = {
    "water":      HSVRange(h_min=95,  s_min=60,  v_min=60,  h_max=130, s_max=255, v_max=255),
    "vegetation": HSVRange(h_min=40,  s_min=40,  v_min=50,  h_max=85,  s_max=255, v_max=255),
    # Courbes de niveau AMS/GSGS : brun pâle, S faible → S_min=40
    "contours":   HSVRange(h_min=6,   s_min=40,  v_min=75,  h_max=25,  s_max=160, v_max=215),
    # Routes rouges : rouge pur H=0-10, saturation élevée → squelettisé en lignes
    "red_roads":  HSVRange(h_min=0,   s_min=90,  v_min=70,  h_max=10,  s_max=255, v_max=255),
    # Bâtiments : gris foncé / anthracite, S faible, V faible
    "buildings":  HSVRange(h_min=0,   s_min=0,   v_min=30,  h_max=180, s_max=50,  v_max=130),
}

# Plage supplémentaire pour le rouge cramoisi (H=170-180) utilisé sur certaines
# cartes pour les routes secondaires. Fusionné avec red_roads dans extract_all_color_layers.
_RED_ROADS_HIGH = HSVRange(h_min=170, s_min=90, v_min=70, h_max=180, s_max=255, v_max=255)


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
                               min_length_px: int = 50) -> np.ndarray:
    """
    Élimine les composantes connexes du squelette plus courtes que
    `min_length_px` pixels. Évite les milliers de petits fragments
    (transforme par exemple 7434 features bruitées en ~44 vraies courbes).

    À utiliser APRÈS skeletonize_mask().
    """
    binary = (skeleton > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_length_px:
            out[labels == i] = 255
    return out


# -----------------------------------------------------------
# API haut niveau
# -----------------------------------------------------------
def _clean_thin_lines(mask: np.ndarray, *, min_area: int = 20) -> np.ndarray:
    """
    Nettoyage adapté aux lignes fines (courbes de niveau, 1-3 px de large).

    Contrairement à clean_mask(), on N'APPLIQUE PAS de morphological open
    (qui éroderait et détruirait les lignes minces). On applique uniquement :
      - une fermeture 2x2 pour combler les micro-gaps d'1 px
      - un filtrage par taille (supprime les composantes < min_area px)
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def extract_all_color_layers(hsv: np.ndarray, *,
                              clean: bool = True,
                              skeletonize_contours: bool = True,
                              dense_filter_zones: bool = True,
                              density_window: int = 25,
                              density_threshold: float = 0.25,
                              contour_min_area: int = 20,
                              contour_min_skeleton_length: int = 50,
                              red_min_skeleton_length: int = 30,
                              ranges: Optional[Dict[str, HSVRange]] = None,
                              red_roads_high: Optional[HSVRange] = None,
                              ) -> Dict[str, np.ndarray]:
    """
    Extrait toutes les couches colorimétriques standard d'une carte.

    Arguments :
        clean : applique morphologie + filtrage par taille (défaut : oui).
        skeletonize_contours : transforme les courbes en squelette 1px.
        dense_filter_zones : applique density_filter sur eau + végétation.
        density_window, density_threshold : voir density_filter().
        contour_min_area : aire mini AVANT squelettisation des courbes.
        contour_min_skeleton_length : longueur mini APRÈS squelettisation
            (50 garde les vraies courbes, élimine les fragments parasites).
        red_min_skeleton_length : idem pour les routes rouges (30 garde les
            vraies routes même courtes).
    """
    layers: Dict[str, np.ndarray] = {}
    active_ranges = ranges or DEFAULT_RANGES
    active_red_high = red_roads_high or _RED_ROADS_HIGH

    # Plages "zones" (eau, végétation) : on veut des régions denses
    for name in ("water", "vegetation"):
        rng = active_ranges[name]
        m = mask_from_range(hsv, rng)
        if clean:
            m = clean_mask(m, open_kernel=3, close_kernel=7, min_area=100)
        if dense_filter_zones:
            m = density_filter(m, window=density_window,
                                min_density=density_threshold)
            m = clean_mask(m, open_kernel=3, close_kernel=5, min_area=200)
        layers[name] = m

    # Courbes de niveau : lignes fines (1-3px) → pas de morphological open
    contour_raw = mask_from_range(hsv, active_ranges["contours"])
    if clean:
        contour_raw = _clean_thin_lines(contour_raw, min_area=contour_min_area)
    if skeletonize_contours:
        contour_raw = skeletonize_mask(contour_raw)
        # CRITIQUE : filtre par longueur APRÈS squelettisation
        # (sans ça, on a 7434 fragments au lieu de 44 vraies courbes).
        contour_raw = filter_skeleton_by_length(
            contour_raw, min_length_px=contour_min_skeleton_length)
    layers["contours"] = contour_raw

    # Routes rouges : fusion des deux plages rouge pur + cramoisi, puis squelettisation
    # → LineStrings propres (pas de polygones de bord de route).
    rd_low  = mask_from_range(hsv, active_ranges["red_roads"])
    rd_high = mask_from_range(hsv, active_red_high)
    rd_raw  = cv2.bitwise_or(rd_low, rd_high)
    # Libère les deux sous-masques dès la fusion faite (économise H×W octets ×2)
    del rd_low, rd_high
    if clean:
        rd_raw = clean_mask(rd_raw, open_kernel=2, close_kernel=3, min_area=30)
    rd_raw = skeletonize_mask(rd_raw)
    rd_raw = filter_skeleton_by_length(rd_raw, min_length_px=red_min_skeleton_length)
    layers["red_roads"] = rd_raw

    # Bâtiments : polygones fermés — ouverture + fermeture standard
    if "buildings" in active_ranges:
        bld_raw = mask_from_range(hsv, active_ranges["buildings"])
        if clean:
            bld_raw = clean_mask(bld_raw, open_kernel=3, close_kernel=5, min_area=50)
        layers["buildings"] = bld_raw

    # Sur grosses images on libère tout le scratch numpy avant de rendre la main
    if hsv.size > 8_000_000:   # ~ 2400×3400×3 dépasse ce seuil
        gc.collect()

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
