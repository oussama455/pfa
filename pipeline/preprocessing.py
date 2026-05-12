"""
Prétraitement d'une carte raster avant segmentation.

Opérations :
    - Lecture + resize adaptatif (évite OOM sur GPU 4 GB)
    - Débruitage bilatéral (préserve les bords)
    - Conversion colorimétrique BGR → HSV / RGB
    - Normalisation CLAHE (optionnel)
    - Détection du cadre cartographique (neatline) — Stage 1
    - Suppression de la légende interne               — Stage 2  ← NOUVEAU
    - Crop + validation de la zone utile

Testé sur 8 cartes militaires réelles (Tunisie + Algérie, 1:50 000 WWII) :
    Bizerte, Tunis, Aïn El Kseïba, Aïn Bessem, Alger, Terny, Warnier, Renault
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# I/O & resize
# ════════════════════════════════════════════════════════════════════════════

def load_image(path: str | Path) -> np.ndarray:
    """Charge une image BGR. Lève FileNotFoundError si introuvable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Carte introuvable : {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return img


def downscale_if_too_large(image_bgr: np.ndarray, *,
                            max_dimension: int = 2400,
                            interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Redimensionne si la plus grande dimension dépasse max_dimension.
    Sur RTX 2050 (4 GB VRAM) : max_dimension=2400 → sûr pour l'inférence.
    """
    H, W = image_bgr.shape[:2]
    longest = max(H, W)
    if longest <= max_dimension:
        return image_bgr
    scale = max_dimension / longest
    return cv2.resize(image_bgr,
                      (int(round(W * scale)), int(round(H * scale))),
                      interpolation=interpolation)


# ════════════════════════════════════════════════════════════════════════════
# Filtres image
# ════════════════════════════════════════════════════════════════════════════

def denoise(image_bgr: np.ndarray, d: int = 9,
            sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    """Débruitage bilatéral — préserve les bords des routes."""
    return cv2.bilateralFilter(image_bgr, d=d,
                                sigmaColor=sigma_color, sigmaSpace=sigma_space)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_illumination(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE sur canal L (Lab) — utile pour scans avec éclairage inhomogène."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 : Détection du cadre cartographique (neatline)
# ════════════════════════════════════════════════════════════════════════════

def _detect_frame_by_content_density(image_bgr, *,
                                       content_threshold: int = 230,
                                       margin_ratio: float = 0.01,
                                       min_content_ratio: float = 0.05):
    """
    Fallback : detecte le cadre par densite de contenu plutot que par contour.

    Idee : meme si le neatline n'est pas un contour ferme, le contenu utile
    de la carte (encre coloree + lignes) occupe une zone centrale, alors
    que les marges sont essentiellement du papier blanc/creme.

    Algo :
      1. Tout pixel dont gray < content_threshold (= "encre" ou couleur) est marque.
      2. On somme par ligne et par colonne pour trouver les bornes ou le contenu
         devient dense (>= min_content_ratio * dimension).
      3. On retourne (x1, y1, x2, y2) en serrant ces bornes.

    Marche sur les cartes German GSGS 1:25000 ou la frame n'est qu'un trait
    fin discontinu, mais le contenu de la carte est tres marque.
    """
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    content = (gray < content_threshold).astype(np.uint8)

    # Lisse pour ignorer le bruit ponctuel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)

    row_sum = content.sum(axis=1)  # densite par ligne
    col_sum = content.sum(axis=0)  # densite par colonne
    row_thr = min_content_ratio * W
    col_thr = min_content_ratio * H

    rows_active = np.where(row_sum >= row_thr)[0]
    cols_active = np.where(col_sum >= col_thr)[0]
    if len(rows_active) == 0 or len(cols_active) == 0:
        return None

    y1, y2 = int(rows_active.min()), int(rows_active.max())
    x1, x2 = int(cols_active.min()), int(cols_active.max())

    # Marge de securite
    mx = int(W * margin_ratio)
    my = int(H * margin_ratio)
    x1 = max(0, x1 + mx); y1 = max(0, y1 + my)
    x2 = min(W, x2 - mx); y2 = min(H, y2 - my)

    if x2 - x1 < 0.30 * W or y2 - y1 < 0.30 * H:
        return None  # pas assez de contenu pour etre une carte valable
    return (x1, y1, x2 - x1, y2 - y1)


def _detect_frame_at_threshold(image_bgr, dark_threshold, min_area_ratio,
                                  max_area_ratio, aspect_ratio_range,
                                  morph_kernel_size: int = 5):
    """
    Implementation de bas niveau : binarise a `dark_threshold` et cherche
    le meilleur contour rectangulaire. Retourne (x, y, w, h) ou None.
    """
    H, W = image_bgr.shape[:2]
    img_area = W * H
    cx, cy = W // 2, H // 2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (morph_kernel_size, morph_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    ar_min, ar_max = aspect_ratio_range
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
            continue
        aspect = h / max(w, 1)
        if not (ar_min <= aspect <= ar_max):
            continue
        contains_center = (x <= cx <= x + w) and (y <= cy <= y + h)
        score = area / img_area
        if contains_center:
            score += 0.5
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            score += 0.2
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def detect_map_frame(image_bgr: np.ndarray, *,
                     dark_threshold: int = 80,
                     min_area_ratio: float = 0.30,
                     max_area_ratio: float = 0.95,
                     pad: int = 5,
                     aspect_ratio_range: Tuple[float, float] = (0.5, 2.5),
                     prefer_centered: bool = True,
                     inner_margin_ratio: float = 0.015,
                     retry_thresholds: Optional[Tuple[int, ...]] = (110, 140, 170, 200),
                     adaptive_min_area: bool = True,
                     min_area_floor: float = 0.05,
                     ) -> Tuple[int, int, int, int]:
    """
    Detecte le cadre rectangulaire (neatline) de la zone cartographique.

    Strategie en cascade pour couvrir plusieurs styles cartographiques :

    1. Premier essai au `dark_threshold` standard (80) avec `min_area_ratio=0.30`.
       Marche sur les cartes AMS / GSGS Tunisie+Algerie qui ont un neatline noir epais.

    2. Si echec : retry sur `retry_thresholds=(110, 140, 170)` qui captent les
       neatlines plus pales des cartes TXU / UT Libraries (German GSGS,
       Italian, Reykjavik, etc.) ou la frame est imprimee en gris plutot que noir.

    3. Si `adaptive_min_area=True` : a chaque retry on baisse aussi
       `min_area_ratio` (0.30 -> 0.20 -> 0.15 -> 0.10) pour tolerer les
       cartes dont la frame n'occupe que 50-70% de l'image scannee.

    NEW — `inner_margin_ratio` (defaut 1.5%) :
        Apres detection du neatline, recule de inner_margin_ratio × largeur/hauteur
        vers l'interieur pour exclure les tirets de graduation et les
        numeros de coordonnees imprimes juste a l'interieur.

    Retourne (x1, y1, x2, y2) en pixels. Si aucun cadre trouve, retourne l'image entiere.
    """
    H, W = image_bgr.shape[:2]

    # Construit la liste des (threshold, min_area_ratio) a essayer.
    # On commence strict (peu de faux positifs) puis on relache progressivement.
    thresholds_to_try = [(dark_threshold, min_area_ratio)]
    if retry_thresholds:
        if adaptive_min_area:
            # Decroissance lineaire de min_area_ratio jusqu'a min_area_floor.
            n = len(retry_thresholds)
            step = (min_area_ratio - min_area_floor) / max(n, 1)
            for i, thr in enumerate(retry_thresholds, start=1):
                mar = max(min_area_floor, min_area_ratio - i * step)
                thresholds_to_try.append((thr, mar))
        else:
            for thr in retry_thresholds:
                thresholds_to_try.append((thr, min_area_ratio))

    bbox_inner = None
    used_thr = None
    used_mar = None
    for thr, mar in thresholds_to_try:
        bbox_inner = _detect_frame_at_threshold(
            image_bgr, dark_threshold=thr, min_area_ratio=mar,
            max_area_ratio=max_area_ratio,
            aspect_ratio_range=aspect_ratio_range,
        )
        if bbox_inner is not None:
            used_thr, used_mar = thr, mar
            break

    used_method = "neatline_contour"
    if bbox_inner is None:
        # Fallback : detection par densite de contenu (utile pour les cartes
        # German GSGS dont le neatline est compose de traits separes).
        logger.debug(f"Neatline contour-based echoue ; essai fallback densite...")
        bbox_inner = _detect_frame_by_content_density(image_bgr)
        used_method = "content_density"

    if bbox_inner is None:
        logger.debug(f"Aucune frame detectee, retour image entiere")
        return (0, 0, W, H)

    x, y, w, h = bbox_inner

    # Inner margin : neatline border + ticks + coord numbers
    inner_x = max(pad, int(W * inner_margin_ratio))
    inner_y = max(pad, int(H * inner_margin_ratio))
    x1 = max(0, x + inner_x)
    y1 = max(0, y + inner_y)
    x2 = min(W, x + w - inner_x)
    y2 = min(H, y + h - inner_y)

    logger.debug(f"Neatline: ({x1},{y1})->({x2},{y2}) "
                 f"[thr={used_thr}, min_area={used_mar}]")
    return (x1, y1, x2, y2)


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 : Suppression de la légende interne
# ════════════════════════════════════════════════════════════════════════════

def detect_legend_x(cropped_bgr: np.ndarray) -> int:
    """
    Détecte la position x de la légende interne (panneau droit) dans l'image
    déjà cropée sur le neatline.

    La légende est DANS le neatline sur toutes les cartes AMS/GSGS testées —
    `detect_map_frame` seul ne suffit pas.

    Deux méthodes calibrées sur 8 cartes réelles :

    Méthode A — séparateur sombre (cartes Tunisie : Bizerte, Tunis) :
        Une ligne noire verticale épaisse sépare le contenu cartographique
        de la légende. Détectée si >25 % des pixels de la colonne sont noirs.

    Méthode B — zone claire sans routes rouges (cartes Algérie) :
        La légende a un fond beige uni sans routes rouges.
        Détectée si moyenne de luminosité > 210 ET somme de pixels rouges < 50.

    Méthode C — fallback 87 % :
        Si le tiers droit est plus uniforme que le reste, on coupe à 87 %.
        Observé : toutes les 8 cartes testées ont la légende entre 86-95 %.

    Returns:
        x où couper (0 = pas de légende détectée → garder pleine largeur).
    """
    h, w    = cropped_bgr.shape[:2]
    gray    = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    hsv     = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    search  = int(w * 0.70)   # chercher uniquement dans le 30 % droit

    # ── Méthode A : séparateur sombre vertical ──────────────────────────────
    col_dark = (gray < 80).sum(axis=0)
    cands_a  = np.where(col_dark[search:] > h * 0.25)[0]
    if len(cands_a):
        lx = int(cands_a[0]) + search
        right_mean = gray[:, lx + 5:lx + 50].mean() if lx + 50 < w else 0
        if right_mean > 180:
            logger.debug(f"Legend (A — separator) at x={lx} ({lx/w*100:.1f}%)")
            return lx

    # ── Méthode B : fond clair + absence de routes rouges ───────────────────
    r1   = cv2.inRange(hsv, (0,  60, 60), (15, 255, 255))
    r2   = cv2.inRange(hsv, (155,60, 60), (180,255, 255))
    red  = cv2.bitwise_or(r1, r2)
    sm_b = np.convolve(np.mean(gray[:, search:], axis=0), np.ones(10)/10, 'valid')
    sm_r = np.convolve(red[:, search:].sum(axis=0),       np.ones(10)/10, 'valid')
    cands_b = np.where((sm_b > 210) & (sm_r < 50))[0]
    if len(cands_b):
        lx = int(cands_b[0]) + search
        logger.debug(f"Legend (B — bright+no_red) at x={lx} ({lx/w*100:.1f}%)")
        return lx

    # ── Méthode C : fallback 87 % (si le tiers droit est uniforme) ──────────
    fb = int(w * 0.87)
    if gray[:, fb:].std() < gray[:, search:fb].std() * 0.7:
        logger.debug(f"Legend (C — fallback 87%) at x={fb}")
        return fb

    logger.debug("Legend not detected — keeping full width")
    return 0


def crop_to_frame(image: np.ndarray,
                  bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Recadre à (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


# ════════════════════════════════════════════════════════════════════════════
# API haut niveau
# ════════════════════════════════════════════════════════════════════════════

def preprocess(path: str | Path, *,
               denoise_on: bool = True,
               normalize_on: bool = False,
               max_dimension: int = 2400) -> Tuple[np.ndarray, np.ndarray]:
    """Prétraitement sans recadrage. Retourne (image_bgr, image_hsv)."""
    img = load_image(path)
    if max_dimension:
        img = downscale_if_too_large(img, max_dimension=max_dimension)
    if normalize_on:
        img = normalize_illumination(img)
    if denoise_on:
        img = denoise(img)
    return img, to_hsv(img)


def preprocess_with_crop(path: str | Path, *,
                          denoise_on: bool = True,
                          normalize_on: bool = False,
                          auto_crop: bool = True,
                          remove_legend: bool = True,
                          manual_bbox: Optional[Tuple[int, int, int, int]] = None,
                          max_dimension: int = 2400,
                          ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Prétraitement + recadrage en deux étapes :
        Stage 1 : neatline (marges extérieures, titre, barre d'échelle)
        Stage 2 : légende interne droite   [remove_legend=True]

    Args:
        remove_legend : active la suppression de la légende interne.
                        Défaut True — recommandé pour toutes les cartes
                        AMS/GSGS 1:50 000 Tunisie/Algérie.
        manual_bbox   : force un recadrage manuel (x1,y1,x2,y2).
                        Prime sur auto_crop.

    Returns:
        (image_bgr_recadrée, image_hsv_recadrée, bbox_utilisée)
    """
    img = load_image(path)
    if max_dimension:
        img = downscale_if_too_large(img, max_dimension=max_dimension)
    if normalize_on:
        img = normalize_illumination(img)
    if denoise_on:
        img = denoise(img)

    H, W = img.shape[:2]

    # Stage 1 : neatline
    if manual_bbox is not None:
        x1, y1, x2, y2 = manual_bbox
    elif auto_crop:
        x1, y1, x2, y2 = detect_map_frame(img)
    else:
        x1, y1, x2, y2 = 0, 0, W, H

    img_cropped = crop_to_frame(img, (x1, y1, x2, y2))

    # Stage 2 : légende interne
    legend_x = 0
    if remove_legend and manual_bbox is None:
        legend_x = detect_legend_x(img_cropped)
        if legend_x > 0:
            img_cropped = img_cropped[:, :legend_x].copy()
            logger.info(f"Legend removed: kept x=0→{legend_x} "
                        f"({legend_x/img_cropped.shape[1]*100:.1f}% of crop width)")

    final_bbox = (x1, y1, x1 + img_cropped.shape[1], y2)
    hsv_cropped = to_hsv(img_cropped)

    logger.info(f"preprocess_with_crop: {W}×{H} → {img_cropped.shape[1]}×{img_cropped.shape[0]} "
                f"(legend_x={legend_x})")
    return img_cropped, hsv_cropped, final_bbox
