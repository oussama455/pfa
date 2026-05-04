"""
Prétraitement d'une carte raster avant segmentation.

Opérations :
    - Lecture de l'image
    - Débruitage léger (bilatéral : préserve les bords, utile pour les cartes)
    - Conversion d'espace colorimétrique (BGR → RGB / HSV)
    - Normalisation d'illumination (optionnel, pour cartes mal scannées)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """
    Charge une image depuis disque.

    Retourne une image au format BGR (convention OpenCV), avec 3 canaux.
    Lève FileNotFoundError si le fichier n'existe pas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Carte introuvable : {path}")

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {path}")

    return image_bgr


def downscale_if_too_large(image_bgr: np.ndarray, *,
                            max_dimension: int = 2400,
                            interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Redimensionne l'image si la plus grande dimension depasse max_dimension.

    Utile pour eviter les crashes "kernel died" sur des images > 4000 px sur
    des PCs avec RAM limitee. Pour le 1:50000, 2400 px de large est largement
    suffisant pour la segmentation couleur (1 px ~ 12 m sur le terrain).

    Conserve le ratio. Si l'image est deja plus petite, retourne telle quelle.
    """
    H, W = image_bgr.shape[:2]
    longest = max(H, W)
    if longest <= max_dimension:
        return image_bgr
    scale = max_dimension / longest
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)


def denoise(image_bgr: np.ndarray, d: int = 9, sigma_color: int = 75,
            sigma_space: int = 75) -> np.ndarray:
    """
    Débruite en préservant les bords (filtre bilatéral).

    Paramètres par défaut adaptés à des cartes scannées à 300 DPI.
    Pour des cartes très bruitées, augmenter sigma_color.
    """
    return cv2.bilateralFilter(image_bgr, d=d, sigmaColor=sigma_color,
                                sigmaSpace=sigma_space)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Convertit BGR → HSV (pour seuillage couleur)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convertit BGR → RGB (pour affichage matplotlib)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_illumination(image_bgr: np.ndarray) -> np.ndarray:
    """
    Normalise l'illumination via CLAHE sur le canal L (Lab).

    Utile si la carte a été scannée avec un éclairage inhomogène.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess(path: str | Path, *,
               denoise_on: bool = True,
               normalize_on: bool = False,
               max_dimension: int = 2400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline de prétraitement complet.

    Si l'image est plus large que max_dimension, elle est redimensionnee
    en preservant le ratio. Cela evite les crashs "kernel died" lies a
    l'OOM sur les images > 4000 px et acceleree fortement le pipeline.
    Mettre max_dimension=None pour conserver la resolution d'origine.

    Retourne un tuple (image_bgr_prête, image_hsv).
    """
    img = load_image(path)
    if max_dimension:
        img = downscale_if_too_large(img, max_dimension=max_dimension)
    if normalize_on:
        img = normalize_illumination(img)
    if denoise_on:
        img = denoise(img)
    hsv = to_hsv(img)
    return img, hsv


# -----------------------------------------------------------
# Détection et recadrage du cadre cartographique (neatline)
# -----------------------------------------------------------
def detect_map_frame(image_bgr: np.ndarray, *,
                     dark_threshold: int = 80,
                     min_area_ratio: float = 0.30,
                     max_area_ratio: float = 0.95,
                     pad: int = 5,
                     aspect_ratio_range: Tuple[float, float] = (0.5, 2.5),
                     prefer_centered: bool = True,
                     ) -> Tuple[int, int, int, int]:
    """
    Détecte automatiquement le cadre rectangulaire de la zone cartographique.

    Méthode : binarise l'image, trouve les contours rectangulaires plausibles
    (taille raisonnable, ratio d'aspect carte, contenant le centre de l'image),
    et sélectionne le meilleur candidat. Sur les cartes militaires TUNIS
    1:50000, le neatline est une ligne noire double délimitant la zone utile.

    Trois critères supplémentaires par rapport à la version naïve :
        1. Filtre par aspect ratio : le neatline d'une carte topo a un ratio
           h/w typiquement entre 0.5 et 2.5. Élimine la légende qui est
           souvent étroite (h/w > 3 ou < 0.3).
        2. Préférence pour les contours qui CONTIENNENT le centre de l'image
           (le neatline est centré, la légende est en marge).
        3. Approximation polygonale : un vrai neatline a 4 sommets après
           cv2.approxPolyDP — on privilégie ces contours.

    Paramètres :
        dark_threshold     : seuil binarisation pour traits noirs.
        min_area_ratio     : couverture minimum (fraction de l'image).
        max_area_ratio     : couverture maximum (évite le bord du scan).
        pad                : marge de sécurité (pixels) après détection.
        aspect_ratio_range : (min, max) du rapport h/w accepté.
        prefer_centered    : True = privilégie les contours contenant le centre.

    Retourne (x1, y1, x2, y2) — coordonnées du rectangle intérieur.
    Si aucun cadre n'est détecté, retourne l'image entière.
    """
    H, W = image_bgr.shape[:2]
    img_area = W * H
    cx, cy = W // 2, H // 2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, W, H)

    candidates = []
    ar_min, ar_max = aspect_ratio_range

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # 1) taille
        if area < min_area_ratio * img_area:
            continue
        if area > max_area_ratio * img_area:
            continue
        # 2) aspect ratio
        aspect = h / max(w, 1)
        if aspect < ar_min or aspect > ar_max:
            continue
        # 3) le contour doit contenir (ou presque) le centre de l'image
        contains_center = (x <= cx <= x + w) and (y <= cy <= y + h)

        # Score : surface normalisee + bonus si centré + bonus si quadrilatère
        score = area / img_area
        if contains_center:
            score += 0.5
        # Bonus si le contour est approximable par 4 sommets (quadrilatère)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            score += 0.2
        # Bonus si le rectangle est "rempli" par le contour (vrai neatline
        # rectangulaire plutôt qu'une forme libre)
        cnt_area = cv2.contourArea(cnt)
        fill_ratio = cnt_area / max(area, 1)
        if fill_ratio < 0.50:
            # contour très creux (ex: anneau du quadrillage extérieur) :
            # ça reste valable mais on ne booste pas
            pass
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        return (0, 0, W, H)

    # Tri par score décroissant
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, (x, y, w, h) = candidates[0]

    x1 = max(0, x + pad)
    y1 = max(0, y + pad)
    x2 = min(W, x + w - pad)
    y2 = min(H, y + h - pad)
    return (x1, y1, x2, y2)


def crop_to_frame(image: np.ndarray,
                  bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Recadre une image (BGR ou HSV ou masque) à la bbox (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def preprocess_with_crop(path: str | Path, *,
                         denoise_on: bool = True,
                         normalize_on: bool = False,
                         auto_crop: bool = True,
                         manual_bbox: Tuple[int, int, int, int] | None = None,
                         max_dimension: int = 2400,
                         ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Prétraitement + recadrage sur le cadre cartographique.

    Si manual_bbox est fourni, il prime sur la détection automatique.
    Si auto_crop=False et manual_bbox=None, aucun recadrage n'est appliqué.

    max_dimension : si l'image est plus large que cette valeur, elle est
        redimensionnee EN AMONT (avant detection du cadre). Cela evite les
        plantages "kernel died" / OOM sur les cartes haute resolution.
        Mettre max_dimension=None pour conserver la resolution d'origine.
        Le bbox retourne est dans le repere de l'image (potentiellement
        redimensionnee).

    Retourne (image_bgr_recadrée, image_hsv_recadrée, bbox_utilisée).
    Le bbox sert à reprojetter les coordonnées vectorielles vers l'image
    d'origine si besoin.
    """
    img = load_image(path)
    if max_dimension:
        img = downscale_if_too_large(img, max_dimension=max_dimension)
    if normalize_on:
        img = normalize_illumination(img)
    if denoise_on:
        img = denoise(img)

    H, W = img.shape[:2]
    if manual_bbox is not None:
        bbox = manual_bbox
    elif auto_crop:
        bbox = detect_map_frame(img)
    else:
        bbox = (0, 0, W, H)

    img_cropped = crop_to_frame(img, bbox)
    hsv_cropped = to_hsv(img_cropped)
    return img_cropped, hsv_cropped, bbox
