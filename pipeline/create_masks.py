"""
Création de masques GT pour cartes militaires Nord-Africaines 1:50,000
──────────────────────────────────────────────────────────────────────
Calibré sur 8 cartes réelles :
  Tunisia  : Bizerte, Tunis, Aïn El Kseïba
  Algeria  : Aïn Bessem, Alger, Terny, Warnier, Renault

Plages HSV calibrées sur données réelles (dataset_config.json)
Coverage observée : 14-50% selon densité de la carte

Usage :
  from create_masks import create_mask
  mask, report = create_mask(img_bgr)
"""

import json, logging
import cv2
import numpy as np
from pathlib import Path
from skimage import measure

logger = logging.getLogger(__name__)

# ── Charger la config calibrée ────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parents[1]
_cfg_path = _project_root / "data" / "dataset_config.json"
if not _cfg_path.exists():
    _cfg_path = Path(__file__).parent / "dataset_config.json"
_CFG = json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}

def _hsv(key):
    return _CFG.get("hsv_ranges", {}).get(key, {})

# Ranges calibrés sur 8 cartes réelles
_RED1_LO  = tuple(_hsv("red_roads_primary").get("low1",  [0,   60,  60]))
_RED1_HI  = tuple(_hsv("red_roads_primary").get("high1", [15, 255, 255]))
_RED2_LO  = tuple(_hsv("red_roads_primary").get("low2",  [155, 60,  60]))
_RED2_HI  = tuple(_hsv("red_roads_primary").get("high2", [180,255, 255]))
_VEG_LO   = tuple(_hsv("vegetation").get("low",  [35,  20,  50]))
_VEG_HI   = tuple(_hsv("vegetation").get("high", [95, 255, 220]))
_WAT_LO   = tuple(_hsv("water_blue").get("low",  [95,  40,  40]))
_WAT_HI   = tuple(_hsv("water_blue").get("high", [145,255, 200]))

_MIN_AREA_PX     = 8
_COV_LOW         = 5.0
_COV_HIGH        = 55.0    # terny 49.8% → limite 55%
_GRID_FRACTION   = 0.60


def create_mask(img_bgr: np.ndarray,
                use_frame_detection: bool = False) -> tuple:
    """
    Crée un masque GT binaire pour une carte militaire Nord-Africaine.

    Features détectées (calibrées sur 8 cartes réelles) :
      - Routes rouges primaires  (rouge HSV — coverage 6-21%)
      - Features sombres / routes noires (Otsu + filtre largeur adaptative)
      - Végétation verte (35,20,50)-(95,255,220) — importante sur Terny (35%)
      - Eau / hydrographie bleue

    Args:
        img_bgr:              image BGR (préférablement déjà cropée)
        use_frame_detection:  si True, applique MapFrameDetector en interne

    Returns:
        (mask uint8, report dict)
    """
    report = {}
    h, w   = img_bgr.shape[:2]

    # ── Frame detection optionnelle ───────────────────────────────────────────
    work = img_bgr
    if use_frame_detection:
        try:
            from .map_frame_detector import MapFrameDetector
            det    = MapFrameDetector()
            result = det.detect(img_bgr)
            if result.confidence > 0.5:
                work = result.image
                report['frame'] = f"{result.method} conf={result.confidence:.2f}"
        except Exception as e:
            report['frame_error'] = str(e)

    h, w   = work.shape[:2]
    gray   = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv    = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    # ── Calcul min_road_width sur les dimensions COMPLÈTES du crop ────────────
    # Bug précédent : mrw calculé sur des zones partielles (dimensions petites)
    # → filtre trop faible → texte/tirets capturés
    # Fix : utilise TOUJOURS min(full_w, full_h) pour le calcul
    mrw = max(3, min(w, h) // 200)

    # ── 1. Routes rouges primaires ────────────────────────────────────────────
    # Calibré : coverage 6-21% sur les 8 cartes
    red1   = cv2.inRange(hsv, _RED1_LO, _RED1_HI)
    red2   = cv2.inRange(hsv, _RED2_LO, _RED2_HI)
    red    = cv2.bitwise_or(red1, red2)
    k_red  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red    = cv2.dilate(red, k_red, iterations=2)   # combler les trous dans les lignes
    report['red_px']   = int((red > 0).sum())
    report['red_pct']  = round((red > 0).mean() * 100, 1)

    # ── 2. Features sombres — Otsu + filtre largeur adaptative ───────────────
    # mrw déjà calculé ci-dessus sur les dimensions complètes du crop
    _, dark_raw = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (mrw, mrw))
    dark   = cv2.morphologyEx(dark_raw, cv2.MORPH_OPEN, k_open, iterations=1)
    report['dark_px']        = int((dark > 0).sum())
    report['dark_raw_px']    = int((dark_raw > 0).sum())
    report['min_road_width'] = mrw

    # ── 3. Végétation verte ───────────────────────────────────────────────────
    # Importante : Terny 35%, Warnier 11%, Ain Bessem 8%
    green   = cv2.inRange(hsv, _VEG_LO, _VEG_HI)
    k_veg   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green   = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k_veg, iterations=2)
    green   = cv2.morphologyEx(green, cv2.MORPH_OPEN,  k_veg, iterations=1)
    report['green_px']  = int((green > 0).sum())
    report['green_pct'] = round((green > 0).mean() * 100, 1)

    # ── 4. Eau / Hydrographie ─────────────────────────────────────────────────
    water   = cv2.inRange(hsv, _WAT_LO, _WAT_HI)
    k_wat   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water   = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k_wat, iterations=2)
    report['water_px']  = int((water > 0).sum())

    # ── 5. Combiner toutes les features ──────────────────────────────────────
    combined = cv2.bitwise_or(dark, red)
    combined = cv2.bitwise_or(combined, green)
    combined = cv2.bitwise_or(combined, water)

    # ── 6. Supprimer le bruit (composantes < MIN_AREA_PX) ────────────────────
    # Calibré : supprime artéfacts scan sans toucher aux routes
    labels    = measure.label(combined)
    regions   = measure.regionprops(labels)
    clean     = np.zeros_like(combined)
    for r in regions:
        if r.area >= _MIN_AREA_PX:
            clean[labels == r.label] = 255

    noise_removed = int((combined > 0).sum()) - int((clean > 0).sum())
    report['noise_px_removed']  = noise_removed
    report['components_kept']   = sum(1 for r in regions if r.area >= _MIN_AREA_PX)
    report['components_total']  = len(regions)

    # ── 7. Qualité ────────────────────────────────────────────────────────────
    cov = float((clean > 0).mean()) * 100
    report['coverage_pct'] = round(cov, 1)

    if cov < _COV_LOW:
        report['warning'] = f"Coverage {cov:.1f}% < {_COV_LOW}% — under-detection"
    elif cov > _COV_HIGH:
        report['warning'] = f"Coverage {cov:.1f}% > {_COV_HIGH}% — over-segmentation"
    else:
        report['status'] = 'ok'

    return clean, report
