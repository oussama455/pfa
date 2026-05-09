"""
Map Frame Detector
──────────────────
Tested on 8 real military maps (Tunisia + Algeria, 1:50,000 WWII):
  Bizerte, Tunis, Aïn El Kseïba, Aïn Bessem, Alger, Terny, Warnier, Renault

TWO-STAGE approach:
  Stage 1: Neatline detection  → removes outer margin, title, scale bar
  Stage 2: Legend detection    → removes internal legend (right ~12-15%)

Legend is INSIDE the neatline on all tested maps → must be handled separately.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CropResult:
    image:          np.ndarray
    x1: int; y1: int
    x2: int; y2: int
    method:         str
    confidence:     float
    original_shape: Tuple[int, int]
    legend_x:       int = -1   # x-coord of legend separator (-1 = not found)


class MapFrameDetector:
    """
    Two-stage crop:
      1. Find and crop to neatline (outer map border)
      2. Detect and remove internal legend panel (right side)
    """

    def __init__(self,
                 min_map_fraction: float = 0.30,
                 border_thickness_range: Tuple[int, int] = (2, 20),
                 margin_fallback_pct: float = 0.04):
        self.min_map_fraction       = min_map_fraction
        self.border_thickness_range = border_thickness_range
        self.margin_fallback_pct    = margin_fallback_pct

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def detect(self, image_bgr: np.ndarray) -> CropResult:
        """
        Full two-stage crop.
        Returns image with outer margin AND legend removed.
        """
        H, W = image_bgr.shape[:2]

        # ── Stage 1: Neatline ─────────────────────────────────────────────
        result = self._detect_neatline(image_bgr)
        if result:
            x1, y1, x2, y2, conf = result
            method = "neatline"
        else:
            result2 = self._detect_content_region(image_bgr)
            if result2:
                x1, y1, x2, y2, conf = result2
                method = "content_region"
            else:
                x1, y1, x2, y2 = self._margin_trim(H, W)
                conf = 0.3
                method = "margin_trim"

        logger.info(f"MapFrame stage1 ({method}): ({x1},{y1})→({x2},{y2}) conf={conf:.2f}")

        # ── Stage 2: Legend removal ───────────────────────────────────────
        cropped_stage1 = image_bgr[y1:y2, x1:x2].copy()
        legend_x_local = self._detect_legend(cropped_stage1)

        if legend_x_local > 0:
            # Exclude legend → crop right side
            final_img = cropped_stage1[:, :legend_x_local].copy()
            legend_x_global = x1 + legend_x_local
            logger.info(f"MapFrame stage2 (legend): removed x={legend_x_local}→end "
                        f"({cropped_stage1.shape[1]-legend_x_local}px)")
        else:
            final_img = cropped_stage1
            legend_x_global = -1
            logger.info("MapFrame stage2: no legend detected")

        return CropResult(
            image          = final_img,
            x1=x1, y1=y1,
            x2=x1 + final_img.shape[1],
            y2=y2,
            method         = method,
            confidence     = conf,
            original_shape = (H, W),
            legend_x       = legend_x_global,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: NEATLINE DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _detect_neatline(self, img: np.ndarray) -> Optional[Tuple]:
        """
        Find the thick rectangular neatline.
        Adds inner margin of 1.5% to exclude border ticks and coord numbers.
        """
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best      = None
        best_area = 0

        for canny_lo, canny_hi in [(20, 80), (30, 100), (50, 150)]:
            edges  = cv2.Canny(gray, canny_lo, canny_hi)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges  = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_map_fraction * H * W:
                    continue
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
                if not (4 <= len(approx) <= 6):
                    continue
                x, y, w, h = cv2.boundingRect(approx)
                if w < 0.3 * W or h < 0.3 * H:
                    continue
                fill_ratio = area / (w * h)
                if fill_ratio < 0.75:
                    continue
                if area > best_area:
                    best_area = area
                    best = (x, y, w, h, fill_ratio)

        if best is None:
            return None

        x, y, w, h, fill_ratio = best

        # Inner margin: neatline line + coord ticks + border numbers (~1.5%)
        inner_x = max(int(W * 0.015), self.border_thickness_range[1])
        inner_y = max(int(H * 0.015), self.border_thickness_range[1])

        x1 = max(0, x + inner_x)
        y1 = max(0, y + inner_y)
        x2 = min(W, x + w - inner_x)
        y2 = min(H, y + h - inner_y)

        coverage   = (x2 - x1) * (y2 - y1) / (H * W)
        confidence = fill_ratio * min(1.0, coverage / 0.6)
        return (x1, y1, x2, y2, round(confidence, 2))

    def _detect_content_region(self, img: np.ndarray) -> Optional[Tuple]:
        H, W = img.shape[:2]
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        blur = cv2.GaussianBlur(l, (31, 31), 0)
        diff = cv2.absdiff(l, blur)
        _, content = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, k, iterations=3)
        content = cv2.morphologyEx(content, cv2.MORPH_OPEN,  k, iterations=2)
        coords = cv2.findNonZero(content)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        if w < 0.3 * W or h < 0.3 * H:
            return None
        pad = 10
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
        return (x1, y1, x2, y2, 0.55)

    def _margin_trim(self, H: int, W: int) -> Tuple[int,int,int,int]:
        pct = self.margin_fallback_pct
        return int(W*pct), int(H*pct), int(W*(1-pct)), int(H*(1-pct))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: LEGEND DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _detect_legend(self, cropped: np.ndarray) -> int:
        """
        Detect the internal legend panel (right side) and return its x position.
        Returns 0 if no legend found (use full width).

        Two methods tested on 8 real maps:
          Method A: vertical dark separator line (Tunisia maps — Bizerte, Tunis)
                    → dark column with >25% black pixels
          Method B: high-brightness + no red roads (Algeria maps)
                    → column where map becomes beige and roadless

        Returns:
            x coordinate to crop at (0 = no legend detected)
        """
        h, w   = cropped.shape[:2]
        gray   = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        hsv    = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Search only right 30% (legend never starts before 70%)
        search_start = int(w * 0.70)

        # ── Method A: Dark vertical separator line ────────────────────────
        # Tunisia maps: thick dark line separates map content from legend
        col_dark = (gray < 80).sum(axis=0)
        dark_threshold = h * 0.25   # column is separator if >25% pixels dark
        candidates_a = np.where(col_dark[search_start:] > dark_threshold)[0]

        if len(candidates_a) > 0:
            lx = int(candidates_a[0]) + search_start
            # Verify: right of this line should be brighter (legend bg)
            if lx + 20 < w:
                right_mean = gray[:, lx+5:lx+50].mean() if lx+50 < w else 0
                if right_mean > 180:   # bright = legend background
                    logger.info(f"Legend (method A — separator): x={lx} ({lx/w*100:.1f}%)")
                    return lx

        # ── Method B: Brightness + absence of red roads ───────────────────
        # Algeria maps: no clear separator, but legend is beige with no roads
        r1    = cv2.inRange(hsv, (0,   60, 60), (15,  255, 255))
        r2    = cv2.inRange(hsv, (155, 60, 60), (180, 255, 255))
        red   = cv2.bitwise_or(r1, r2)

        col_mean_bright = np.mean(gray[:, search_start:], axis=0)
        col_red_sum     = red[:, search_start:].sum(axis=0)

        # Smooth over 10 columns to reduce noise
        sm_bright = np.convolve(col_mean_bright, np.ones(10)/10, 'valid')
        sm_red    = np.convolve(col_red_sum,    np.ones(10)/10, 'valid')

        # Legend: bright (>210) AND essentially no red roads (<50 px)
        candidates_b = np.where((sm_bright > 210) & (sm_red < 50))[0]

        if len(candidates_b) > 0:
            lx = int(candidates_b[0]) + search_start
            logger.info(f"Legend (method B — bright+no_red): x={lx} ({lx/w*100:.1f}%)")
            return lx

        # ── Method C: Hard fallback ───────────────────────────────────────
        # If nothing detected → apply conservative 87% crop
        # (all 8 tested maps had legend starting between 86-95%)
        fallback_x = int(w * 0.87)
        # Only apply if right strip looks different from map content
        right_strip_std = gray[:, fallback_x:].std()
        left_strip_std  = gray[:, search_start:fallback_x].std()

        if right_strip_std < left_strip_std * 0.7:
            # Right strip is more uniform → likely legend
            logger.info(f"Legend (method C — fallback 87%): x={fallback_x}")
            return fallback_x

        logger.info("Legend: not detected (keeping full width)")
        return 0


# ── Debug helper ─────────────────────────────────────────────────────────────

def draw_crop_debug(original: np.ndarray, result: CropResult) -> np.ndarray:
    vis = original.copy()
    # Green box = map content after legend removal
    cv2.rectangle(vis, (result.x1, result.y1), (result.x2, result.y2),
                  (0, 255, 0), 3)
    # Red line = legend separator
    if result.legend_x > 0:
        cv2.line(vis, (result.legend_x, result.y1),
                 (result.legend_x, result.y2), (0, 0, 255), 2)
    label = f"{result.method} conf={result.confidence:.2f}"
    if result.legend_x > 0:
        label += f" | legend@{result.legend_x}"
    cv2.putText(vis, label, (result.x1, max(0, result.y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis
