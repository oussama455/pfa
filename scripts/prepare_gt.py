"""
Pipeline de préparation des données GT pour le fine-tuning
──────────────────────────────────────────────────────────
Ordre d'exécution (GIGO — qualité des données = qualité du modèle) :

  ÉTAPE 1 : Détection du cadre (map_frame_detector)
             → supprime legend, hachures, texte de marge

  ÉTAPE 2 : Filtrage des lignes de grille (grid filter)
             → supprime la grille UTM/Lambert qui polluera le masque

  ÉTAPE 3 : Création du masque GT (create_masks)
             → détecte routes rouges + features sombres + eau

  ÉTAPE 4 : Génération de l'image QA (3 panneaux)
             → vérification visuelle avant tout entraînement

Usage :
  python prepare_gt.py --img_dir data/tunis/images --out_dir data/tunis/gt

Structure attendue en entrée :
  data/tunis/images/   carte1.jpg  carte2.jpg  ...

Structure produite :
  data/tunis/gt/
    masks/      carte1.png  carte2.png  ...  (masques GT, prêts pour training)
    cropped/    carte1.jpg  ...              (cartes sans legend, pour debug)
    qa/         carte1_qa.png ...            (images de vérification QA)
"""

import sys, os, argparse, logging
import cv2
import numpy as np
from pathlib import Path

# ── Imports projet ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "soduco_unet"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mapvec.settings")
try:
    import django; django.setup()
except Exception:
    pass

from .map_frame_detector import MapFrameDetector
from .create_masks import create_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : FILTRAGE DES LIGNES DE GRILLE
# ═══════════════════════════════════════════════════════════════════════════════

def detect_grid_lines(img_bgr: np.ndarray,
                      min_line_fraction: float = 0.60) -> np.ndarray:
    """
    Détecte les lignes de grille (UTM, Lambert, coordonnées).

    Les lignes de grille sont des lignes droites continues qui traversent
    au moins `min_line_fraction` de la largeur ou hauteur de l'image.

    Args:
        img_bgr:            image BGR
        min_line_fraction:  fraction minimale pour considérer une ligne
                            comme grille (0.6 = 60% de la largeur/hauteur)

    Returns:
        Masque binaire (uint8) où 255 = ligne de grille détectée
    """
    h, w    = img_bgr.shape[:2]
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Binarisation Otsu
    _, bin_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Lignes horizontales ───────────────────────────────────────────────────
    # Kernel très large horizontalement = ne détecte que les lignes continues
    min_h_len   = int(w * min_line_fraction)
    kh          = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_lines     = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kh)
    # Dilater légèrement pour capturer toute l'épaisseur
    h_lines     = cv2.dilate(h_lines,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
                             iterations=1)

    # ── Lignes verticales ─────────────────────────────────────────────────────
    min_v_len   = int(h * min_line_fraction)
    kv          = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_lines     = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kv)
    v_lines     = cv2.dilate(v_lines,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
                             iterations=1)

    grid_mask = cv2.bitwise_or(h_lines, v_lines)
    return grid_mask


def remove_grid_from_mask(mask: np.ndarray,
                           grid_mask: np.ndarray) -> np.ndarray:
    """Supprime les pixels de grille du masque GT."""
    # Dilater légèrement la grille pour s'assurer de supprimer les bords
    grid_dilated = cv2.dilate(
        grid_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )
    return cv2.bitwise_and(mask, cv2.bitwise_not(grid_dilated))


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : IMAGE QA (4 panneaux)
# ═══════════════════════════════════════════════════════════════════════════════

def make_qa_image(original: np.ndarray,
                  cropped: np.ndarray,
                  mask: np.ndarray,
                  grid_mask: np.ndarray,
                  report: dict,
                  filename: str) -> np.ndarray:
    """
    Génère une image de vérification à 4 panneaux :
      [1] Originale (avec frame box)
      [2] Après crop + grille détectée (rouge)
      [3] Masque GT final (binaire)
      [4] Overlay final (vert = feature retenu)
    """
    # Taille cible pour chaque panneau
    target_w, target_h = 320, 240

    def fit(img, to_gray=False):
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        out   = cv2.resize(img, (int(w * scale), int(h * scale)))
        # Pad to target size
        ph    = target_h - out.shape[0]
        pw    = target_w - out.shape[1]
        out   = cv2.copyMakeBorder(out, 0, ph, 0, pw,
                                   cv2.BORDER_CONSTANT, value=(40, 40, 40))
        if to_gray and len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    # Panneau 1 : Originale + message
    p1 = fit(original)
    _label(p1, "1. Originale", (200, 200, 200))

    # Panneau 2 : Cropped + grille surlignée en rouge
    p2 = fit(cropped.copy())
    if (grid_mask > 0).any():
        gm_fit = cv2.resize(grid_mask, (p2.shape[1], p2.shape[0]))
        p2[gm_fit > 0] = [0, 0, 200]   # rouge = lignes grille supprimées
    n_grid_px = int((grid_mask > 0).sum())
    _label(p2, f"2. Cadre coupé | grille: {n_grid_px}px supprimés", (0, 0, 220))

    # Panneau 3 : Masque GT binaire
    p3 = fit(mask, to_gray=True)
    cov = report.get('coverage_pct', '?')
    _label(p3, f"3. Masque GT | couverture: {cov}%", (200, 200, 200))

    # Panneau 4 : Overlay vert sur carte coupée
    overlay = cropped.copy()
    overlay[mask[: cropped.shape[0], : cropped.shape[1]] > 0] = [30, 180, 30]
    p4 = fit(cv2.addWeighted(cropped, 0.45, overlay, 0.55, 0))
    n_feat = report.get('components_kept', '?')
    _label(p4, f"4. Overlay | features: {n_feat}", (30, 200, 30))

    # Assembler en 2×2
    top    = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    qa     = np.vstack([top, bottom])

    # Titre
    cv2.rectangle(qa, (0, 0), (qa.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(qa, f"QA — {filename}", (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 100), 1)

    return qa


def _label(img: np.ndarray, text: str, color=(200, 200, 200)):
    cv2.rectangle(img, (0, img.shape[0] - 20), (img.shape[1], img.shape[0]),
                  (20, 20, 20), -1)
    cv2.putText(img, text, (4, img.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def process_one(img_path: Path,
                out_mask: Path,
                out_cropped: Path,
                out_qa: Path,
                detector: MapFrameDetector,
                grid_fraction: float = 0.60) -> dict:
    """
    Traite une seule carte militaire.
    Retourne un rapport de qualité.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "error": "cannot read"}

    h_orig, w_orig = img.shape[:2]

    # ── ÉTAPE 1 : Détection du cadre ─────────────────────────────────────────
    frame_result = detector.detect(img)
    cropped      = frame_result.image
    logger.info(
        f"[{img_path.name}] Frame: {frame_result.method} "
        f"conf={frame_result.confidence:.2f} "
        f"→ {cropped.shape[1]}×{cropped.shape[0]} px"
    )

    # ── ÉTAPE 2 : Détection + mémorisation des lignes de grille ─────────────
    grid_mask   = detect_grid_lines(cropped, min_line_fraction=grid_fraction)
    n_grid_px   = int((grid_mask > 0).sum())
    grid_pct    = n_grid_px / (cropped.shape[0] * cropped.shape[1]) * 100
    logger.info(
        f"[{img_path.name}] Grille: {n_grid_px} px ({grid_pct:.1f}%) supprimés"
    )

    # ── ÉTAPE 3 : Masque GT sur image coupée (sans legend, sans grille) ──────
    raw_mask, mask_report = create_mask(cropped, use_frame_detection=False)

    # Supprimer les lignes de grille du masque
    if n_grid_px > 0:
        raw_mask = remove_grid_from_mask(raw_mask, grid_mask)

    mask_report["grid_px_removed"] = n_grid_px
    mask_report["frame_method"]    = frame_result.method
    mask_report["frame_conf"]      = frame_result.confidence

    # ── ÉTAPE 4 : Sauvegarder + QA ───────────────────────────────────────────
    cv2.imwrite(str(out_mask),    raw_mask)
    cv2.imwrite(str(out_cropped), cropped)

    qa_img = make_qa_image(img, cropped, raw_mask,
                           grid_mask, mask_report, img_path.name)
    cv2.imwrite(str(out_qa), qa_img)

    # ── Rapport final ─────────────────────────────────────────────────────────
    report = {
        "file":            img_path.name,
        "original_size":   f"{w_orig}×{h_orig}",
        "cropped_size":    f"{cropped.shape[1]}×{cropped.shape[0]}",
        "frame_method":    frame_result.method,
        "frame_conf":      round(frame_result.confidence, 2),
        "grid_px_removed": n_grid_px,
        "coverage_pct":    mask_report["coverage_pct"],
        "components_kept": mask_report["components_kept"],
        "noise_removed":   mask_report["noise_px_removed"],
        "status":          "ok" if 3 <= mask_report["coverage_pct"] <= 35 else "warn",
        "warning":         mask_report.get("warning", ""),
    }
    return report


def process_directory(img_dir: Path, out_dir: Path,
                      grid_fraction: float = 0.60):
    """Traite tout un dossier de cartes."""
    masks_dir   = out_dir / "masks"
    cropped_dir = out_dir / "cropped"
    qa_dir      = out_dir / "qa"
    for d in [masks_dir, cropped_dir, qa_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exts    = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images  = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in exts]

    if not images:
        logger.error(f"Aucune image dans {img_dir}")
        return

    logger.info(f"Traitement de {len(images)} carte(s)")
    logger.info("Pipeline : Frame → Grille → Masque → QA")
    print()

    detector = MapFrameDetector()
    reports  = []

    for img_path in images:
        report = process_one(
            img_path    = img_path,
            out_mask    = masks_dir  / (img_path.stem + ".png"),
            out_cropped = cropped_dir / img_path.name,
            out_qa      = qa_dir     / (img_path.stem + "_qa.png"),
            detector    = detector,
            grid_fraction = grid_fraction,
        )
        reports.append(report)

        ok = "✓" if report.get("status") == "ok" else "⚠"
        print(f"  {ok}  {report['file']}")
        print(f"     Taille     : {report['original_size']} → {report['cropped_size']}")
        print(f"     Frame      : {report['frame_method']} (conf={report['frame_conf']})")
        print(f"     Grille     : {report['grid_px_removed']} px supprimés")
        print(f"     Couverture : {report['coverage_pct']}%  "
              f"({report['components_kept']} features, "
              f"bruit: {report['noise_removed']} px)")
        if report.get("warning"):
            print(f"     ⚠  {report['warning']}")
        print()

    # Résumé
    ok_count = sum(1 for r in reports if r.get("status") == "ok")
    print("─" * 55)
    print(f"Résultat : {ok_count}/{len(reports)} cartes prêtes pour le fine-tuning")
    print(f"\nFichiers produits :")
    print(f"  Masques GT  → {masks_dir}")
    print(f"  Cartes crop → {cropped_dir}")
    print(f"  Images QA   → {qa_dir}")
    print()
    print("Prochaine étape :")
    print("  1. Ouvrez le dossier qa/ et vérifiez chaque image (4 panneaux)")
    print("  2. Si les masques sont bons → lancez le fine-tuning :")
    print(f"     python finetune_tunis.py \\")
    print(f"       --weights ../soduco_unet/pretrain_weight/unet_best_weight.pth \\")
    print(f"       --img_dir  {cropped_dir} \\")
    print(f"       --gt_dir   {masks_dir} \\")
    print(f"       --output   finetuned_tunis.pth --epochs 50")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Pipeline GT : Frame → Grille → Masque → QA"
    )
    p.add_argument("--img_dir", required=True,
                   help="Dossier images sources (cartes complètes)")
    p.add_argument("--out_dir", default="data/tunis/gt",
                   help="Dossier de sortie (masks/, cropped/, qa/)")
    p.add_argument("--grid_fraction", type=float, default=0.60,
                   help="Fraction min pour détecter une ligne de grille (0.5-0.8)")
    args = p.parse_args()

    process_directory(
        img_dir       = Path(args.img_dir),
        out_dir       = Path(args.out_dir),
        grid_fraction = args.grid_fraction,
    )


if __name__ == "__main__":
    main()
