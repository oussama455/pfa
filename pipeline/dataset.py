"""
Dataset PyTorch pour la segmentation de cartes historiques.

Charge les images RGB et les labels (codes en couleur BGR) du dataset
SODUCO/Benchmark Historical Maps. Les labels sont convertis en index de
classe (uint8) pour la cross-entropy.

Usage :
    from pipeline.dataset import MapSegDataset
    ds = MapSegDataset(
        images_dir='data/historical_maps/train/images',
        labels_dir='data/historical_maps/train/labels',
        classes_json='data/historical_maps/classes.json',
        target_size=(512, 512),
        augment=True,
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    Dataset = object  # type: ignore

try:
    import albumentations as A
    ALBU_OK = True
except ImportError:
    ALBU_OK = False


# ---------------------------------------------------------------------
# Helper : conversion image-de-couleurs -> masque d'index
# ---------------------------------------------------------------------
def _build_palette(classes_data: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne :
        - palette_bgr : Nx3 array uint8 (couleurs BGR)
        - class_ids   : N array uint8 (id de classe correspondant)
    """
    palette_bgr = np.array([c["color_bgr"] for c in classes_data], dtype=np.uint8)
    class_ids = np.array([c["id"] for c in classes_data], dtype=np.uint8)
    return palette_bgr, class_ids


def color_label_to_index(label_bgr: np.ndarray,
                          palette_bgr: np.ndarray,
                          class_ids: np.ndarray,
                          ignore_index: int = 255) -> np.ndarray:
    """
    Convertit un label HxWx3 (couleurs BGR de la palette) en HxW (index).

    Les pixels qui ne matchent aucune couleur de la palette reçoivent
    `ignore_index` (255 par défaut), ce qui permet à la CrossEntropy de
    les ignorer avec ignore_index=255.
    """
    H, W = label_bgr.shape[:2]
    out = np.full((H, W), ignore_index, dtype=np.uint8)
    flat = label_bgr.reshape(-1, 3)
    out_flat = out.reshape(-1)
    for color, cls_id in zip(palette_bgr, class_ids):
        mask = (flat == color).all(axis=1)
        out_flat[mask] = cls_id
    return out


def index_to_color(label_idx: np.ndarray,
                    palette_bgr: np.ndarray,
                    class_ids: np.ndarray) -> np.ndarray:
    """Reconvertit un masque d'index HxW en image couleur HxWx3 BGR."""
    H, W = label_idx.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for color, cls_id in zip(palette_bgr, class_ids):
        out[label_idx == cls_id] = color
    return out


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class MapSegDataset(Dataset):
    """
    Dataset PyTorch pour la segmentation sémantique de cartes.

    Conventions :
        - images_dir / labels_dir contiennent des fichiers de même nom.
        - Les images sont en RGB après chargement (conversion OpenCV).
        - Les labels sont convertis du codage couleur BGR vers un index
          de classe uint8 selon `classes_json`.

    Si `target_size` est fourni, redimensionne avec INTER_AREA pour
    l'image et INTER_NEAREST pour le label (préservation des classes).
    """

    def __init__(self,
                 images_dir: str | Path,
                 labels_dir: str | Path,
                 classes_json_path: str | Path,
                 *,
                 target_size: Optional[Tuple[int, int]] = (512, 512),
                 augment: bool = False,
                 file_extensions: Tuple[str, ...] = (".png", ".jpg", ".tif", ".jpeg"),
                 ignore_index: int = 255,
                 normalize_imagenet: bool = True):
        if not TORCH_OK:
            raise ImportError(
                "PyTorch requis pour MapSegDataset. "
                "Installe : pip install torch"
            )
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.target_size = target_size
        self.augment = augment
        self.normalize_imagenet = normalize_imagenet

        with open(classes_json_path, "r", encoding="utf-8") as f:
            classes_data = json.load(f)
        self.classes = classes_data["classes"]
        self.num_classes = classes_data.get("num_classes", len(self.classes))
        # ignore_index dans le JSON est utilise par la classe `unknown` ;
        # mais en pratique pour la CrossEntropy on prefere 255.
        self.json_ignore = classes_data.get("ignore_index", -1)
        self.ignore_index = ignore_index

        self.palette_bgr, self.class_ids = _build_palette(self.classes)

        # Liste des fichiers (images dont un label correspondant existe)
        self.samples: List[Tuple[Path, Path]] = []
        for img in sorted(self.images_dir.iterdir()):
            if img.suffix.lower() not in file_extensions:
                continue
            lbl = self.labels_dir / img.name
            if lbl.exists():
                self.samples.append((img, lbl))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune paire (image, label) trouvee dans "
                f"{self.images_dir} et {self.labels_dir}."
            )

        # Pipeline d'augmentation (si albumentations dispo)
        self._aug = None
        if augment and ALBU_OK:
            self._aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # Image : BGR -> RGB
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Impossible de lire {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Label : BGR couleur -> index uint8
        label_bgr = cv2.imread(str(lbl_path), cv2.IMREAD_COLOR)
        if label_bgr is None:
            raise RuntimeError(f"Impossible de lire {lbl_path}")

        # Resize AVANT conversion en index pour utiliser INTER_NEAREST
        if self.target_size:
            tw, th = self.target_size
            image_rgb = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_AREA)
            label_bgr = cv2.resize(label_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)

        label_idx = color_label_to_index(
            label_bgr, self.palette_bgr, self.class_ids,
            ignore_index=self.ignore_index)

        # Marquer la classe "unknown" du JSON comme ignorée
        if 0 <= self.json_ignore < 255:
            label_idx[label_idx == self.json_ignore] = self.ignore_index

        # Augmentations
        if self._aug is not None:
            aug_out = self._aug(image=image_rgb, mask=label_idx)
            image_rgb = aug_out["image"]
            label_idx = aug_out["mask"]

        # Normalisation ImageNet (standard pour encodeurs pretrained)
        img = image_rgb.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        image_tensor = torch.from_numpy(img.copy())
        label_tensor = torch.from_numpy(label_idx.astype(np.int64).copy())
        return image_tensor, label_tensor


def palette_from_classes_json(classes_json_path: str | Path):
    """Helper pour charger juste la palette (utile pour le post-traitement)."""
    with open(classes_json_path, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    return _build_palette(classes_data["classes"])
