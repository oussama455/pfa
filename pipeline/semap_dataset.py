"""
SemapDataset — chargeur PyTorch pour le dataset SEMAP (Petitpierre 2025).

Differences avec MapSegDataset (SODUCO/historical_maps) :
    - Labels = uint8 single-channel avec index direct (0-5), pas de couleurs BGR
    - Dataset HORS du repo Git : chemin lu depuis data/semap_config.json
    - Splits officiels : partitions/{train,val,test}.txt
    - Permet d'inclure ou exclure les images synthetiques (12 122 images)

Usage :
    from pipeline.semap_dataset import SemapDataset
    train_ds = SemapDataset(split='train', target_size=(512, 512), augment=True)
    val_ds   = SemapDataset(split='val',   target_size=(512, 512), augment=False)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

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


def _project_root() -> Path:
    """Racine du projet (parent du dossier pipeline/)."""
    return Path(__file__).resolve().parent.parent


def _load_config(config_path: Optional[str | Path] = None) -> dict:
    """Charge data/semap_config.json par defaut."""
    if config_path is None:
        config_path = _project_root() / "data" / "semap_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_external_root(config: dict) -> Path:
    """Resout le chemin externe + valide qu'il existe."""
    p = Path(config["external_root"])
    if not p.exists():
        raise FileNotFoundError(
            f"SEMAP external_root introuvable : {p}\n"
            f"Edite data/semap_config.json -> 'external_root' avec le bon chemin."
        )
    return p


class SemapDataset(Dataset):
    """
    Dataset SEMAP avec splits officiels.

    Args :
        split             : 'train' | 'val' | 'test'
        target_size       : (W, H) pour resize. Aucun resize si None.
        augment           : True -> applique les augmentations albumentations.
        include_synthetic : False -> ne garde que les images reelles (1 439).
                            True (defaut) -> garde tout (13 561 echantillons).
        config_path       : chemin custom vers semap_config.json.
        normalize_imagenet: normalisation ImageNet (defaut True pour les
                            encodeurs pre-entraines).
    """

    def __init__(self,
                 split: str = "train",
                 *,
                 target_size: Optional[Tuple[int, int]] = (512, 512),
                 augment: bool = False,
                 include_synthetic: bool = True,
                 config_path: Optional[str | Path] = None,
                 normalize_imagenet: bool = True,
                 ignore_index: int = 255):
        if not TORCH_OK:
            raise ImportError("PyTorch requis. pip install torch")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split invalide : {split}. Choisis train|val|test.")

        self.config = _load_config(config_path)
        self.root = resolve_external_root(self.config)
        self.split = split
        self.target_size = target_size
        self.augment = augment
        self.include_synthetic = include_synthetic
        self.normalize_imagenet = normalize_imagenet
        self.ignore_index = ignore_index

        self.classes = self.config["classes"]
        self.num_classes = self.config["num_classes"]

        # Charge la liste des fichiers depuis partitions/{split}.txt
        partitions_dir = self.root / self.config["subdirs"]["partitions"]
        partition_file = partitions_dir / f"{split}.txt"
        if not partition_file.exists():
            raise FileNotFoundError(f"Partition introuvable : {partition_file}")

        with open(partition_file, "r", encoding="utf-8") as f:
            stems = [line.strip() for line in f if line.strip()]

        # Pour chaque stem, on trouve le sous-dossier (real ou synthetic)
        # NOTE : les noms de fichiers synthetiques sont du genre "12_output_591"
        # alors que les reels sont du genre "bnf_001250-7_6345_1736".
        # On essaie d'abord real, puis synthetic.
        img_real_dir = self.root / self.config["subdirs"]["images_real"]
        img_syn_dir  = self.root / self.config["subdirs"]["images_synthetic"]
        lbl_real_dir = self.root / self.config["subdirs"]["labels_real"]
        lbl_syn_dir  = self.root / self.config["subdirs"]["labels_synthetic"]

        img_ext = self.config.get("image_extension", ".jpg")
        lbl_ext = self.config.get("label_extension", ".png")

        self.samples: List[Tuple[Path, Path, str]] = []
        skipped = 0
        for stem in stems:
            # Test real d'abord
            img_real = img_real_dir / f"{stem}{img_ext}"
            lbl_real = lbl_real_dir / f"{stem}{lbl_ext}"
            if img_real.exists() and lbl_real.exists():
                self.samples.append((img_real, lbl_real, "real"))
                continue
            # Sinon synthetic
            if include_synthetic:
                img_syn = img_syn_dir / f"{stem}{img_ext}"
                lbl_syn = lbl_syn_dir / f"{stem}{lbl_ext}"
                if img_syn.exists() and lbl_syn.exists():
                    self.samples.append((img_syn, lbl_syn, "synthetic"))
                    continue
            skipped += 1

        if not self.samples:
            raise RuntimeError(
                f"Aucun echantillon trouve pour split={split}. "
                f"Verifie external_root={self.root} et la partition {partition_file}."
            )

        # Pipeline d'augmentation (albumentations si dispo)
        self._aug = None
        if augment and ALBU_OK:
            self._aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                            contrast_limit=0.2, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def stats(self) -> dict:
        """Retourne le compte real vs synthetic + total."""
        n_real = sum(1 for _, _, kind in self.samples if kind == "real")
        n_syn  = sum(1 for _, _, kind in self.samples if kind == "synthetic")
        return {"split": self.split, "total": len(self.samples),
                "real": n_real, "synthetic": n_syn}

    def __getitem__(self, idx: int):
        img_path, lbl_path, kind = self.samples[idx]

        # Image RGB
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Lecture impossible : {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Label uint8 single-channel (index direct 0-5)
        label_idx = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if label_idx is None:
            raise RuntimeError(f"Lecture impossible : {lbl_path}")
        if label_idx.ndim == 3:
            # securite : si le label est en RGB on prend le premier canal
            label_idx = label_idx[:, :, 0]

        # Resize (INTER_AREA pour image, INTER_NEAREST pour label)
        if self.target_size:
            tw, th = self.target_size
            image_rgb = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_AREA)
            label_idx = cv2.resize(label_idx, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Augmentations (image + masque ensemble)
        if self._aug is not None:
            out = self._aug(image=image_rgb, mask=label_idx)
            image_rgb = out["image"]
            label_idx = out["mask"]

        # Normalisation + tensor
        img = image_rgb.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        image_tensor = torch.from_numpy(img.copy())
        label_tensor = torch.from_numpy(label_idx.astype(np.int64).copy())
        return image_tensor, label_tensor


def index_to_color_semap(label_idx: np.ndarray) -> np.ndarray:
    """
    Convertit un masque d'index (0-5) en image couleur visualisable (BGR).
    Palette inspiree de SODUCO + variations.
    """
    palette = np.array([
        [255, 255, 255],  # 0 background -> blanc
        [128, 128, 128],  # 1 contours   -> gris
        [255,   0, 255],  # 2 built      -> magenta
        [128, 255, 128],  # 3 non_built  -> vert clair
        [255,   0,   0],  # 4 water      -> bleu (BGR)
        [  0,   0, 255],  # 5 road       -> rouge (BGR)
    ], dtype=np.uint8)
    H, W = label_idx.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        out[label_idx == i] = color
    return out
