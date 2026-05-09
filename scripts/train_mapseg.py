"""
Entrainement du U-Net sur le dataset historical_maps (SODUCO/Benchmark).

Lance simplement :
    python scripts/train_mapseg.py

Ou avec options :
    python scripts/train_mapseg.py --epochs 30 --batch-size 8 --lr 1e-4

Sortie :
    external/weight/best.pth      (meilleur modele selon mIoU eval)
    external/weight/last.pth      (dernier checkpoint)
    external/weight/training.log  (logs CSV)

Le pipeline detecte automatiquement la GPU si dispo (RTX 2050 / Colab T4),
sinon tombe sur CPU (beaucoup plus lent).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from pipeline.dataset import MapSegDataset
from pipeline.semantic_segmentation import get_device


# ---------------------------------------------------------------------
# Loss : combinaison CrossEntropy + Dice (robuste sur classes desequilibrees)
# ---------------------------------------------------------------------
class CEDiceLoss(nn.Module):
    """Cross-entropy + Dice. Ignore_index pour les pixels non labellises."""
    def __init__(self, num_classes: int, ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        # Dice : on exclut les pixels ignores
        valid_mask = (target != self.ignore_index)
        target_clean = target.clone()
        target_clean[~valid_mask] = 0
        target_oh = F.one_hot(target_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        # On masque les pixels invalides
        valid_mask_f = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_f
        target_oh = target_oh * valid_mask_f
        dims = (0, 2, 3)
        intersect = (probs * target_oh).sum(dims)
        cardinality = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersect + 1e-6) / (cardinality + 1e-6)
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


# ---------------------------------------------------------------------
# Métriques : IoU par classe
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_iou(preds: torch.Tensor, target: torch.Tensor,
                num_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Retourne un array de IoU par classe (NaN si classe absente)."""
    ious = []
    valid = target != ignore_index
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        target_c = (target == c) & valid
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return np.array(ious, dtype=np.float64)


# ---------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, *, log_every=10):
    model.train()
    total_loss = 0.0
    n_seen = 0
    for step, (img, lbl) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        n_seen += img.size(0)
        if (step + 1) % log_every == 0:
            print(f"    step {step+1}/{len(loader)}  loss={loss.item():.4f}")
    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index=255):
    model.eval()
    total_loss = 0.0
    n_seen = 0
    cls_iou_sum = np.zeros(num_classes, dtype=np.float64)
    cls_iou_count = np.zeros(num_classes, dtype=np.int64)
    for img, lbl in loader:
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, lbl)
        total_loss += loss.item() * img.size(0)
        n_seen += img.size(0)
        preds = torch.argmax(logits, dim=1)
        ious = compute_iou(preds, lbl, num_classes, ignore_index)
        for c, v in enumerate(ious):
            if not np.isnan(v):
                cls_iou_sum[c] += v
                cls_iou_count[c] += 1
    mean_iou_per_class = np.where(cls_iou_count > 0,
                                    cls_iou_sum / np.maximum(cls_iou_count, 1),
                                    np.nan)
    miou = np.nanmean(mean_iou_per_class)
    return total_loss / max(n_seen, 1), miou, mean_iou_per_class


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrainement U-Net cartes")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--target-size", type=int,  default=512,
                        help="Taille des patchs (carre).")
    parser.add_argument("--encoder",    type=str,   default="resnet34")
    parser.add_argument("--device",     choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int,  default=2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Chemin vers un checkpoint .pth a reprendre.")
    args = parser.parse_args()

    # Paths
    images_train  = BASE_DIR / "data" / "historical_maps" / "train" / "images"
    labels_train  = BASE_DIR / "data" / "historical_maps" / "train" / "labels"
    images_eval   = BASE_DIR / "data" / "historical_maps" / "eval"  / "images"
    labels_eval   = BASE_DIR / "data" / "historical_maps" / "eval"  / "labels"
    classes_json  = BASE_DIR / "data" / "historical_maps" / "classes.json"
    weight_dir    = BASE_DIR / "external" / "weight"
    weight_dir.mkdir(parents=True, exist_ok=True)

    for p in (images_train, labels_train, images_eval, labels_eval, classes_json):
        if not p.exists():
            print(f"ERREUR : {p} introuvable.")
            sys.exit(1)

    # Device
    if args.device == "auto":
        device = get_device(verbose=True)
    else:
        device = args.device
    print(f"Device : {device}")

    # Datasets / loaders
    print("Chargement dataset...")
    target = (args.target_size, args.target_size)
    train_ds = MapSegDataset(images_train, labels_train, classes_json,
                              target_size=target, augment=not args.no_augment)
    eval_ds  = MapSegDataset(images_eval,  labels_eval,  classes_json,
                              target_size=target, augment=False)
    print(f"  train : {len(train_ds)} images")
    print(f"  eval  : {len(eval_ds)} images")
    print(f"  classes : {train_ds.num_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))

    # Modele
    print(f"Modele : U-Net {args.encoder} (classes={train_ds.num_classes})")
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=train_ds.num_classes,
    ).to(device)

    if args.resume:
        print(f"Reprise depuis {args.resume}")
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    # Loss + optim + scheduler
    criterion = CEDiceLoss(num_classes=train_ds.num_classes, ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Log CSV
    log_path = weight_dir / "training.log"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["epoch", "lr", "train_loss", "eval_loss", "mIoU"]
                 + [f"IoU_{c['name']}" for c in train_ds.classes])

    best_miou = -1.0
    print()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"=== Epoch {epoch}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e}) ===")

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                       criterion, device)
        eval_loss, miou, iou_per_cls = evaluate(model, eval_loader, criterion,
                                                  device, train_ds.num_classes)

        scheduler.step()
        dt = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  mIoU={miou:.4f}  ({dt:.1f}s)")
        for c, v in zip(train_ds.classes, iou_per_cls):
            print(f"      IoU {c['name']:18s} = {v:.4f}")

        log.writerow([epoch, optimizer.param_groups[0]['lr'],
                      train_loss, eval_loss, miou] + iou_per_cls.tolist())
        log_file.flush()

        # Sauvegarde
        torch.save(model.state_dict(), weight_dir / "last.pth")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), weight_dir / "best.pth")
            print(f"  Nouveau best : mIoU={miou:.4f} -> external/weight/best.pth")

    log_file.close()
    print(f"\nTermine. Best mIoU = {best_miou:.4f}")
    print(f"Poids : {weight_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
