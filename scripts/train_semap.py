"""
Entrainement U-Net resnet34 sur le dataset SEMAP (Petitpierre 2025).

Usage minimal :
    python scripts/train_semap.py

Avec options :
    python scripts/train_semap.py --epochs 30 --batch-size 8 --lr 1e-4
    python scripts/train_semap.py --no-synthetic   # uniquement images reelles (1 439)
    python scripts/train_semap.py --target-size 384

Sortie :
    external/weight/semap_unet_best.pth   (meilleur mIoU sur val)
    external/weight/semap_unet_last.pth   (dernier checkpoint)
    external/weight/semap_training.log    (CSV)

Le pipeline detecte la GPU automatiquement (RTX 2050 / Colab T4),
sinon tombe sur CPU (entrainement lent).

Note : pour le modele Mask2Former Swin-L pre-entraine fourni avec
SEMAP (best_mIoU_iter_138828.pth, 908 MB), voir docs/mask2former.md.
Ce script forme un U-Net leger qui marche sans mmsegmentation.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from pipeline.semap_dataset import SemapDataset
from pipeline.semantic_segmentation import get_device


# ---------------------------------------------------------------------
# Loss CE+Dice (robuste aux classes desequilibrees)
# ---------------------------------------------------------------------
class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                       weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        valid = (target != self.ignore_index)
        target_clean = target.clone()
        target_clean[~valid] = 0
        target_oh = F.one_hot(target_clean, num_classes=self.num_classes
                               ).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        target_oh = target_oh * valid_f
        dims = (0, 2, 3)
        intersect = (probs * target_oh).sum(dims)
        cardinality = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersect + 1e-6) / (cardinality + 1e-6)
        return self.ce_weight * ce + self.dice_weight * (1.0 - dice.mean())


# ---------------------------------------------------------------------
# Metriques : IoU par classe
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_iou(preds, target, num_classes, ignore_index=255):
    valid = target != ignore_index
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        targ_c = (target == c) & valid
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return np.array(ious, dtype=np.float64)


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, optim, criterion, device, log_every=20):
    model.train()
    total = 0.0; n = 0
    for step, (img, lbl) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        optim.zero_grad()
        logits = model(img)
        loss = criterion(logits, lbl)
        loss.backward()
        optim.step()
        total += loss.item() * img.size(0)
        n += img.size(0)
        if (step + 1) % log_every == 0:
            print(f"    step {step+1}/{len(loader)}  loss={loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index=255):
    model.eval()
    total = 0.0; n = 0
    iou_sum = np.zeros(num_classes); iou_cnt = np.zeros(num_classes, dtype=int)
    for img, lbl in loader:
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, lbl)
        total += loss.item() * img.size(0)
        n += img.size(0)
        preds = torch.argmax(logits, dim=1)
        ious = compute_iou(preds, lbl, num_classes, ignore_index)
        for c, v in enumerate(ious):
            if not np.isnan(v):
                iou_sum[c] += v; iou_cnt[c] += 1
    per_cls = np.where(iou_cnt > 0, iou_sum / np.maximum(iou_cnt, 1), np.nan)
    miou = float(np.nanmean(per_cls))
    return total / max(n, 1), miou, per_cls


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Entrainement U-Net sur SEMAP")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--target-size", type=int,  default=512)
    p.add_argument("--encoder",    type=str,   default="resnet34")
    p.add_argument("--device",     choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--num-workers", type=int,  default=2)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-synthetic", action="store_true",
                    help="Exclut les 12 122 images synthetiques (entraine sur les 1 439 reelles).")
    p.add_argument("--resume",     type=str,   default=None)
    args = p.parse_args()

    weight_dir = BASE_DIR / "external" / "weight"
    weight_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(verbose=True) if args.device == "auto" else args.device
    print(f"Device : {device}\n")

    target = (args.target_size, args.target_size)
    include_syn = not args.no_synthetic

    print("Chargement dataset SEMAP...")
    train_ds = SemapDataset("train", target_size=target,
                              augment=not args.no_augment,
                              include_synthetic=include_syn)
    val_ds   = SemapDataset("val",   target_size=target,
                              augment=False, include_synthetic=include_syn)
    print(f"  train : {train_ds.stats()}")
    print(f"  val   : {val_ds.stats()}")
    print(f"  classes : {train_ds.num_classes}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))

    print(f"Modele : U-Net {args.encoder} (classes={train_ds.num_classes})")
    model = smp.Unet(encoder_name=args.encoder,
                      encoder_weights="imagenet",
                      in_channels=3,
                      classes=train_ds.num_classes).to(device)

    if args.resume:
        print(f"Reprise depuis {args.resume}")
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    criterion = CEDiceLoss(num_classes=train_ds.num_classes, ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    log_path = weight_dir / "semap_training.log"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["epoch", "lr", "train_loss", "val_loss", "mIoU"]
                 + [f"IoU_{c['name']}" for c in train_ds.classes])

    best_miou = -1.0
    print()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"=== Epoch {ep}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e}) ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, miou, iou_cls = evaluate(model, val_loader, criterion,
                                             device, train_ds.num_classes)
        scheduler.step()
        dt = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={miou:.4f}  ({dt:.1f}s)")
        for c, v in zip(train_ds.classes, iou_cls):
            print(f"      IoU {c['name']:14s} = {v:.4f}")
        log.writerow([ep, optimizer.param_groups[0]['lr'],
                      train_loss, val_loss, miou] + iou_cls.tolist())
        log_file.flush()

        torch.save(model.state_dict(), weight_dir / "semap_unet_last.pth")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), weight_dir / "semap_unet_best.pth")
            print(f"  Nouveau best : mIoU={miou:.4f} -> external/weight/semap_unet_best.pth")

    log_file.close()
    print(f"\nTermine. Best mIoU = {best_miou:.4f}")
    print(f"Poids : {weight_dir / 'semap_unet_best.pth'}")


if __name__ == "__main__":
    main()
