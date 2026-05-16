"""
scripts/train.py — entrainement unifie U-Net pour SODUCO et SEMAP.

Remplace les anciens scripts/train_mapseg.py (SODUCO) et scripts/train_semap.py
(SEMAP). La logique de loss / loop / metriques est identique ; seul le dataset
loader change.

Usage :
    # Dataset SODUCO (data/historical_maps/, labels BGR couleur, 5 classes)
    python scripts/train.py --dataset soduco --epochs 20 --batch-size 8

    # Dataset SEMAP (data/config.json -> semap.external_root, 6 classes index)
    python scripts/train.py --dataset semap --epochs 20 --batch-size 8

    # Options SEMAP-specifiques
    python scripts/train.py --dataset semap --no-synthetic --epochs 30
    python scripts/train.py --dataset semap --target-size 384 --batch-size 4

    # Reprise depuis un checkpoint
    python scripts/train.py --dataset semap --resume external/weight/last.pth

Sorties :
    external/weight/{soduco|semap}_unet_best.pth
    external/weight/{soduco|semap}_unet_last.pth
    external/weight/{soduco|semap}_training.log
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from pipeline.semantic_segmentation import get_device


# ---------------------------------------------------------------------
# Loss CE + Dice (commun aux 2 datasets)
# ---------------------------------------------------------------------
class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, ce_w=1.0, dice_w=1.0,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_w = ce_w
        self.dice_w = dice_w
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                       weight=class_weights)

    def forward(self, logits, target):
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
        inter = (probs * target_oh).sum(dims)
        card = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * inter + 1e-6) / (card + 1e-6)
        return self.ce_w * ce + self.dice_w * (1.0 - dice.mean())


@torch.no_grad()
def compute_iou(preds, target, num_classes, ignore_index=255):
    valid = target != ignore_index
    ious = []
    for c in range(num_classes):
        pc = (preds == c) & valid
        tc = (target == c) & valid
        inter = (pc & tc).sum().item()
        union = (pc | tc).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return np.array(ious, dtype=np.float64)


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
        total += loss.item() * img.size(0); n += img.size(0)
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
        total += loss.item() * img.size(0); n += img.size(0)
        preds = torch.argmax(logits, dim=1)
        ious = compute_iou(preds, lbl, num_classes, ignore_index)
        for c, v in enumerate(ious):
            if not np.isnan(v):
                iou_sum[c] += v; iou_cnt[c] += 1
    per_cls = np.where(iou_cnt > 0, iou_sum / np.maximum(iou_cnt, 1), np.nan)
    return total / max(n, 1), float(np.nanmean(per_cls)), per_cls


# ---------------------------------------------------------------------
# Loaders par dataset
# ---------------------------------------------------------------------
def build_soduco(target_size, augment):
    from pipeline.dataset import MapSegDataset
    classes_json = BASE_DIR / "data" / "historical_maps" / "classes.json"
    train_ds = MapSegDataset(
        BASE_DIR / "data" / "historical_maps" / "train" / "images",
        BASE_DIR / "data" / "historical_maps" / "train" / "labels",
        classes_json, target_size=target_size, augment=augment)
    val_ds   = MapSegDataset(
        BASE_DIR / "data" / "historical_maps" / "eval" / "images",
        BASE_DIR / "data" / "historical_maps" / "eval" / "labels",
        classes_json, target_size=target_size, augment=False)
    return train_ds, val_ds


def build_semap(target_size, augment, include_synthetic):
    from pipeline.semap_dataset import SemapDataset
    train_ds = SemapDataset("train", target_size=target_size,
                              augment=augment,
                              include_synthetic=include_synthetic)
    val_ds   = SemapDataset("val", target_size=target_size,
                              augment=False,
                              include_synthetic=include_synthetic)
    return train_ds, val_ds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Entrainement U-Net unifie")
    p.add_argument("--dataset", choices=["soduco", "semap"], required=True,
                    help="soduco = data/historical_maps/  |  semap = data/config.json:semap.external_root")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-synthetic", action="store_true",
                    help="SEMAP only : exclut les 12 122 images synthetiques.")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    weight_dir = BASE_DIR / "external" / "weight"
    weight_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(verbose=True) if args.device == "auto" else args.device
    print(f"Device : {device}\n")

    target = (args.target_size, args.target_size)
    print(f"Dataset : {args.dataset.upper()}")
    if args.dataset == "soduco":
        train_ds, val_ds = build_soduco(target, not args.no_augment)
    else:  # semap
        train_ds, val_ds = build_semap(target, not args.no_augment,
                                          include_synthetic=not args.no_synthetic)
    print(f"  train : {len(train_ds)} echantillons")
    print(f"  val   : {len(val_ds)} echantillons")
    print(f"  classes : {train_ds.num_classes}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                      in_channels=3, classes=train_ds.num_classes).to(device)
    print(f"Modele : U-Net {args.encoder} (classes={train_ds.num_classes})")

    if args.resume:
        print(f"Reprise depuis {args.resume}")
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    criterion = CEDiceLoss(num_classes=train_ds.num_classes, ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Sauvegarde sous {dataset}_unet_*.pth pour ne pas ecraser un autre run
    prefix = args.dataset
    log_path = weight_dir / f"{prefix}_training.log"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["epoch", "lr", "train_loss", "val_loss", "mIoU"]
                 + [f"IoU_{c['name']}" for c in train_ds.classes])

    best_miou = -1.0
    print()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"=== Epoch {ep}/{args.epochs} "
              f"(lr={optimizer.param_groups[0]['lr']:.2e}) ===")
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, miou, iou_cls = evaluate(model, val_loader, criterion, device,
                                       train_ds.num_classes)
        scheduler.step()
        print(f"  train_loss={tl:.4f}  val_loss={vl:.4f}  mIoU={miou:.4f}  "
              f"({time.time()-t0:.1f}s)")
        for c, v in zip(train_ds.classes, iou_cls):
            print(f"      IoU {c['name']:14s} = {v:.4f}")
        log.writerow([ep, optimizer.param_groups[0]['lr'],
                      tl, vl, miou] + iou_cls.tolist())
        log_file.flush()

        torch.save(model.state_dict(), weight_dir / f"{prefix}_unet_last.pth")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(),
                        weight_dir / f"{prefix}_unet_best.pth")
            print(f"  Nouveau best : mIoU={miou:.4f} -> "
                  f"external/weight/{prefix}_unet_best.pth")

    log_file.close()
    print(f"\nTermine. Best mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()
