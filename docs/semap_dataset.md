# Dataset SEMAP (Petitpierre 2025)

## Vue d'ensemble

**SEMAP** (Semantic Segmentation Map Dataset, EPFL 2025) — le plus gros dataset annoté pour la segmentation sémantique de cartes historiques.

- **1 439 images réelles** annotées manuellement
- **12 122 images synthétiques** générées
- **6 classes** : `background`, `contours`, `built`, `non_built`, `water`, `road_network`
- **Tailles** : 768×768 à 1000×1000 px
- **Splits officiels** : 10 703 train / 2 712 val / 143 test
- **Modèle pré-entraîné** : Mask2Former Swin-L (908 MB) + config + log

**Citation** :
```
Petitpierre R, Gomez Donoso D, Kriesel B (2025). Semantic Segmentation Map Dataset (Semap). EPFL.
DOI: 10.5281/zenodo.16164781
```

## Localisation

Le dataset est **externe au repo Git** (taille 2 GB) :

```
C:\Users\ochou\Documents\Claude\pfa\19048095\
├── images/images/{real,synthetic}/   *.jpg
├── labels/labels/{real,synthetic}/   *.png  (uint8 single-channel, valeurs 0-5)
├── partitions/partitions/            train.txt val.txt test.txt
└── model/model/                      best_mIoU_iter_138828.pth + config + log
```

Le projet pointe dessus via `data/semap_config.json` → champ `external_root`.

## Utilisation

### 1. Vérifier la connexion au dataset

```python
from pipeline.semap_dataset import SemapDataset

ds = SemapDataset(split="train", target_size=(512, 512), augment=True)
print(ds.stats())
# {'split': 'train', 'total': 10703, 'real': 1153, 'synthetic': 9550}
```

### 2. Entraîner un U-Net léger

```bash
# Toutes les images (real + synthetic) — recommandé
python scripts/train_semap.py --epochs 20 --batch-size 8

# Uniquement les images réelles (plus rapide, moins généralisé)
python scripts/train_semap.py --no-synthetic --epochs 30

# Petite taille pour GPU 4 GB (RTX 2050)
python scripts/train_semap.py --target-size 384 --batch-size 4
```

Sortie : `external/weight/semap_unet_best.pth`.

### 3. Inférence dans le pipeline

```bash
python -m pipeline.pipeline data/raw/carte.png \
       -o data/processed \
       --semantic --weights external/weight/semap_unet_best.pth
```

Le pipeline détecte automatiquement que les poids contiennent "semap" et charge `data/semap_config.json` pour utiliser les 6 classes SEMAP.

## Modèle Mask2Former pré-entraîné (option avancée)

Le dataset fournit `model/model/best_mIoU_iter_138828.pth` — un Mask2Former Swin-L entraîné par l'équipe Petitpierre. Il a une qualité **supérieure** au U-Net mais nécessite l'écosystème mmsegmentation.

### Installation

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmsegmentation>=1.2.2"
```

### Inférence

```python
from mmseg.apis import init_model, inference_model
import cv2

CFG = "C:/Users/ochou/Documents/Claude/pfa/19048095/model/model/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"
PTH = "C:/Users/ochou/Documents/Claude/pfa/19048095/model/model/best_mIoU_iter_138828.pth"

model = init_model(CFG, PTH, device="cuda:0")
result = inference_model(model, "data/raw/carte.png")
# result.pred_sem_seg.data : tenseur d'index (0-5)
```

### Avantages / Inconvénients vs U-Net

| Critère | U-Net resnet34 | Mask2Former Swin-L |
|---|---|---|
| **mIoU sur SEMAP val** | ~0.55-0.65 (estimé) | ~0.78 (rapporté) |
| **Taille modèle** | ~24 MB | 908 MB |
| **VRAM inference** | ~1 GB | ~6 GB |
| **VRAM training** | ~3 GB (batch=8) | ~14 GB (batch=2) |
| **Temps inference** | ~50 ms/img | ~400 ms/img |
| **Compatibilité RTX 2050** | OK | NON (4 GB VRAM insuffisant) |
| **Dépendances** | torch + smp | mmsegmentation + mmcv |

**Recommandation** :
- **PFA + RTX 2050 + Anaconda** → U-Net (`scripts/train_semap.py`)
- **Colab T4 (15 GB VRAM)** → Mask2Former pré-entraîné directement
- **Production** → distillation Mask2Former → U-Net

## Distillation (knowledge distillation)

Pour obtenir les performances de Mask2Former dans un U-Net léger :

1. Sur Colab T4, utilise Mask2Former pour produire des masques pseudo-GT sur tes propres cartes Tunis/Algeria.
2. Combine ces masques avec ton dataset SEMAP via `data/historical_maps/`.
3. Réentraîne `train_semap.py` avec ces données augmentées.

Le student U-Net atteint typiquement 90-95% du mIoU du teacher Mask2Former, pour 10× moins de paramètres.

## Licence

Dataset SEMAP : **CC BY 4.0**.
Voir `19048095/license_images.md` pour les politiques de réutilisation des images sources (BnF, Library of Congress, David Rumsey, etc.).
