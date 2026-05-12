# Guide d'utilisation des datasets et modèles externes

Ce guide explique pas à pas comment **installer, vérifier, entraîner et utiliser** les trois ressources externes du projet CartoVec :

1. **SODUCO Benchmark** — dataset Paris BnF (256+49 images, 5 classes BGR)
2. **SEMAP (Petitpierre 2025)** — dataset EPFL (13 561 images, 6 classes)
3. **Mask2Former Swin-L pré-entraîné** — modèle de référence (908 MB)

> Pré-requis : env conda `pfa` actif (`conda activate pfa`), GPU détecté par `python scripts/test_gpu.py` (optionnel mais fortement recommandé pour l'entraînement).

---

## Tableau récapitulatif

| Critère | SODUCO | SEMAP | Mask2Former pré-entraîné |
|---|---|---|---|
| **Volume** | 305 images | 13 561 images | 908 MB poids |
| **Classes** | 5 (BGR couleur) | 6 (index 0-5) | 6 (SEMAP-compatible) |
| **Résolution** | 1000 × 1000 | 768-1000 px | 384 × 384 patches |
| **Framework** | PyTorch + smp | PyTorch + smp | mmsegmentation |
| **VRAM training** | ~3 GB (batch=8) | ~3 GB (batch=8) | ~14 GB (batch=2) |
| **VRAM inference** | ~1 GB | ~1 GB | ~6 GB |
| **mIoU attendu** | ~0.55 | ~0.65 | ~0.78 |
| **Compatible RTX 2050** | OK | OK | NON (Colab T4 requis) |
| **Licence** | MIT | CC BY 4.0 | CC BY 4.0 |
| **Effort install** | Trivial (déjà là) | 5 min | 15-20 min (mmcv) |

**Recommandation par cas d'usage** :

- **Premier prototype rapide** → SODUCO (`train_mapseg.py`)
- **Modèle production léger** → SEMAP avec U-Net (`train_semap.py`)
- **Pseudo-labellisation tes cartes locales** → Mask2Former pré-entraîné sur Colab
- **Production haute précision** → distillation Mask2Former → U-Net (étape avancée)

---

## 10.1 SODUCO Benchmark

### Présentation

Dataset de référence pour la vectorisation de cartes historiques, issu de **SODUCO** (Sorbonne Université) et **BnF Gallica**. Centré sur le Paris historique (BnF / David Rumsey / Berkeley Library).

**Repo source** : [https://github.com/soduco/Benchmark_historical_map_vectorization](https://github.com/soduco/Benchmark_historical_map_vectorization)
**Licence** : MIT

**Classes** (encodage couleur BGR, voir `data/historical_maps/classes.json`) :

| ID | Nom | BGR | Description |
|---|---|---|---|
| 0 | `background` | (255, 255, 255) | Fond papier |
| 1 | `inside_blocks` | (255, 255, 0) | Intérieur îlots / bâtiments |
| 2 | `building_walls` | (255, 0, 255) | Contours de bâtiments (magenta) |
| 3 | `street_network` | (255, 0, 0) | Réseau de rues (bleu BGR) |
| 4 | `unknown` | (0, 0, 0) | Zones non labellisées |

### Installation

Le dataset est **déjà dans le repo** sous `data/historical_maps/` :

```
data/historical_maps/
├── classes.json                 # palette des 5 classes
├── train/
│   ├── images/      256 fichiers PNG (1000×1000)
│   └── labels/      256 fichiers PNG (RGB couleur)
└── eval/
    ├── images/      49 fichiers
    └── labels/      49 fichiers
```

### Vérification rapide

```bash
python -c "
from pipeline.dataset import MapSegDataset
ds = MapSegDataset(
    images_dir='data/historical_maps/train/images',
    labels_dir='data/historical_maps/train/labels',
    classes_json='data/historical_maps/classes.json',
    target_size=(512, 512))
print(f'Train : {len(ds)} échantillons')
img, lbl = ds[0]
print(f'Image : {img.shape}, Label : {lbl.shape}, classes uniques : {lbl.unique().tolist()}')
"
```

Sortie attendue :
```
Train : 256 échantillons
Image : torch.Size([3, 512, 512]), Label : torch.Size([512, 512]), classes uniques : [0, 1, 2, 3]
```

### Entraînement (U-Net resnet34)

```bash
# Tout le dataset, 20 epochs, batch=8 (RTX 2050 OK)
python scripts/train_mapseg.py --epochs 20 --batch-size 8

# Plus petit batch si OOM
python scripts/train_mapseg.py --batch-size 4 --target-size 384

# Reprise depuis un checkpoint
python scripts/train_mapseg.py --resume external/weight/last.pth --epochs 30
```

**Sorties** :
- `external/weight/best.pth` — meilleur modèle sur val mIoU
- `external/weight/last.pth` — dernier checkpoint
- `external/weight/training.log` — métriques CSV (epoch, lr, loss, mIoU par classe)

**Temps estimé** :
- RTX 2050 4 GB : ~5-8 min par epoch (batch=4, target=512)
- Colab T4 15 GB : ~1-2 min par epoch (batch=16, target=512)
- CPU : ~30 min par epoch (déconseillé)

### Inférence dans le pipeline

```bash
python -m pipeline.pipeline data/raw/ma_carte.png \
       -o data/processed \
       --semantic \
       --weights external/weight/best.pth
```

Le pipeline détecte automatiquement `data/historical_maps/classes.json` et charge la palette BGR pour la conversion masque → couleurs.

### Trucs et astuces

- **Petit dataset (305 images)** : risque de sur-apprentissage. Active toujours l'augmentation (`augment=True` par défaut). Considère ajouter SEMAP pour la généralisation.
- **Format des labels** : RGB, pas index ! La classe `MapSegDataset` convertit automatiquement via `color_label_to_index()`.
- **Distinction `inside_blocks` vs `building_walls`** : utile en aval pour vectoriser séparément l'intérieur (polygones pleins) et le contour (polylignes).

---

## 10.2 SEMAP (Petitpierre 2025)

### Présentation

Le **plus gros dataset annoté** pour la segmentation de cartes historiques à ce jour. Couvre BnF, Library of Congress, David Rumsey, Leiden, NYPL, Berkeley, Princeton, etc.

**Source** : [Zenodo DOI 10.5281/zenodo.16164781](https://doi.org/10.5281/zenodo.16164781)
**Auteurs** : Rémi Petitpierre, Damien Gomez Donoso, Ben Kriesel (EPFL, 2025)
**Licence** : CC BY 4.0
**Citation** :
```
@misc{semap_petitpierre_2025,
  author = {Petitpierre, Rémi and Gomez Donoso, Damien and Kriesel, Ben},
  title  = {Semantic Segmentation Map Dataset (Semap)},
  year   = {2025},
  publisher = {EPFL},
  doi    = {10.5281/zenodo.16164781}
}
```

**6 classes** (labels uint8 single-channel, valeurs 0-5) :

| ID | Nom | Description |
|---|---|---|
| 0 | `background` | Papier / fond |
| 1 | `contours` | Lignes de contour, neatlines, séparateurs |
| 2 | `built` | Zones construites (bâtiments) |
| 3 | `non_built` | Espaces non construits (cours, parcs) |
| 4 | `water` | Hydrographie |
| 5 | `road_network` | Réseau de rues / routes |

### Téléchargement

Le dataset fait **2 GB** — il reste **hors du repo Git**.

1. Va sur [https://doi.org/10.5281/zenodo.16164781](https://doi.org/10.5281/zenodo.16164781)
2. Télécharge `images.zip`, `labels.zip`, `partitions.zip`, `model.zip` (~3 GB au total avec le modèle)
3. Décompresse dans un dossier de ton choix, par exemple `C:\Users\ochou\Documents\Claude\pfa\19048095\`

Structure attendue après décompression (note : les zip créent un double dossier `images/images/` etc.) :

```
19048095/
├── images/images/{real,synthetic}/      *.jpg
├── labels/labels/{real,synthetic}/      *.png  (uint8 single-channel)
├── partitions/partitions/               train.txt val.txt test.txt
└── model/model/                         best_mIoU_iter_138828.pth + config + log
```

### Configuration

Édite `data/semap_config.json` pour pointer vers ton dossier :

```json
{
  "external_root": "C:/Users/ochou/Documents/Claude/pfa/19048095",
  ...
}
```

**Important** : sous Windows, utilise des slashes `/` ou doubles backslashes `\\` (pas un simple `\`).

### Vérification

```python
from pipeline.semap_dataset import SemapDataset

train_ds = SemapDataset(split="train", target_size=(512, 512), augment=True)
val_ds   = SemapDataset(split="val",   target_size=(512, 512))
test_ds  = SemapDataset(split="test",  target_size=(512, 512))

print(train_ds.stats())  # {'split': 'train', 'total': 10703, 'real': 1153, 'synthetic': 9550}
print(val_ds.stats())    # {'split': 'val',   'total': 2712,  ...}
print(test_ds.stats())   # {'split': 'test',  'total': 143,   ...}

img, lbl = train_ds[0]
print(img.shape, lbl.shape, lbl.unique().tolist())
```

Si tu obtiens `FileNotFoundError`, vérifie le champ `external_root` du config.

### Entraînement (U-Net resnet34, recommandé)

```bash
# Toutes les images (real + synthetic) — meilleure généralisation
python scripts/train_semap.py --epochs 20 --batch-size 8

# Uniquement les 1 439 images réelles (3-4× plus rapide, moins généralisé)
python scripts/train_semap.py --no-synthetic --epochs 30

# RTX 2050 4 GB : si OOM
python scripts/train_semap.py --target-size 384 --batch-size 4

# Reprise depuis un checkpoint
python scripts/train_semap.py --resume external/weight/semap_unet_last.pth
```

**Sorties** :
- `external/weight/semap_unet_best.pth`
- `external/weight/semap_unet_last.pth`
- `external/weight/semap_training.log`

**Temps estimé** (10 703 train + 2 712 val) :
- RTX 2050 batch=4 target=384 : ~25-35 min par epoch
- Colab T4 batch=8 target=512 : ~8-10 min par epoch

### Inférence

```bash
python -m pipeline.pipeline data/raw/ma_carte.png \
       -o data/processed \
       --semantic \
       --weights external/weight/semap_unet_best.pth \
       --device auto
```

Le pipeline **détecte automatiquement** que les poids contiennent `semap` dans leur nom → charge `data/semap_config.json` et utilise les 6 classes SEMAP au lieu des 3 par défaut.

### Trucs et astuces

- **Synthetic vs real** : les 12 122 images synthétiques améliorent la généralisation mais introduisent un *domain gap*. Bonne pratique : pré-entraîne sur tout (`--epochs 15`), puis fine-tune sur real uniquement (`--no-synthetic --epochs 10 --resume best.pth`).
- **`ignore_index=255`** : la fonction de perte ignore les pixels marqués 255 (cas rare où un label sort de la palette). Ne change pas cette valeur sauf si tu sais ce que tu fais.
- **`--target-size`** : pour la RTX 2050, garde 384-512. Au-delà → OOM. Sur Colab T4, monte à 640-768.
- **Validation mIoU** : si tu obtiens < 0.30, vérifie que les labels sont bien des index uint8 (pas RGB), et que `external_root` pointe vers le bon endroit.

---

## 10.3 Mask2Former Swin-L pré-entraîné

### Présentation

Modèle de **référence** entraîné par l'équipe Petitpierre sur SEMAP. Architecture **Mask2Former** avec backbone **Swin-L** (ImageNet-22k pretrain + fine-tuning sur ADE20k 640×640 puis SEMAP).

**Fichier** : `19048095/model/model/best_mIoU_iter_138828.pth` (**908 MB**)
**Config** : `mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py`
**Framework** : [mmsegmentation v1.2.2](https://github.com/open-mmlab/mmsegmentation) (OpenMMLab)
**mIoU rapporté** : ~0.78 sur SEMAP val

### Pourquoi l'utiliser

| Cas d'usage | Recommandation |
|---|---|
| Tu as une GPU avec ≥ 8 GB VRAM (Colab T4, A100, RTX 3080+) | **Utilise-le directement** |
| Tu veux des pseudo-labels pour tes cartes locales (Tunis, Bizerte…) | **Distillation** (voir plus bas) |
| Tu es sur RTX 2050 4 GB | **Inutilisable directement**. Passe par distillation depuis Colab |

### Installation mmsegmentation

```bash
conda activate pfa
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmsegmentation>=1.2.2"

# Vérification
python -c "from mmseg.apis import init_model; print('OK')"
```

**Note importante** : `mmcv` est **lourd à compiler** (~10-15 min sur Windows sans wheel précompilée). Sur Colab c'est instantané.

### Inférence directe

```python
from mmseg.apis import init_model, inference_model
import cv2
import numpy as np

# Chemins
SEMAP_ROOT = "C:/Users/ochou/Documents/Claude/pfa/19048095"
CFG = f"{SEMAP_ROOT}/model/model/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"
PTH = f"{SEMAP_ROOT}/model/model/best_mIoU_iter_138828.pth"

# Charger
model = init_model(CFG, PTH, device="cuda:0")

# Inférer sur une carte
img_path = "data/raw/ma_carte.png"
result = inference_model(model, img_path)

# result.pred_sem_seg.data : tenseur (1, H, W) avec index 0-5
pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
print(f"Forme : {pred.shape}, classes : {np.unique(pred).tolist()}")

# Sauvegarde du masque
cv2.imwrite("data/processed/mask_mask2former.png", pred)
```

### Visualiser la prédiction

```python
from pipeline.semap_dataset import index_to_color_semap
import matplotlib.pyplot as plt

mask_colored = index_to_color_semap(pred)
mask_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
plt.imshow(mask_rgb); plt.axis('off'); plt.show()
```

### Stratégie de distillation (étape avancée)

**Principe** : Mask2Former est trop gros pour ta RTX 2050, mais on peut s'en servir comme **professeur** pour entraîner un U-Net **étudiant** plus léger.

Le student U-Net atteint typiquement **90-95% du mIoU du teacher Mask2Former**, pour ~10× moins de paramètres.

**Étapes** :

1. **Collecte tes cartes locales** non-labellisées :
   ```
   data/raw_tunisia_algeria/
   ├── bizerte.png
   ├── tunis_1969.jpg
   ├── ain_bessem.tif
   └── ...
   ```

2. **Lance Mask2Former sur Colab T4** (15 GB VRAM dispo gratuit) pour produire des pseudo-labels :

   ```python
   # notebook Colab
   from mmseg.apis import init_model, inference_model
   import cv2, os
   from pathlib import Path

   model = init_model(CFG, PTH, device="cuda:0")
   INPUT_DIR  = Path("/content/drive/MyDrive/pfa/data/raw_tunisia_algeria")
   OUTPUT_DIR = Path("/content/drive/MyDrive/pfa/data/pseudo_labels")
   OUTPUT_DIR.mkdir(exist_ok=True)

   for img_path in INPUT_DIR.glob("*"):
       result = inference_model(model, str(img_path))
       pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype("uint8")
       cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}.png"), pred)
   ```

3. **Ajoute ces pseudo-labels au dataset SEMAP local** (en local sur ton PC) :

   ```bash
   # Copie tes cartes locales + pseudo-labels dans la structure SEMAP
   cp data/raw_tunisia_algeria/*.png 19048095/images/images/real/
   cp data/pseudo_labels/*.png       19048095/labels/labels/real/
   # Ajoute les noms (sans extension) à partitions/train.txt
   ```

4. **Réentraîne le U-Net** sur ce dataset augmenté :

   ```bash
   python scripts/train_semap.py --epochs 15 --batch-size 8 \
          --resume external/weight/semap_unet_best.pth
   ```

5. **Évalue sur tes cartes locales** que tu connais bien (qualitatif) ou sur une vérité terrain (quantitatif via `IoU` dans le log).

**Avantages de la distillation** :
- Le U-Net étudiant tourne sur ta RTX 2050 en ~50 ms/image
- Apprend du Mask2Former + s'adapte au style cartographique tunisien/algérien
- Réversible : si les pseudo-labels sont mauvais, retire-les et relance

**Trucs et astuces** :
- Filtre les pseudo-labels où Mask2Former est peu confiant (utilise `result.seg_logits` et ne garde que les pixels où `softmax.max(dim=0) > 0.8`).
- Mélange pseudo-labels et vrais labels en 1:1 pour éviter l'effondrement.
- Garde toujours `data/historical_maps/eval/` ou `SEMAP val` comme set de référence indépendant.

---

## FAQ / Dépannage

### Q : Quel dataset choisir si je dois en choisir un seul ?

**SEMAP**, sans hésiter. 50× plus de données, 6 classes (au lieu de 5), et tu profites du modèle Mask2Former pré-entraîné en bonus.

### Q : Je suis sur RTX 2050 4 GB et `train_semap.py` me sort un `CUDA out of memory`.

Réduis dans cet ordre :
1. `--batch-size 4`
2. `--target-size 384`
3. `--num-workers 0`
4. En dernier recours : `--batch-size 2 --target-size 256`

Si rien ne marche → entraîne sur Colab T4, puis utilise les poids en inférence sur ta RTX.

### Q : Comment savoir si mon entraînement marche bien ?

Surveille `external/weight/*_training.log` ou affiche les valeurs :

```python
import pandas as pd
log = pd.read_csv("external/weight/semap_training.log")
print(log[["epoch", "train_loss", "val_loss", "mIoU"]].tail(10))
```

**Indicateurs sains** :
- `train_loss` qui descend régulièrement
- `val_loss` qui descend les premières epochs puis se stabilise
- `mIoU` qui monte (vers 0.5 puis 0.6+)

**Indicateurs problématiques** :
- `val_loss` qui re-monte → sur-apprentissage, baisse `--epochs` ou ajoute regularisation
- `train_loss` qui stagne → augmente `--lr` ou vérifie les labels (mauvaise palette ?)
- `mIoU = 0` → labels mal formatés (RGB au lieu d'index, ou inverse)

### Q : Mes prédictions sont toutes en classe 0 (background) ?

Probablement un déséquilibre de classes. Vérifie le log :

```python
print(log[[c for c in log.columns if 'IoU' in c]].mean())
```

Si toutes les classes non-background sont à ~0, **augmente leur poids** dans la loss :

```python
# Dans scripts/train_semap.py, après la creation de criterion :
import torch
class_weights = torch.tensor([0.1, 1.0, 2.0, 2.0, 3.0, 3.0]).to(device)
criterion = CEDiceLoss(num_classes=6, ignore_index=255, class_weights=class_weights)
```

### Q : Le Mask2Former pré-entraîné refuse de se charger.

Vérifie la version `mmcv` : il faut **strictement** `>=2.0.0,<2.2.0`. Avec mmcv 2.2+ il y a des breaking changes incompatibles avec le checkpoint.

```bash
pip show mmcv
# Doit afficher Version: 2.1.x
```

Si tu as 2.2+, désinstalle et réinstalle :

```bash
pip uninstall mmcv mmcv-full
mim install "mmcv>=2.0.0,<2.2.0"
```

### Q : Combien de temps pour entraîner un modèle utilisable ?

Estimations pour atteindre **mIoU ≥ 0.55** :

| Setup | Dataset | Temps |
|---|---|---|
| Colab T4 | SODUCO (305 images) | ~30 min (20 epochs × 1.5 min) |
| Colab T4 | SEMAP real-only (1 439) | ~1h (15 epochs × 4 min) |
| Colab T4 | SEMAP complet (13 561) | ~3-4h (15 epochs × 12-15 min) |
| RTX 2050 | SEMAP real-only | ~3-4h (15 epochs × 15 min) |
| RTX 2050 | SEMAP complet | ~10-12h (déconseillé, utilise Colab) |

---

## Ressources additionnelles

- Doc détaillée SEMAP : [`docs/semap_dataset.md`](./semap_dataset.md)
- Doc Mask2Former : [https://mmsegmentation.readthedocs.io/en/main/](https://mmsegmentation.readthedocs.io/en/main/)
- Repo SODUCO : [https://github.com/soduco/Benchmark_historical_map_vectorization](https://github.com/soduco/Benchmark_historical_map_vectorization)
- Repo SMP (Segmentation Models PyTorch) : [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Article SEMAP : [arXiv 2603.05037](https://doi.org/10.48550/arXiv.2603.05037) (à paraître 2026)

Pour toute question sur l'intégration de ces datasets dans le pipeline existant, consulte aussi le code de référence :

- Chargement données : `pipeline/dataset.py` (SODUCO) et `pipeline/semap_dataset.py` (SEMAP)
- Entraînement : `scripts/train_mapseg.py` et `scripts/train_semap.py`
- Inférence : `pipeline/pipeline.py` → fonction `run_pipeline(with_semantic=True, unet_weights=...)`
