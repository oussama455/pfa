# PFA — Agent IA de vectorisation de cartes militaires

**Étudiant :** Mohamed GHARBI — Géomatique 2ᵉ année
**Encadrant :** Kamel BENRAIS — Ingénieur Principal, ELFOULADH
**École :** EABA — Tunisie

> Environnement : **Anaconda** (Python 3.10, voir `environment.yml`)

> Agent intelligent pour transformer des cartes topographiques raster (scan/papier) en données vectorielles géoréférencées (GeoJSON / Shapefile), utilisables dans un SIG militaire.

---

## Philosophie

Ce projet **n'est pas** une réécriture de modèles de deep learning depuis zéro.
Il **assemble** des briques open source éprouvées dans un pipeline cohérent, exposé via une interface Django.

Pour le détail de la stratégie et de l'architecture, voir le document de cadrage : [`PFA_Cadrage_Mohamed_GHARBI.docx`](./PFA_Cadrage_Mohamed_GHARBI.docx).

---

## Architecture

```
Carte raster (PNG/TIF)
        │
        ▼
┌───────────────────────┐
│  Prétraitement        │  OpenCV (débruitage, HSV, normalisation)
└───────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌───────────────────────┐    ┌───────────────────────┐
│  Segmentation couleur │    │  Segmentation U-Net   │
│  (eau, végétation,    │    │  (routes, bâtiments)  │
│   courbes, rouge)     │    │                       │
│  — OpenCV             │    │  — PyTorch + SMP      │
└───────────────────────┘    └───────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
        ┌───────────────────────────────┐
        │  Vectorisation                │  rasterio.features + Shapely
        │  (raster masks → geometries)  │
        └───────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │  Géoréférencement             │  GDAL / rasterio (GCPs)
        └───────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │  Export GeoJSON / Shapefile   │  GeoPandas
        └───────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │  Django + Leaflet             │  Visualisation + download
        └───────────────────────────────┘
```

---

## Arborescence du projet

```
pfa/
├── README.md
├── requirements.txt
├── .gitignore
├── PFA_Cadrage_Mohamed_GHARBI.docx   ← document de cadrage
│
├── pipeline/              ← cœur du traitement (modules Python)
│   ├── __init__.py
│   ├── preprocessing.py       → nettoyage, HSV, détection auto du cadre carto
│   ├── color_segmentation.py  → eau, végétation, courbes, rouge (OpenCV + density_filter)
│   ├── grid_extraction.py     → détection auto du quadrillage kilométrique
│   ├── semantic_segmentation.py → routes, bâtiments (U-Net, optionnel)
│   ├── vectorization.py       → raster → vecteur (Shapely)
│   ├── georeferencing.py      → GCPs auto depuis 4 coins + quadrillage (GDAL)
│   └── pipeline.py            → orchestrateur bout en bout
│
├── webapp/                ← application Django
│   ├── manage.py
│   ├── cartovec/              → projet Django
│   └── vectorizer/            → app Django
│
├── notebooks/             ← exploration et démos
│   ├── 01_color_segmentation_demo.ipynb   → segmentation couleur + vectorisation
│   ├── 02_hsv_calibration.ipynb           → calibration interactive HSV
│   └── 03_georeferencing.ipynb            → géoréférencement automatique
│
├── data/
│   ├── raw/                   → cartes raster d'entrée
│   └── processed/             → vecteurs de sortie
│
├── external/              ← repos open source clonés (via scripts/clone_sources.sh)
│
├── scripts/               ← utilitaires
│   └── clone_sources.sh       → clone soduco et autres
│
├── tests/                 ← tests unitaires
└── docs/                  ← rapport, slides (à venir)
```

---

## Option A — Google Colab (recommandé : aucune install, GPU T4 gratuit)

Si tu n'as pas envie de te battre avec Anaconda + Windows DLLs, ouvre les notebooks dans Colab.

**Avantages** :
- Pas d'installation locale
- GPU T4 gratuit (15 Go VRAM, ~3x plus que ta RTX 2050) pour le U-Net
- Tu peux travailler depuis n'importe quel PC (ou ton téléphone)

**Comment faire** :

1. Pousse ton projet sur GitHub (ou utilise Google Drive — voir option 3).
2. Va sur [colab.research.google.com](https://colab.research.google.com/).
3. **File > Upload notebook** → sélectionne `notebooks/01_color_segmentation_demo.ipynb`.
4. Active la GPU : **Exécution > Modifier le type d'exécution > T4 GPU**.
5. Dans la première cellule du notebook, mets ton `REPO_URL` (par exemple `https://github.com/Mohamed-GHARBI/pfa.git`).
6. Lance les cellules dans l'ordre. La cellule de Setup pip-install les dépendances et clone ton repo automatiquement.

**3 façons de fournir tes données dans Colab** :

| Méthode | Quand l'utiliser | Comment |
|---|---|---|
| **GitHub clone** | Données pas trop grosses (<100 Mo), versionnées | `REPO_URL = '...'` dans la première cellule |
| **Google Drive** | Cartes lourdes que tu ne veux pas pousser sur GitHub | `USE_DRIVE = True`, projet dans `MyDrive/pfa/` |
| **Upload manuel** | Test ponctuel, une seule carte | Icône dossier 📁 → glisse-dépose dans `pfa/data/raw/` |

**Note importante** : si tu modifies un fichier `.py` du pipeline pendant que le notebook tourne dans Colab, fais `Exécution > Redémarrer l'exécution` (sinon les changements ne sont pas pris en compte).

---

## Option B — Local Anaconda (recommandé sur Windows)

GDAL, rasterio, geopandas et fiona sont pénibles à installer via pip sur Windows. Avec conda-forge ils s'installent en une commande.

### 1. Prérequis

- **Anaconda** ou **Miniconda** ([miniconda.org](https://docs.conda.io/en/latest/miniconda.html))
- **Git**
- Optionnel : **QGIS** pour valider visuellement les sorties

### 2. Création de l'environnement

Ouvre **Anaconda Prompt** (pas le CMD Windows classique) et lance :

```bash
cd C:\Users\ochou\Documents\Claude\Projects\pfa
conda env create -f environment.yml
conda activate pfa
```

La création prend 5-10 min (téléchargement des paquets conda-forge). Tu auras un env nommé `pfa` avec Python 3.10, GDAL, rasterio, geopandas, opencv, jupyter, etc.

### 3. Enregistrer le kernel Jupyter

```bash
python -m ipykernel install --user --name pfa --display-name "Python (pfa)"
```

→ Le kernel apparaît dans Jupyter comme `Python (pfa)`. Choisis-le dans `Kernel > Change kernel` à l'ouverture des notebooks.

### 3 bis. Activer le GPU (RTX 2050)

PyTorch installé via `environment.yml` est en version CPU par défaut. Pour bénéficier de ton GPU NVIDIA RTX 2050 (forte accélération sur l'U-Net) :

```bash
conda activate pfa
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Vérifie ensuite que CUDA est bien détectée :

```bash
python scripts/test_gpu.py
```

Sortie attendue :

```
PyTorch       : 2.x.x+cu121
CUDA dispo    : True
GPUs detectees: 1
  [0] NVIDIA GeForce RTX 2050  (compute 8.6, 4.00 Go VRAM)
Benchmark : forward pass U-Net resnet34, batch (1, 3, 512, 512)
  CPU  : ~2500 ms / forward
  GPU  : ~50 ms / forward
  Acceleration GPU : x50
```

**Si pas de GPU détectée** → le pipeline fait du fallback automatique sur CPU. Tout fonctionne, c'est juste plus lent pour la partie U-Net (la segmentation couleur reste rapide quel que soit le device).

Le pipeline auto-détecte le device via `pipeline.semantic_segmentation.get_device()`. Tu peux forcer le device en CLI :

```bash
python -m pipeline.pipeline data/raw/carte.png --semantic --device cuda     # force GPU
python -m pipeline.pipeline data/raw/carte.png --semantic --device cpu      # force CPU
python -m pipeline.pipeline data/raw/carte.png --semantic --device auto     # défaut
```

### 4. Cloner les repos open source externes

```bash
bash scripts/clone_sources.sh
```

Cela clone dans `external/` : `soduco/Benchmark_historical_map_vectorization`, `farhad-dalirani/Satellite-Imagery-Road-Segmentation`, `makinacorpus/django-leaflet`.

### 5. Lancer Jupyter et les notebooks

```bash
jupyter lab
# ou : jupyter notebook
```

Ouvre dans l'ordre :
1. `notebooks/01_color_segmentation_demo.ipynb` — premier résultat visuel sans deep learning.
2. `notebooks/02_hsv_calibration.ipynb` — calibre les plages HSV sur ta carte.
3. `notebooks/03_georeferencing.ipynb` — génère les GCPs depuis le quadrillage et géoréférence.

Chaque notebook commence par une cellule **Vérification de l'environnement** qui contrôle que les imports critiques fonctionnent.

### 6. Lancer l'application web

```bash
cd webapp
python manage.py migrate
python manage.py runserver
```

→ http://127.0.0.1:8000/

### Alternative : venv + pip (Linux/macOS)

Si tu n'utilises pas Anaconda :

```bash
python -m venv .venv
source .venv/bin/activate     # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

Sur Linux il faut d'abord `sudo apt install gdal-bin libgdal-dev` avant `pip install rasterio`.

---

## Plan de travail (résumé)

Voir le document de cadrage pour le détail. Résumé :

| Phase | Durée | Livrable |
|-------|-------|----------|
| P1 | S1–S2 | Environnement + carte de test + notebook |
| P2 | S3–S4 | Script de segmentation couleur (OpenCV) |
| P3 | S5–S6 | Script de segmentation U-Net (routes/bâtiments) |
| P4 | S7–S8 | Vectorisation + géoréférencement + GeoJSON |
| P5 | S9–S10 | Interface Django complète |
| P6 | S11–S12 | Rédaction rapport + slides |

---

## Licences des briques externes

| Repo | Licence |
|------|---------|
| soduco/Benchmark_historical_map_vectorization | MIT |
| farhad-dalirani/Satellite-Imagery-Road-Segmentation | MIT |
| makinacorpus/django-leaflet | LGPL |
| OpenCV | Apache 2.0 |
| rasterio / GeoPandas / Shapely | BSD |
| Django | BSD |

Toutes compatibles avec un projet académique et une réutilisation du code.

---

## Contact

Pour toute question sur le projet : cristoumohamedgharbi@gmail.com
