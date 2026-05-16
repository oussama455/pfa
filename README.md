# CartoVec - PFA vectorisation de cartes historiques

**Etudiant :** Oussama CHOUAIBI - Geomatique 2e annee  
**Encadrant :** Kamel BENRAIS - Ingenieur Principal, ELFOULADH  
**Ecole :** EABA - Tunisie

CartoVec transforme des cartes topographiques raster scannees en couches
vectorielles exploitables dans un SIG. Le projet combine un pipeline Python
OpenCV/PyTorch/GeoPandas, un agent LangGraph, une API Django REST et une
interface React/Vite avec Leaflet pour visualiser et corriger les resultats.

## Etat actuel

- Pipeline image : pretraitement, segmentation couleur, segmentation U-Net,
  vectorisation, georeferencement et export GeoJSON.
- Detection en 2 etapes du cadre cartographique : neatline (marge externe)
  + legende interne (panneau droit). Calibre sur 8 cartes reelles
  AMS/GSGS 1:50 000 Tunisie + Algerie : Bizerte, Tunis, Ain El Kseiba,
  Ain Bessem, Alger, Terny, Warnier, Renault.
- Plages HSV calibrees (`data/config.json`) : amelioration nette
  de la detection des routes rouges (S_min 90 to 60) et de la vegetation
  delavee (S_min 40 to 20).
- Agent IA : perception, traitement, QA, auto-correction, georeferencement et
  export avec journal d'audit. Fallback automatique vers pipeline classique
  si LangGraph est indisponible.
- Active Learning : les corrections humaines ajustent progressivement les
  plages HSV par serie de carte (EMA alpha=0.3).
- Backend Django : upload, suivi de statut, API REST, stockage des corrections
  et exposition des GeoJSON produits.
- Frontend React/Vite : upload, liste des cartes, visualisation Leaflet,
  edition/suppression de features et panneau Active Learning.
- 2 datasets disponibles :
  - SODUCO Benchmark (`data/historical_maps/`, 256 train + 49 eval,
    5 classes BGR).
  - SEMAP (Petitpierre 2025, 1 439 reelles + 12 122 synthetiques,
    6 classes index-based, 13 561 echantillons au total).
- Modele Mask2Former Swin-L pre-entraine fourni avec SEMAP (908 MB,
  framework mmsegmentation, mIoU rapporte ~0.78).

Le document de cadrage est disponible ici :
[`PFA_Cadrage_Oussama_Chouaibi.docx`](./PFA_Cadrage_Oussama_Chouaibi.docx).

## Quickstart (5 lignes)

```bash
conda env create -f environment.yml && conda activate pfa
python scripts/diagnose.py --check env                          # verifie l'env
python -m pipeline.pipeline data/raw/carte.png -o data/processed
cd webapp && python manage.py migrate && python manage.py runserver  &
cd webapp/frontend && npm install && npm run dev
```

Puis ouvre [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Arborescence

```text
pfa/
|-- README.md
|-- requirements.txt
|-- environment.yml               # env conda unifie (CPU + instructions CUDA)
|-- data/
|   |-- config.json               # config unifiee (HSV, frame_detection, semap, soduco)
|   |-- raw/                      # cartes raster d'entree, hors Git
|   |-- processed/                # sorties generees, hors Git
|-- docs/
|-- external/
|   |-- weight/                   # checkpoints locaux, hors Git
|-- notebooks/
|-- pipeline/                     # coeur Python du traitement
|-- scripts/                      # entrainement, diagnostics, preparation GT
|-- shared/
|-- webapp/
|   |-- cartovec/                 # configuration Django
|   |-- vectorizer/               # models, views, API, tasks
|   |-- frontend/                 # React/Vite + Leaflet
```

## Installation

Sur Windows, Conda est recommande pour GDAL, Rasterio, Fiona et GeoPandas.

### CPU / environnement standard

```bash
conda env create -f environment.yml
conda activate pfa
python -m ipykernel install --user --name pfa --display-name "Python (pfa)"
```

### GPU NVIDIA / CUDA 12.1

```bash
conda env create -f environment.yml
conda activate pfa
pip uninstall -y torch torchvision
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
python scripts/diagnose.py --check gpu
```

L'ancien `environment_cuda.yml` a ete fusionne dans `environment.yml` —
la difference CUDA/CPU se gere par les commandes pip ci-dessus.

### Alternative pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Les dependances frontend sont gerees separement par npm :

```bash
cd webapp/frontend
npm install
```

## Lancer le projet complet

Terminal 1 - backend Django :

```bash
conda activate pfa
cd webapp
python manage.py migrate
python manage.py runserver
```

Backend/API : http://127.0.0.1:8000/

Terminal 2 - frontend React :

```bash
cd webapp/frontend
npm install
npm run dev
```

Frontend : http://127.0.0.1:5173/

Le serveur Vite proxifie automatiquement `/api` et `/media` vers Django sur
`127.0.0.1:8000`.

## Relation pipeline - webapp

1. Le frontend envoie une carte a `POST /api/maps/`.
2. Django cree un `MapUpload` puis appelle `enqueue_pipeline()`.
3. `webapp/vectorizer/tasks.py` lance un thread de traitement.
4. Le thread appelle `pipeline.agent.run_agent()`.
5. Le pipeline ecrit les GeoJSON dans `webapp/media/processed/map_<id>/`.
6. `MapUpload.output_layers` expose ces fichiers sous forme d'URLs `/media/...`.
7. React charge `/api/maps/<id>/geojson/` et affiche les couches dans Leaflet.
8. Les corrections HITL sont envoyees a `/api/maps/<id>/corrections/`.
9. `api_v2.py` sauvegarde les corrections et met a jour `pipeline.active_learning`.

Principaux endpoints REST :

```text
GET    /api/maps/
POST   /api/maps/
GET    /api/maps/<id>/
GET    /api/maps/<id>/status/
GET    /api/maps/<id>/geojson/
PATCH  /api/maps/<id>/corrections/
GET    /api/calibration/<series>/
GET    /api/calibration/history/?map_id=<id>
POST   /api/calibration/<series>/reset/
```

## Utilisation du pipeline seul

Segmentation couleur, vectorisation et detection 2-stages du cadre :

```bash
python -m pipeline.pipeline data/raw/carte.png -o data/processed
```

Options CLI principales :

```bash
# Desactive la suppression de la legende interne (Stage 2)
python -m pipeline.pipeline data/raw/carte.png --no-remove-legend

# Desactive les plages HSV calibrees (utilise les valeurs generiques)
python -m pipeline.pipeline data/raw/carte.png --no-calibrated-hsv

# Recadrage manuel si la detection automatique echoue
python -m pipeline.pipeline data/raw/carte.png --bbox 180 220 2400 1800

# Force CPU ou GPU pour l'inference U-Net
python -m pipeline.pipeline data/raw/carte.png --device cpu
python -m pipeline.pipeline data/raw/carte.png --device cuda
```

Avec segmentation semantique U-Net :

```bash
python -m pipeline.pipeline data/raw/carte.png ^
  -o data/processed ^
  --semantic ^
  --weights external/weight/semap_unet_best.pth ^
  --device auto
```

Le pipeline detecte automatiquement le nombre de classes a charger selon le
nom des poids : `semap_*.pth` -> 6 classes (SEMAP), sinon retombe sur les
5 classes SODUCO si `data/historical_maps/classes.json` est present, sinon
3 classes par defaut.

Agent IA :

```python
from pipeline.agent import run_agent

result = run_agent(
    raster_path="data/raw/tunis_1969.jpg",
    output_dir="data/processed/tunis_1969",
    map_name="tunis",
    weights_path="external/weight/semap_unet_best.pth",
    device="auto",
)

print(result["output_geojsons"])
```

## Datasets

> Guide pas a pas detaille pour les 3 datasets : [`docs/datasets_guide.md`](./docs/datasets_guide.md)

### 1. SODUCO Benchmark (`data/historical_maps/`)

Dataset Paris BnF avec labels en couleurs BGR (palette via
`data/historical_maps/classes.json`). 5 classes : `background`,
`inside_blocks`, `building_walls`, `street_network`, `unknown`.

- 256 images train + 49 eval
- Resolution 1000x1000
- Utilise par `scripts/train.py --dataset soduco`

### 2. SEMAP (Petitpierre 2025, EPFL)

Le plus gros dataset annote pour la segmentation de cartes historiques.
6 classes index-based : `background`, `contours`, `built`, `non_built`,
`water`, `road_network`.

- 1 439 images reelles + 12 122 images synthetiques (13 561 total)
- Splits officiels : 10 703 train / 2 712 val / 143 test
- Resolution 768x768 a 1000x1000
- Modele Mask2Former Swin-L pre-entraine fourni (908 MB)
- Citation : Petitpierre R, Gomez Donoso D, Kriesel B (2025).
  DOI 10.5281/zenodo.16164781

Le dataset reste hors Git (2 GB). Le chemin local est defini dans
[`data/config.json (bloc `semap`)`](./data/config.json (bloc `semap`)), champ `external_root`.

Verifier le dataset :

```python
from pipeline.semap_dataset import SemapDataset

ds = SemapDataset(split="train", target_size=(512, 512), augment=True)
print(ds.stats())
# {'split': 'train', 'total': 10703, 'real': 1153, 'synthetic': 9550}
```

Entrainer un U-Net leger (compatible RTX 2050 4 GB) :

```bash
# Tout le dataset (real + synthetic)
python scripts/train.py --dataset semap --epochs 20 --batch-size 8

# Uniquement les images reelles
python scripts/train.py --dataset semap --no-synthetic --epochs 30

# Image plus petite pour GPU 4 GB
python scripts/train.py --dataset semap --target-size 384 --batch-size 4
```

Les checkpoints sont ecrits dans `external/weight/` (ignores par Git).

### 3. Mask2Former pre-entraine (option avancee)

Utilisable directement via mmsegmentation pour de meilleurs scores
(mIoU ~0.78 contre ~0.65 pour le U-Net). Necessite ~6 GB VRAM en
inference -> Colab T4 recommande. Voir [`docs/semap_dataset.md`](./docs/semap_dataset.md).

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0,<2.2.0" "mmsegmentation>=1.2.2"
```

---

## Entraînement des datasets dans Google Colab

Tu peux entraîner tous les modèles (U-Net, Mask2Former) sur Google Colab avec GPU/TPU. Voici comment faire :

1. **Ouvre un notebook dans Colab** (ou dans VS Code avec le kernel Colab).
2. **Clone le repo et installe les dépendances** :
  ```python
  from notebooks.colab_setup import setup

  project_root = setup(
     repo_url="https://github.com/<user>/pfa.git",
     branch="main",
     install=True,
     use_drive=True,  # Recommandé pour les gros fichiers
  )
  ```
  Place tes datasets dans `/content/drive/MyDrive/pfa/data/` si tu utilises Google Drive.
3. **Lance l'entraînement** :
  ```python
  !python scripts/train.py --dataset semap --epochs 20 --batch-size 8
  ```
  ou pour Mask2Former :
  ```python
  !pip install -U openmim
  !mim install mmengine "mmcv>=2.0.0,<2.2.0" "mmsegmentation>=1.2.2"
  # puis ton script d'entraînement avancé
  ```

Les checkpoints et résultats peuvent être sauvegardés sur Google Drive pour éviter toute perte.

---

### Cartes Tunisie + Algerie calibrees

Le fichier [`data/config.json`](./data/config.json) contient
les plages HSV calibrees sur 8 cartes militaires reelles (AMS/GSGS 1:50 000,
WWII) :

| Region | Cartes |
|---|---|
| Tunisie | Bizerte, Tunis, Ain El Kseiba |
| Algerie | Ain Bessem, Alger, Terny, Warnier, Renault |

Coverage observee : 14-50% selon densite. Activees par defaut via
`--no-calibrated-hsv` pour les desactiver.

## Fichiers generes et Git

Les elements suivants ne doivent pas etre versionnes :

- `webapp/media/uploads/`
- `webapp/media/processed/`
- `webapp/frontend/node_modules/`
- `webapp/frontend/dist/`
- `data/processed/`, `outputs/`, `runs/`
- checkpoints PyTorch (`*.pth`, `*.pt`, `*.ckpt`)
- exports GeoJSON/Shapefile/GeoTIFF generes

`webapp/frontend/package-lock.json` doit etre conserve pour rendre les builds
frontend reproductibles.

## Diagnostics et tests

Scripts utiles pour verifier ton installation :

```bash
python scripts/diagnose.py --check env    # verifie cv2, rasterio, geopandas, GDAL, GPU
python scripts/diagnose.py --check gpu        # benchmark forward U-Net CPU vs GPU
```

`diagnose_env.py` identifie precisement le module qui plante en cas de
"DLL load failed" sous Windows. La cause la plus frequente sur Anaconda est
un conflit entre pyogrio (backend GeoPandas) et fiona — `vectorization.py`
bascule automatiquement de l'un a l'autre si le premier echoue.

## Google Colab

Les notebooks peuvent etre lances dans Colab avec :

```python
from notebooks.colab_setup import setup

project_root = setup(
    repo_url="https://github.com/<user>/pfa.git",
    branch="main",
    install=True,
)
```

Pour des cartes lourdes, utilisez `use_drive=True` et placez le projet dans
`/content/drive/MyDrive/pfa`.

## Licences

| Ressource | Licence |
|---|---|
| SEMAP dataset | CC BY 4.0 |
| SODUCO Benchmark historical map vectorization | MIT |
| Satellite Imagery Road Segmentation | MIT |
| django-leaflet | LGPL |
| OpenCV | Apache 2.0 |
| Rasterio / GeoPandas / Shapely | BSD |
| Django | BSD |
| React / Vite / Leaflet | MIT / BSD-style |

## Contact

ochouaibi1919@gmail.com
