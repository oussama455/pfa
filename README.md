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
- Agent IA : perception, traitement, QA, auto-correction, georeferencement et
  export avec journal d'audit.
- Active Learning : les corrections humaines ajustent progressivement les
  plages HSV par serie de carte.
- Backend Django : upload, suivi de statut, API REST, stockage des corrections
  et exposition des GeoJSON produits.
- Frontend React/Vite : upload, liste des cartes, visualisation Leaflet,
  edition/suppression de features et panneau Active Learning.
- Dataset SEMAP : chargeur PyTorch et scripts d'entrainement U-Net.

Le document de cadrage est disponible ici :
[`PFA_Cadrage_Oussama_Chouaibi.docx`](./PFA_Cadrage_Oussama_Chouaibi.docx).

## Arborescence

```text
pfa/
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- environment_cuda.yml
|-- data/
|   |-- dataset_config.json       # configuration HSV canonique
|   |-- semap_config.json         # chemin local du dataset SEMAP
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
conda env create -f environment_cuda.yml
conda activate pfa
python scripts/test_gpu.py
```

`environment_cuda.yml` installe PyTorch CUDA 12.1 depuis l'index officiel
PyTorch. Si CUDA n'est pas disponible, le pipeline peut retomber sur CPU.

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

Segmentation couleur et vectorisation :

```bash
python -m pipeline.pipeline data/raw/carte.png -o data/processed
```

Avec segmentation semantique U-Net :

```bash
python -m pipeline.pipeline data/raw/carte.png ^
  -o data/processed ^
  --semantic ^
  --weights external/weight/semap_unet_best.pth ^
  --device auto
```

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

## Dataset SEMAP

SEMAP est utilise pour entrainer un U-Net de segmentation semantique en
6 classes :

`background`, `contours`, `built`, `non_built`, `water`, `road_network`.

Le dataset reste hors Git. Le chemin local est defini dans
[`data/semap_config.json`](./data/semap_config.json), champ `external_root`.

Verifier le dataset :

```python
from pipeline.semap_dataset import SemapDataset

ds = SemapDataset(split="train", target_size=(512, 512), augment=True)
print(ds.stats())
```

Entrainer le modele :

```bash
python scripts/train_semap.py --epochs 20 --batch-size 8
python scripts/train_semap.py --target-size 384 --batch-size 4
python scripts/train_semap.py --no-synthetic --epochs 30
```

Les checkpoints sont ecrits dans `external/weight/` et ignores par Git.

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
