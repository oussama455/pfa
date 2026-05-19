# Pipeline CartoVec simplifie

Objectif PFA :

```text
carte scannee -> recadrage -> segmentation HSV -> vectorisation -> GeoJSON
```

## Commande recommandee

```bash
python -m pipeline.simple_pipeline data/raw/carte.png -o data/processed/simple
```

## Fichiers a connaitre pour la soutenance

| Fichier | Role simple |
|---|---|
| `simple_pipeline.py` | Chaine courte pour demo |
| `preprocessing.py` | Recadre la carte et retire la legende |
| `color_segmentation.py` | Detecte eau, vegetation, routes rouges, courbes, batiments |
| `vectorization.py` | Transforme les masques en GeoJSON |
| `pipeline.py` | Version complete avec options avancees |

## Fichiers avances

| Fichier | A presenter seulement si question |
|---|---|
| `semantic_segmentation.py` | U-Net pour segmentation Deep Learning |
| `semap_dataset.py` | Chargement dataset SEMAP |
| `dataset.py` | Chargement dataset SODUCO |
| `georeferencing.py` | GCP et calage spatial |
| `agent.py` | Agent LangGraph |
| `active_learning.py` | Ajustement HSV par corrections |
| `cc_postprocess.py` | Nettoyage par composantes connexes |
| `grid_extraction.py` | Extraction de grille |
| `create_masks.py` | Preparation de masques |
| `paths.py` | Chemins projet |

## Message de soutenance

CartoVec commence par une version simple et robuste : il lit une carte
militaire scannee, garde le cadre utile, detecte les couleurs cartographiques,
puis exporte des couches SIG. Les modules U-Net et agent IA sont des extensions.
