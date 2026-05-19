# `_archive_geo/` — données globales archivées

Ce dossier contient des feuilles de métadonnées **géographiques globales**
qui étaient utilisées par les anciennes versions du pipeline lorsque le
géoréférencement systématique était activé.

Depuis le refactor *pixel-d'abord* (mai 2026), le pipeline tourne par défaut
en coordonnées pixel image — ces feuilles ne sont plus nécessaires pour
produire des GeoJSON utilisables dans Leaflet `CRS.Simple`.

Elles sont conservées ici (et non supprimées) parce qu'elles redeviennent
utiles dès qu'on coche **"Activer le géoréférencement (SIG)"** dans
l'interface, et que `pipeline.georeferencing.AMS_ALGERIA_SHEETS` doit être
complété par d'autres séries.

## Fichiers présents

| Fichier | Contenu | Source |
|---|---|---|
| `metadata_BnF_paris.xlsx` | Métadonnées scans BnF (Bibliothèque nationale de France) — cartes historiques de Paris | Petitpierre 2023 (SEMAP) |
| `metadata_world.xlsx`     | Métadonnées scans monde entier — cartes historiques diverses     | Petitpierre 2023 (SEMAP) |

## Pour les ré-utiliser

```python
import pandas as pd
df = pd.read_excel("data/_archive_geo/metadata_world.xlsx")
# colonnes typiques : lon_nw, lat_nw, lon_se, lat_se, sheet, year, ...
```

Ces métadonnées peuvent alimenter `pipeline/georeferencing.py` pour
calculer une transformation affine par carte, à condition de passer
`georeference=True` au pipeline.
