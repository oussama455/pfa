# CartoVec — Interface web Django

Interface web du pipeline de vectorisation. Permet à un utilisateur de :

1. Téléverser une carte raster (PNG / JPG / TIFF)
2. Suivre le statut du traitement en direct
3. Visualiser les couches extraites sur une carte Leaflet
4. Télécharger les GeoJSON

## Lancement en développement

Depuis la racine du projet `pfa/` :

```bash
source .venv/bin/activate         # ou .venv\Scripts\activate sous Windows
cd webapp
python manage.py migrate          # initialise la base SQLite
python manage.py createsuperuser  # crée un compte admin (optionnel)
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000/

## Structure

```
webapp/
├── manage.py
├── cartovec/               ← projet Django (settings, urls, wsgi)
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── vectorizer/             ← app principale
│   ├── models.py               → MapUpload
│   ├── forms.py                → formulaire d'upload
│   ├── views.py                → 3 vues : upload, result, status (JSON)
│   ├── urls.py
│   ├── tasks.py                → orchestrateur pipeline (thread)
│   ├── admin.py
│   └── templates/vectorizer/
│       ├── base.html
│       ├── upload.html
│       └── result.html
└── media/                  ← uploads + résultats GeoJSON (généré)
    ├── uploads/
    └── processed/
        └── map_<id>/           → un dossier par carte traitée
```

## Limitations V1 et évolutions V2

**V1 (actuel)**

- Base SQLite, pas de PostGIS
- Pipeline exécuté dans un thread simple (pas Celery)
- Pas d'authentification utilisateur
- Segmentation couleur uniquement (pas de U-Net activé)

**V2 (à faire pour la soutenance, si temps)**

- PostgreSQL + PostGIS, activer `django.contrib.gis`
- Celery + Redis pour le pipeline en tâche de fond
- Authentification utilisateur (login/logout, permissions)
- Activation du module U-Net (`with_semantic=True` dans `tasks.py`)
- Saisie des points de contrôle (GCPs) directement sur la carte Leaflet
