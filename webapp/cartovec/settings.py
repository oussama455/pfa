"""
Réglages Django — Projet cartovec.

Configuration minimale pour un PFA :
    - SQLite (pour démarrer ; migration vers PostGIS pour la V2)
    - App vectorizer enregistrée
    - Médias (uploads) dans webapp/media/
    - django-leaflet pour la carte
"""
import os
import sys
from pathlib import Path

CONDA_PREFIX = r'C:\Users\ochou\anaconda3\envs\geo'
GDAL_BIN = os.path.join(CONDA_PREFIX, 'Library', 'bin')

# Add GDAL bin directory so dependent DLLs are found correctly.
os.add_dll_directory(GDAL_BIN)
os.environ['PATH'] = GDAL_BIN + os.path.pathsep + os.environ['PATH']

# Auto-detect GDAL DLL name (e.g., gdal.dll, gdal310.dll) to avoid hardcoded version issues.
gdal_files = [f for f in os.listdir(GDAL_BIN) if f.startswith('gdal') and f.endswith('.dll')]
if gdal_files:
    GDAL_LIBRARY_PATH = os.path.join(GDAL_BIN, gdal_files[0])
else:
    GDAL_LIBRARY_PATH = os.path.join(GDAL_BIN, 'gdal.dll')

GEOS_LIBRARY_PATH = os.path.join(GDAL_BIN, 'geos_c.dll')
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent  # racine du dépôt (pfa/)

# ------------------------------------------------------------
# Sécurité
# ------------------------------------------------------------
SECRET_KEY = 'dev-only-change-me-in-production-0123456789abcdef'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']


# ------------------------------------------------------------
# Applications
# ------------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'leaflet',
    'djgeojson',
    'vectorizer',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'cartovec.urls'
WSGI_APPLICATION = 'cartovec.wsgi.application'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


# ------------------------------------------------------------
# Base de données
# ------------------------------------------------------------
# SQLite pour démarrer.
# V2 : remplacer par PostgreSQL + PostGIS et ajouter 'django.contrib.gis'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# ------------------------------------------------------------
# Internationalisation
# ------------------------------------------------------------
LANGUAGE_CODE = 'fr-fr'
TIME_ZONE = 'Africa/Tunis'
USE_I18N = True
USE_TZ = True


# ------------------------------------------------------------
# Fichiers statiques et médias
# ------------------------------------------------------------
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Dossier où le pipeline écrit les GeoJSON
PROCESSED_DIR = MEDIA_ROOT / 'processed'


# ------------------------------------------------------------
# django-leaflet : réglages de la carte par défaut
# ------------------------------------------------------------
LEAFLET_CONFIG = {
    # Centre sur la Tunisie par défaut
    'DEFAULT_CENTER': (34.0, 9.5),
    'DEFAULT_ZOOM': 6,
    'MIN_ZOOM': 3,
    'MAX_ZOOM': 18,
    'TILES': [
        ('OSM', 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
         '&copy; OpenStreetMap contributors'),
    ],
}


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

