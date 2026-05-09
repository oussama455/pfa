"""
Réglages Django — Projet cartovec.

Configuration minimale pour un PFA :
    - SQLite (pour démarrer ; migration vers PostGIS pour la V2)
    - App vectorizer enregistrée
    - Médias (uploads) dans webapp/media/
    - django-leaflet pour la carte

Toutes les valeurs sensibles sont chargées via .env (voir .env.example).
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ajout du PROJECT_ROOT au sys.path AVANT toute autre logique.
# Sans ça, `from pipeline.agent import run_agent` plante avec
# ModuleNotFoundError parce que Django est lance depuis webapp/ et le
# package pipeline/ se trouve un cran au-dessus.
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BASE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Charge les variables d'environnement depuis webapp/.env (silencieux si absent)
try:
    from dotenv import load_dotenv
    load_dotenv(_BASE_DIR / '.env')
except ImportError:
    pass  # python-dotenv non installé — les variables système sont utilisées directement


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in ('true', '1', 'yes', 'on')


# ---------------------------------------------------------------------------
# GDAL / GEOS — Windows uniquement
# Sur Linux/macOS/Colab ces librairies sont trouvées automatiquement.
#
# Sur Windows, on définit CONDA_ENV_PATH dans .env (voir .env.example).
# Si CONDA_ENV_PATH est vide ou invalide, on essaie quelques fallbacks
# courants avant d'avertir.
# ---------------------------------------------------------------------------
if os.name == 'nt':  # Windows seulement
    _candidate_paths = []
    _user_path = os.environ.get('CONDA_ENV_PATH', '').strip()
    if _user_path:
        _candidate_paths.append(_user_path)
    # Fallbacks classiques pour env conda nommé pfa, geo, ou base
    _username = os.environ.get('USERNAME', '')
    for _base in (rf'C:\Users\{_username}\anaconda3',
                  rf'C:\Users\{_username}\miniconda3',
                  r'C:\ProgramData\anaconda3',
                  r'C:\ProgramData\miniconda3'):
        for _env_name in ('pfa', 'geo', ''):  # '' = base env
            if _env_name:
                _candidate_paths.append(os.path.join(_base, 'envs', _env_name))
            else:
                _candidate_paths.append(_base)

    GDAL_BIN = None
    for _path in _candidate_paths:
        if not _path:
            continue
        _bin = os.path.join(_path, 'Library', 'bin')
        if os.path.isdir(_bin) and os.path.exists(os.path.join(_bin, 'geos_c.dll')):
            GDAL_BIN = _bin
            break

    if GDAL_BIN:
        try:
            os.add_dll_directory(GDAL_BIN)
        except (OSError, AttributeError):
            pass  # add_dll_directory disponible seulement sur Windows + Py 3.8+
        os.environ['PATH'] = GDAL_BIN + os.path.pathsep + os.environ.get('PATH', '')

        gdal_files = [f for f in os.listdir(GDAL_BIN)
                      if f.startswith('gdal') and f.endswith('.dll')]
        GDAL_LIBRARY_PATH = os.path.join(GDAL_BIN, gdal_files[0] if gdal_files else 'gdal.dll')
        GEOS_LIBRARY_PATH = os.path.join(GDAL_BIN, 'geos_c.dll')
    else:
        import warnings
        warnings.warn(
            "GDAL_BIN introuvable. Définis CONDA_ENV_PATH dans webapp/.env "
            "(copier webapp/.env.example) en pointant vers ton env conda. "
            f"Tente : {_candidate_paths[:3]}",
            RuntimeWarning,
            stacklevel=1,
        )

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent  # racine du dépôt (pfa/)

# ------------------------------------------------------------
# Sécurité
# ------------------------------------------------------------
SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'dev-only-change-me-in-production-0123456789abcdef'
)
DEBUG = _str_to_bool(os.environ.get('DEBUG', 'True'))
ALLOWED_HOSTS = [h.strip() for h in
                 os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
                 if h.strip()]


# ------------------------------------------------------------
# Pipeline (configurable via .env)
# ------------------------------------------------------------
PIPELINE_MAX_DIMENSION = int(os.environ.get('PIPELINE_MAX_DIMENSION', '2400'))
PIPELINE_FILTER_MARGIN = int(os.environ.get('PIPELINE_FILTER_MARGIN', '20'))


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
    'rest_framework'
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
