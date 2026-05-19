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
# Toute cette section est ISOLÉE dans une fonction et capture toutes les
# exceptions : si le chargement GDAL plante (mauvais PATH, DLL manquante,
# pywin32 cassé), l'app boote quand même et tombe en mode pixel-only.
# Le drapeau global GDAL_AVAILABLE indique si le SIG est utilisable.
# ---------------------------------------------------------------------------
GDAL_AVAILABLE = False
GDAL_BIN = None
GDAL_LIBRARY_PATH = None
GEOS_LIBRARY_PATH = None


def _configure_gdal_windows():
    """
    Cherche un env conda contenant geos_c.dll + gdal*.dll et configure
    PATH + add_dll_directory pour que rasterio/fiona puissent charger.

    Retourne (GDAL_BIN, gdal_lib_path, geos_lib_path) ou (None, None, None).
    Toutes les exceptions sont capturées et loggées en warning : le boot
    ne doit JAMAIS planter à cause de GDAL — l'app tournera en mode pixel.
    """
    import warnings
    try:
        candidate_paths = []
        user_path = os.environ.get('CONDA_ENV_PATH', '').strip()
        if user_path:
            candidate_paths.append(user_path)
        username = os.environ.get('USERNAME', '')
        for base in (rf'C:\Users\{username}\anaconda3',
                     rf'C:\Users\{username}\miniconda3',
                     r'C:\ProgramData\anaconda3',
                     r'C:\ProgramData\miniconda3'):
            for env_name in ('pfa', 'geo', ''):
                if env_name:
                    candidate_paths.append(os.path.join(base, 'envs', env_name))
                else:
                    candidate_paths.append(base)

        gdal_bin = None
        for path in candidate_paths:
            if not path:
                continue
            bin_dir = os.path.join(path, 'Library', 'bin')
            if os.path.isdir(bin_dir) and os.path.exists(os.path.join(bin_dir, 'geos_c.dll')):
                gdal_bin = bin_dir
                break

        if not gdal_bin:
            warnings.warn(
                "GDAL_BIN introuvable. Mode SIG indisponible — l'app tourne "
                "en pixel-only. Pour activer le SIG, copie webapp/.env.example "
                "et définis CONDA_ENV_PATH. "
                f"Tentés : {candidate_paths[:3]}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None, None

        # Ajoute le dossier au chargeur de DLL (Py 3.8+ Windows)
        try:
            os.add_dll_directory(gdal_bin)
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = gdal_bin + os.path.pathsep + os.environ.get('PATH', '')

        gdal_files = [f for f in os.listdir(gdal_bin)
                      if f.startswith('gdal') and f.endswith('.dll')]
        gdal_lib = os.path.join(gdal_bin, gdal_files[0] if gdal_files else 'gdal.dll')
        geos_lib = os.path.join(gdal_bin, 'geos_c.dll')
        return gdal_bin, gdal_lib, geos_lib

    except Exception as exc:  # noqa: BLE001
        # Filet de sécurité ultime : tout (pywin32, permissions, etc.).
        warnings.warn(
            f"Configuration GDAL ratée ({type(exc).__name__}: {exc}). "
            "L'app boote en mode pixel-only ; le SIG sera indisponible.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, None


if os.name == 'nt':  # Windows seulement
    GDAL_BIN, GDAL_LIBRARY_PATH, GEOS_LIBRARY_PATH = _configure_gdal_windows()
    GDAL_AVAILABLE = GDAL_BIN is not None
else:
    # Linux / macOS / Colab : on présume GDAL trouvable, mais on confirme
    # en testant un import léger. Pas de crash si ça échoue.
    try:
        from osgeo import gdal as _gdal_probe  # noqa: F401
        GDAL_AVAILABLE = True
    except (ImportError, OSError):
        try:
            import rasterio as _rasterio_probe  # noqa: F401
            GDAL_AVAILABLE = True
        except (ImportError, OSError):
            GDAL_AVAILABLE = False

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
