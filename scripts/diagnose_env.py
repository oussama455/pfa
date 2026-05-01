"""
Diagnostic d'environnement pour le PFA.

Lance avec :
    python scripts/diagnose_env.py

Teste un par un les modules dont depend le pipeline et identifie
PRECISEMENT lequel echoue. Utile quand la webapp ou un notebook plante
avec un message generique du genre "DLL load failed while importing lib".
"""
from __future__ import annotations

import os
import platform
import sys
import traceback


def header(title):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def test_import(label, code, hint=""):
    """Tente un import et affiche OK / FAIL + traceback resume + hint."""
    try:
        exec(code, {"__name__": "__main__"})
        print(f"  OK   {label}")
        return True
    except Exception as exc:
        print(f"  FAIL {label}")
        print(f"       {type(exc).__name__}: {exc}")
        if hint:
            print(f"       -> {hint}")
        return False


# ---------------------------------------------------------------------
# Infos systeme
# ---------------------------------------------------------------------
header("Systeme")
print(f"  OS       : {platform.system()} {platform.release()}")
print(f"  Python   : {platform.python_version()}  ({sys.executable})")
print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', '(aucun)')}")
print(f"  PATH (1ere entrees) :")
for p in os.environ.get("PATH", "").split(os.pathsep)[:5]:
    print(f"    {p}")


# ---------------------------------------------------------------------
# Imports critiques
# ---------------------------------------------------------------------
header("Modules critiques")

# Niveau 1 : numpy / matplotlib (rarement en panne)
test_import("numpy",      "import numpy")
test_import("matplotlib", "import matplotlib")

# Niveau 2 : OpenCV (cause classique de DLL load failed sur Windows)
ok_cv2 = test_import(
    "cv2 (OpenCV)",
    "import cv2; assert cv2.__version__",
    hint="Conflict possible conda+pip. Reinstalle proprement :\n"
         "       pip uninstall opencv-python opencv-python-headless\n"
         "       conda remove --force opencv libopencv py-opencv\n"
         "       pip install opencv-python==4.10.0.84",
)

# Niveau 3 : scikit-image
test_import("skimage",     "import skimage")

# Niveau 4 : Shapely
test_import("shapely",     "import shapely; from shapely.geometry import Polygon")

# Niveau 5 : rasterio (cause typique sur Windows : conflit GDAL/PROJ)
ok_rio = test_import(
    "rasterio",
    "import rasterio; from rasterio import features",
    hint="Conflit GDAL/PROJ probable. Recree env conda strict :\n"
         "       conda create -n pfa python=3.10 -c conda-forge "
         "--strict-channel-priority -y rasterio",
)

# Niveau 6 : pyogrio (backend GeoPandas par defaut, cause typique de "DLL load failed lib")
ok_pyogrio = test_import(
    "pyogrio (backend GeoPandas par defaut)",
    "import pyogrio; from pyogrio import _io",
    hint="C'est probablement la cause de ton 'DLL load failed lib'.\n"
         "       Solution : passer a fiona en backup :\n"
         "       pip uninstall pyogrio\n"
         "       pip install fiona\n"
         "       (le pipeline detecte automatiquement et utilise fiona)",
)

# Niveau 7 : fiona (alternative a pyogrio)
ok_fiona = test_import(
    "fiona (alternative pyogrio)",
    "import fiona",
    hint="Si pyogrio est casse et fiona aussi : pip install fiona",
)

# Niveau 8 : geopandas
ok_gpd = test_import(
    "geopandas",
    "import geopandas",
    hint="geopandas depend de pyogrio OU fiona. Installe au moins l'un des deux.",
)

# Niveau 9 : PyTorch
test_import(
    "torch (PyTorch)",
    "import torch; print(f'         CUDA dispo : {torch.cuda.is_available()}')",
    hint="Pas critique pour la segmentation couleur, requis seulement\n"
         "       si tu utilises with_semantic=True (U-Net).",
)


# ---------------------------------------------------------------------
# Test fonctionnel : la pipeline peut-elle ecrire un GeoJSON ?
# ---------------------------------------------------------------------
header("Test fonctionnel : ecriture GeoJSON")

if ok_gpd and (ok_pyogrio or ok_fiona):
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"layer": ["test"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs=None,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
            out_path = tmp.name

        # Essai pyogrio
        if ok_pyogrio:
            try:
                gdf.to_file(out_path, driver="GeoJSON", engine="pyogrio")
                print("  OK   ecriture GeoJSON via pyogrio")
            except Exception as exc:
                print(f"  FAIL ecriture pyogrio : {type(exc).__name__}: {exc}")

        # Essai fiona
        if ok_fiona:
            try:
                gdf.to_file(out_path, driver="GeoJSON", engine="fiona")
                print("  OK   ecriture GeoJSON via fiona")
            except Exception as exc:
                print(f"  FAIL ecriture fiona : {type(exc).__name__}: {exc}")

        try:
            os.unlink(out_path)
        except OSError:
            pass
    except Exception as exc:
        print(f"  FAIL test fonctionnel : {exc}")
        traceback.print_exc()
else:
    print("  Skipped : geopandas ou backend I/O manquant.")


# ---------------------------------------------------------------------
# Resume / recommandations
# ---------------------------------------------------------------------
header("Resume")
problems = []
if not ok_cv2:     problems.append("OpenCV (cv2)")
if not ok_rio:     problems.append("rasterio")
if not (ok_pyogrio or ok_fiona): problems.append("pyogrio ou fiona")
if not ok_gpd:    problems.append("geopandas")

if not problems:
    print("  Tout fonctionne. Le pipeline peut tourner.")
else:
    print(f"  Problemes detectes : {', '.join(problems)}")
    print()
    print("  Solution la plus fiable sur Windows : recreer l'env conda en")
    print("  forcant strict-channel-priority sur conda-forge :")
    print()
    print("    conda deactivate")
    print("    conda env remove -n pfa")
    print("    conda create -n pfa python=3.10 -c conda-forge "
          "--strict-channel-priority -y \\")
    print("        rasterio geopandas shapely fiona pyproj gdal jupyter \\")
    print("        matplotlib ipywidgets scikit-image django pytest")
    print("    conda activate pfa")
    print("    pip install opencv-python django-leaflet django-geojson")
    print("    pip install torch torchvision \\")
    print("        --index-url https://download.pytorch.org/whl/cu121")
