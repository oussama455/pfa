"""
Helper de configuration pour Google Colab.

Usage dans un notebook (premiere cellule) :

    !wget -q -O colab_setup.py https://raw.githubusercontent.com/<user>/pfa/main/notebooks/colab_setup.py
    from colab_setup import setup
    setup()

Ou simplement copier-coller le contenu de cette fonction dans la premiere cellule.

Sans rien faire en local (Anaconda), le module detecte qu'il n'est pas dans
Colab et ajuste juste le sys.path pour importer pipeline/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_PIP_COLAB = [
    # OpenCV est deja la dans Colab mais on force la version
    # qui marche bien avec la pipeline
    "opencv-python==4.10.0.84",
    "scikit-image>=0.22",
    # Pour I/O GeoJSON. fiona en backup au cas ou pyogrio plante
    "rasterio>=1.3",
    "shapely>=2.0",
    "geopandas>=0.14",
    "pyogrio",
    "fiona",
    # IA
    "segmentation-models-pytorch>=0.3.3",
    "albumentations>=1.4",
    # Notebook
    "ipywidgets>=8.1",
    "tqdm>=4.66",
]


def in_colab() -> bool:
    """True si on tourne dans Google Colab."""
    return "google.colab" in sys.modules


def pip_install(packages, quiet: bool = True) -> None:
    """pip install paresseux (skip ce qui est deja la)."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def install_dependencies(quiet: bool = True) -> None:
    """Installe les paquets necessaires sur Colab."""
    if not in_colab():
        print("  (pas dans Colab — skip pip install)")
        return
    print(f"  Installation des dependances ({len(REQUIRED_PIP_COLAB)} paquets)...")
    pip_install(REQUIRED_PIP_COLAB, quiet=quiet)
    print("  OK")


def clone_repo(repo_url: str, target_dir: str = "pfa", branch: str = "main") -> Path:
    """
    Clone le projet PFA si pas deja la. Retourne le chemin du repo.

    Sur Colab : /content/pfa/.
    En local  : pas d'effet (renvoie ce dossier '..').
    """
    if not in_colab():
        # Local : on suppose que le notebook est dans pfa/notebooks/
        # donc le repo est ".." par rapport au notebook
        return Path("..").resolve()

    target = Path("/content") / target_dir
    if target.exists() and (target / "pipeline").exists():
        print(f"  Repo deja clone : {target}")
        return target
    print(f"  Clonage de {repo_url} -> {target}")
    cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(target)]
    subprocess.check_call(cmd)
    return target


def mount_drive(mount_point: str = "/content/drive") -> Path:
    """
    Monte Google Drive pour acceder a tes cartes raster (option recommandee
    si tu ne veux pas pousser sur GitHub). En local : sans effet.

    Retourne le chemin /content/drive/MyDrive.
    """
    if not in_colab():
        print("  (pas dans Colab — skip mount Drive)")
        return Path.home()
    from google.colab import drive  # type: ignore
    if not Path(mount_point).exists() or not (Path(mount_point) / "MyDrive").exists():
        drive.mount(mount_point)
    return Path(mount_point) / "MyDrive"


def setup(repo_url: str | None = None, branch: str = "main",
          install: bool = True, use_drive: bool = False) -> Path:
    """
    Setup complet pour Colab : install + clone + ajuste sys.path.

    Arguments :
        repo_url  : URL du repo GitHub a cloner (necessaire sur Colab si
                    pas use_drive). Exemple :
                    'https://github.com/Mohamed-GHARBI/pfa.git'
        branch    : branche du repo (defaut 'main').
        install   : pip install les deps si True.
        use_drive : si True, monte Google Drive et NE clone PAS — le projet
                    est suppose etre dans /content/drive/MyDrive/pfa.

    Retourne le chemin racine du projet (a utiliser comme cwd).
    """
    print("=" * 60)
    print("  SETUP NOTEBOOK")
    print("=" * 60)

    if install:
        install_dependencies()

    if in_colab():
        if use_drive:
            drive_root = mount_drive()
            project_root = drive_root / "pfa"
            if not (project_root / "pipeline").exists():
                raise FileNotFoundError(
                    f"{project_root} n'existe pas ou ne contient pas pipeline/. "
                    "Copie le projet sur Drive d'abord."
                )
        else:
            if repo_url is None:
                raise ValueError(
                    "repo_url requis si use_drive=False sur Colab. "
                    "Exemple : 'https://github.com/<user>/pfa.git'"
                )
            project_root = clone_repo(repo_url, branch=branch)
    else:
        project_root = Path("..").resolve()

    sys.path.insert(0, str(project_root))
    print(f"\n  Project root : {project_root}")
    print(f"  sys.path[0]  : {sys.path[0]}")

    # Verification rapide
    try:
        from pipeline import preprocessing  # noqa: F401
        print("  Import pipeline.preprocessing : OK")
    except ImportError as exc:
        print(f"  WARN : import pipeline a echoue ({exc})")

    print("=" * 60)
    return project_root
