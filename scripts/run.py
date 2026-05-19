"""
scripts/run.py
==============

Script CLI minimaliste pour lancer le pipeline simple.

Usage :
    python scripts/run.py data/raw/carte.png
    python scripts/run.py data/raw/carte.png -o data/processed/ma_sortie
    python scripts/run.py data/raw/carte.png --keep-legend
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajoute le dossier parent au path pour `from pipeline import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.simple_pipeline import main

if __name__ == "__main__":
    main()
