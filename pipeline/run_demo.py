"""
Point d'entree tres court pour la demo PFA.

Commande :
    python -m pipeline.run_demo data/raw/carte.png
"""
from __future__ import annotations

import argparse

from .simple_pipeline import run_simple_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo CartoVec simplifiee")
    parser.add_argument("input", help="Carte raster scannee")
    parser.add_argument("-o", "--output", default="data/processed/demo")
    args = parser.parse_args()

    run_simple_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
