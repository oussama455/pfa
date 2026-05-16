"""
DEPRECATED — utilise `scripts/train.py --dataset soduco` a la place.

Ce wrapper redirige automatiquement vers le script unifie pour
compatibilite retro avec d'eventuels scripts shell ou documentation.

Equivalence des options :
    python scripts/train_mapseg.py --epochs 20
    -> python scripts/train.py --dataset soduco --epochs 20
"""
import sys, os
from pathlib import Path

print("[DEPRECATED] scripts/train_mapseg.py -> redirection vers "
       "scripts/train.py --dataset soduco")

# Reconstruit la commande en injectant --dataset soduco
script = Path(__file__).parent / "train.py"
args = sys.argv[1:]
if "--dataset" not in args:
    args = ["--dataset", "soduco"] + args
os.execv(sys.executable, [sys.executable, str(script)] + args)
