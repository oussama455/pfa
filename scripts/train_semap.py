"""
DEPRECATED — utilise `scripts/train.py --dataset semap` a la place.

Equivalences :
    python scripts/train_semap.py --epochs 20 --batch-size 8
    -> python scripts/train.py --dataset semap --epochs 20 --batch-size 8

    python scripts/train_semap.py --no-synthetic
    -> python scripts/train.py --dataset semap --no-synthetic
"""
import sys, os
from pathlib import Path

print("[DEPRECATED] scripts/train_semap.py -> redirection vers "
       "scripts/train.py --dataset semap")

script = Path(__file__).parent / "train.py"
args = sys.argv[1:]
if "--dataset" not in args:
    args = ["--dataset", "semap"] + args
os.execv(sys.executable, [sys.executable, str(script)] + args)
