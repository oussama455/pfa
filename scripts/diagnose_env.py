"""DEPRECATED -- utilise scripts/diagnose.py --check env"""
import sys, os
from pathlib import Path
print("[DEPRECATED] scripts/diagnose_env.py -> scripts/diagnose.py --check env")
script = Path(__file__).parent / "diagnose.py"
os.execv(sys.executable, [sys.executable, str(script), "--check", "env"])
