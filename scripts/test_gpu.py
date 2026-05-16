"""DEPRECATED -- utilise scripts/diagnose.py --check gpu"""
import sys, os
from pathlib import Path
print("[DEPRECATED] scripts/test_gpu.py -> scripts/diagnose.py --check gpu")
script = Path(__file__).parent / "diagnose.py"
os.execv(sys.executable, [sys.executable, str(script), "--check", "gpu"])
