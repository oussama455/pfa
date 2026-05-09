"""
Helper : ajoute le PROJECT_ROOT au sys.path pour que `from pipeline...`
fonctionne depuis n'importe quel module webapp/.

Usage (en haut de api.py, tasks.py, etc.) :
    from . import _path_setup  # noqa: F401

settings.py fait deja le meme job au demarrage Django, mais ce module
fournit une assurance contre les contextes d'import alternatifs (tests,
shell, scripts qui importent webapp.* directement).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROJECT_ROOT = _PROJECT_ROOT
