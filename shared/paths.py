"""
DEPRECATED -- utilise `from pipeline.paths import Paths` a la place.

Ce shim reste pour compatibilite avec d'anciens scripts qui font
`from shared.paths import Paths`. Il sera supprime au prochain
nettoyage du repo (`git rm -r shared/`).
"""
from pipeline.paths import *  # noqa: F401,F403
from pipeline.paths import Paths, configure_gdal_windows  # noqa: F401
