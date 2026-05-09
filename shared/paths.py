"""
shared.paths — alias retro-compat vers pipeline.paths.

Le module canonique est `pipeline.paths`. Ce shim permet a tout code
qui fait `from shared.paths import Paths` de continuer a fonctionner.
"""
from pipeline.paths import *  # noqa: F401,F403
from pipeline.paths import Paths, configure_gdal_windows  # noqa: F401
