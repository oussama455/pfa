"""
Pipeline de vectorisation de cartes — PFA Mohamed GHARBI.

Chaîne de traitement :
    raster → prétraitement → segmentation (couleur + IA)
           → vectorisation → géoréférencement → export GeoJSON
"""

__version__ = "0.1.0"
__author__ = "Mohamed GHARBI"

# Imports paresseux : on ne charge PAS les sous-modules au import du package.
# Cela permet d'utiliser, par exemple, `from pipeline import preprocessing`
# sans avoir besoin que skimage / rasterio / torch soient installés
# pour les modules qu'on n'utilise pas.

__all__ = [
    "preprocessing",
    "color_segmentation",
    "grid_extraction",
    "vectorization",
    "georeferencing",
    "pipeline",
    "semantic_segmentation",
]
