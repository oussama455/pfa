"""
Pipeline de vectorisation de cartes -- PFA Oussama CHOUAIBI.

Chaine de traitement : raster -> pretraitement -> segmentation (couleur + IA)
-> vectorisation -> geo-referencement -> export GeoJSON.

Imports paresseux : les sous-modules ne sont charges que si tu les importes
explicitement, pour eviter d'imposer torch / rasterio / skimage quand
tu n'utilises qu'une partie du pipeline.

Modules principaux : preprocessing, color_segmentation, semantic_segmentation,
vectorization, georeferencing, pipeline, agent, active_learning, dataset,
semap_dataset, map_frame_detector, grid_extraction, cc_postprocess,
create_masks, paths.
"""

__version__ = "0.2.0"
__author__ = "Oussama CHOUAIBI"

__all__ = [
    "preprocessing",
    "map_frame_detector",
    "color_segmentation",
    "semantic_segmentation",
    "grid_extraction",
    "vectorization",
    "cc_postprocess",
    "georeferencing",
    "dataset",
    "semap_dataset",
    "pipeline",
    "agent",
    "active_learning",
    "paths",
    "create_masks",
]
