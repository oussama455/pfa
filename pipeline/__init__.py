"""
CartoVec pipeline.

Lecture simple pour soutenance :
    1. preprocessing.py       -> recadrer la carte
    2. color_segmentation.py  -> extraire les couleurs cartographiques
    3. vectorization.py       -> produire des couches GeoJSON

Point d'entree recommande :
    from pipeline.simple_pipeline import run_simple_pipeline

Les autres modules restent disponibles, mais sont avances :
    - semantic_segmentation.py : U-Net
    - georeferencing.py       : GCP et projection
    - agent.py                : LangGraph
    - active_learning.py      : corrections HSV progressives
"""

__version__ = "0.2.0"
__author__ = "Oussama CHOUAIBI"

__all__ = [
    "run_simple_pipeline",
    "simple_pipeline",
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


def __getattr__(name):
    """Import paresseux : charge le pipeline simple seulement si demande."""
    if name == "run_simple_pipeline":
        from .simple_pipeline import run_simple_pipeline
        return run_simple_pipeline
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
