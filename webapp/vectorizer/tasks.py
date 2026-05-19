"""
webapp/vectorizer/tasks.py — Pipeline execution (V2 with LangGraph agent).
Backwards-compatible with V1 thread model.
"""
from __future__ import annotations

# ── Sécurités CUDA — placées avant tout import torch (voir pipeline.py) ─────
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import logging
import sys
import threading
from pathlib import Path

# Ceinture-bretelles : settings.py ajoute deja PROJECT_ROOT au sys.path,
# mais on le re-fait ici pour que tasks.py marche aussi quand il est appele
# hors contexte Django (test unitaire, REPL, manage.py shell).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def enqueue_pipeline(upload) -> None:
    upload.mark_processing()
    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(upload.pk,),
        daemon=True,
        name=f"pipeline-map-{upload.pk}",
    )
    t.start()
    logger.info("[tasks] Pipeline thread started for map_id=%d", upload.pk)


def _run_pipeline_thread(upload_id: int) -> None:
    from django.db import close_old_connections
    close_old_connections()

    from .models import MapUpload
    try:
        upload = MapUpload.objects.get(pk=upload_id)
    except MapUpload.DoesNotExist:
        logger.error("[tasks] MapUpload %d not found", upload_id)
        return

    try:
        # Resout les poids U-Net : choix explicite via upload.unet_weights, sinon
        # auto-detection dans external/weight/.
        weights = _get_weights_path(upload=upload)
        use_semantic = weights is not None

        # Le flag has_georeference est positionné par l'API au moment du POST.
        # Par défaut False → pipeline en coords pixel image.
        georeference = bool(getattr(upload, "has_georeference", False))

        try:
            from pipeline.pipeline import run_pipeline
            result = run_pipeline(
                str(upload.raster_path),
                str(upload.output_dir),
                georeference=georeference,
                with_semantic=use_semantic,
                unet_weights=weights,
                device=None,  # "auto" en CLI correspond à None ici
                verbose=True
            )
            # Le pipeline peut avoir dégradé en pixel si le module SIG manque.
            effective_georef = bool(getattr(result, "has_georeference",
                                              getattr(result, "georeferenced", False)))
            # Marquer comme terminé (adapter selon la sortie de run_pipeline)
            upload.mark_done(
                map_type=getattr(result, "map_type", None),
                confidence_score=getattr(result, "confidence_score", None),
                qa_passed=getattr(result, "qa_passed", False),
                retry_count=getattr(result, "retry_count", 0),
                georef_crs=getattr(result, "georef_crs", None),
                raster_bounds=None,
                has_georeference=effective_georef,
            )
            logger.info("[tasks] Pipeline done map_id=%d georef=%s",
                        upload_id, effective_georef)
        except ImportError:
            logger.warning("[tasks] pipeline import failed, fallback to classic pipeline map_id=%d", upload_id)
            _run_classic_pipeline(upload)
    except Exception as exc:
        logger.exception("[tasks] Pipeline failed map_id=%d: %s", upload_id, exc)
        upload.mark_failed(str(exc))
    finally:
        close_old_connections()


def _run_classic_pipeline(upload) -> None:
    """
    Fallback pipeline minimal (sans agent LangGraph).
    Respecte upload.has_georeference : si False, écrit du GeoJSON pixel direct.
    """
    import sys, json, cv2
    from django.conf import settings
    project_root = Path(settings.BASE_DIR).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipeline.preprocessing import detect_map_frame
    from pipeline.color_segmentation import extract_all_color_layers

    georeference = bool(getattr(upload, "has_georeference", False))

    img = cv2.imread(str(upload.raster_path))
    if img is None:
        raise ValueError(f"Cannot read: {upload.raster_path}")

    scale = 2400 / max(img.shape[:2])
    img_small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                            interpolation=cv2.INTER_AREA)
    bbox = detect_map_frame(img_small)
    x1, y1, x2, y2 = bbox
    img_crop = img_small[y1:y2, x1:x2]
    layers = extract_all_color_layers(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV))

    effective_georef = False
    try:
        from pipeline.vectorization import masks_to_geojson
        # Toujours produire d'abord en pixels — c'est l'invariant.
        geojsons = masks_to_geojson(layers, georeference=False)

        # Géoréférencement optionnel
        if georeference:
            try:
                from pipeline.georeferencing import (
                    AMS_ALGERIA_SHEETS, apply_transform_to_geojson,
                    filter_features_by_bbox,
                )
                corners = (AMS_ALGERIA_SHEETS.get(upload.map_name)
                           if upload.map_name else None)
                if corners:
                    for name, gj in geojsons.items():
                        gj = filter_features_by_bbox(gj, bbox)
                        gj = apply_transform_to_geojson(gj, bbox, corners)
                        gj["crs"] = {"type": "name",
                                     "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
                        geojsons[name] = gj
                    effective_georef = True
                else:
                    logger.info("[tasks] map_name '%s' absent du registre AMS — pixel kept",
                                upload.map_name)
            except (ImportError, OSError) as e:
                logger.warning("[tasks] georeferencing unavailable, staying in pixel mode: %s", e)

        for name, gj in geojsons.items():
            out = upload.output_dir / f"{name}.geojson"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(gj, f, indent=2)
    except ImportError as e:
        logger.warning("[tasks] vectorization import failed: %s", e)

    upload.mark_done(
        georef_crs="EPSG:4326" if effective_georef else None,
        has_georeference=effective_georef,
    )


def _get_weights_path(upload=None):
    """
    Resolution du chemin .pth a utiliser pour l'inference U-Net.

    Règle simplifiée pour la démo PFA :
        1. Si l'utilisateur choisit un fichier .pth dans l'interface, on l'utilise.
        2. Sinon, None -> segmentation HSV seule.

    Important : on ne choisit plus automatiquement un .pth présent sur disque.
    Cela évite de lancer U-Net par surprise sur RTX 2050 pendant une démo.
    """
    from pathlib import Path

    # 1. Choix explicite par l'utilisateur via l'API
    if upload is not None and getattr(upload, "unet_weights", None):
        chosen = Path(upload.unet_weights)
        if chosen.is_file():
            return str(chosen)
        logger.warning("[tasks] unet_weights '%s' introuvable, passage en HSV seul",
                       upload.unet_weights)
    return None

def _extract_bounds(agent_result):
    import json
    for path_str in agent_result.get("output_geojsons", {}).values():
        try:
            with open(path_str) as f:
                gj = json.load(f)
            all_lons, all_lats = [], []
            for feat in gj.get("features", []):
                coords = feat.get("geometry", {}).get("coordinates", [])
                flat = coords
                while flat and isinstance(flat[0], list):
                    flat = [c for ring in flat for c in ring]
                for c in flat:
                    if len(c) >= 2:
                        all_lons.append(c[0]); all_lats.append(c[1])
            if all_lons:
                return [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
        except Exception:
            continue
    return None
