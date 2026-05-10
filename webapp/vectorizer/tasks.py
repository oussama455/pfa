"""
webapp/vectorizer/tasks.py — Pipeline execution (V2 with LangGraph agent).
Backwards-compatible with V1 thread model.
"""
from __future__ import annotations
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
        # Appel direct du pipeline principal avec les mêmes arguments que la commande CLI
        try:
            from pipeline.pipeline import run_pipeline
            result = run_pipeline(
                str(upload.raster_path),
                str(upload.output_dir),
                with_semantic=True,
                unet_weights=r"C:\Users\ochou\Documents\Claude\pfa\pfa.worktrees\dev-ses-pfa\external\weight\semap_unet_best.pth",
                device=None,  # "auto" en CLI correspond à None ici
                verbose=True
            )
            # Marquer comme terminé (adapter selon la sortie de run_pipeline)
            upload.mark_done(
                map_type=getattr(result, "map_type", None),
                confidence_score=getattr(result, "confidence_score", None),
                qa_passed=getattr(result, "qa_passed", False),
                retry_count=getattr(result, "retry_count", 0),
                georef_crs=getattr(result, "georef_crs", None),
                raster_bounds=None,
            )
            logger.info("[tasks] Pipeline done map_id=%d", upload_id)
        except ImportError:
            logger.warning("[tasks] pipeline import failed, fallback to classic pipeline map_id=%d", upload_id)
            _run_classic_pipeline(upload)
    except Exception as exc:
        logger.exception("[tasks] Pipeline failed map_id=%d: %s", upload_id, exc)
        upload.mark_failed(str(exc))
    finally:
        close_old_connections()


def _run_classic_pipeline(upload) -> None:
    import sys, json, cv2
    from django.conf import settings
    project_root = Path(settings.BASE_DIR).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipeline.preprocessing import detect_map_frame
    from pipeline.color_segmentation import extract_all_color_layers

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

    try:
        from pipeline.vectorization import masks_to_geojson
        from pipeline.georeferencing import (
            AMS_ALGERIA_SHEETS, apply_transform_to_geojson, filter_features_by_bbox
        )
        geojsons = masks_to_geojson(layers)
        corners = AMS_ALGERIA_SHEETS.get(upload.map_name) if upload.map_name else None
        for name, gj in geojsons.items():
            if corners:
                gj = filter_features_by_bbox(gj, bbox)
                gj = apply_transform_to_geojson(gj, bbox, corners)
                gj["crs"] = {"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}}
            out = upload.output_dir / f"{name}.geojson"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(gj, f, indent=2)
    except ImportError as e:
        logger.warning("[tasks] vectorization import failed: %s", e)

    upload.mark_done(georef_crs="EPSG:4326" if upload.map_name else None)


def _get_weights_path():
    import os
    from pathlib import Path
    
    # المسار المطلق المباشر (الأكثر أماناً في ويندوز)
    absolute_path = r"C:\Users\ochou\Documents\Claude\pfa\pfa.worktrees\dev-ses-pfa\external\weight\semap_unet_best.pth"
    
    if os.path.exists(absolute_path):
        return absolute_path
    
    # حل احتياطي ديناميكي باستخدام PROJECT_ROOT من ملفك _path_setup
    try:
        from . import _path_setup
        dynamic_path = Path(_path_setup.PROJECT_ROOT) / "external" / "weight" / "semap_unet_best.pth"
        if dynamic_path.is_file():
            return str(dynamic_path)
    except:
        pass
        
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
