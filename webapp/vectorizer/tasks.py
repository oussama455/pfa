"""
Exécution du pipeline de vectorisation en arrière-plan.

Pour la V1, on exécute de façon synchrone dans un thread simple.
Pour la V2 (production), remplacer par Celery + Redis.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from django.conf import settings

from .models import MapUpload

logger = logging.getLogger(__name__)


def _run_pipeline_sync(upload_id: int) -> None:
    """Exécute le pipeline pour une carte donnée."""
    try:
        upload = MapUpload.objects.get(pk=upload_id)
    except MapUpload.DoesNotExist:
        logger.error('Upload %s introuvable', upload_id)
        return

    upload.status = 'processing'
    upload.save(update_fields=['status'])

    try:
        # Import local pour éviter d'alourdir le démarrage Django
        import sys
        sys.path.insert(0, str(Path(settings.PROJECT_ROOT)))
        from pipeline.pipeline import run_pipeline

        input_path = Path(upload.raster.path)
        output_dir = upload.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            with_semantic=False,   # V1 : segmentation couleur seule
            verbose=False,
        )

        upload.status = 'done'
        upload.error_message = ''
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        logger.exception('Pipeline a échoué pour upload %s', upload_id)
        upload.status = 'error'
        upload.error_message = (
            f"{type(exc).__name__}: {exc}\n\n"
            f"--- Traceback complet ---\n{tb}"
        )
    finally:
        upload.save()


def enqueue_pipeline(upload: MapUpload) -> None:
    """
    Lance le pipeline dans un thread détaché.

    V1 uniquement — en production utiliser Celery.
    """
    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(upload.pk,),
        daemon=True,
    )
    thread.start()
