"""
Exécution du pipeline de vectorisation en arrière-plan.
Mise à jour : Activation du GPU (CUDA) et de la segmentation sémantique.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from django.conf import settings
from .models import MapUpload

logger = logging.getLogger(__name__)

def _run_pipeline_sync(upload_id: int) -> None:
    """Exécute le pipeline complet (Couleur + IA sur GPU)."""
    try:
        upload = MapUpload.objects.get(pk=upload_id)
    except MapUpload.DoesNotExist:
        logger.error('Upload %s introuvable', upload_id)
        return

    upload.status = 'processing'
    upload.save(update_fields=['status'])

    try:
        import sys
        sys.path.insert(0, str(Path(settings.PROJECT_ROOT)))
        from pipeline.pipeline import run_pipeline

        input_path = Path(upload.raster.path)
        output_dir = upload.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- MODIFICATIONS V2 : GPU & SEMANTIC ---
        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            with_semantic=True,           # Active l'IA U-Net
            device="cuda",                # Force l'usage du GPU NVIDIA
            unet_weights=None,            # Utilise les poids par défaut si présents
            verbose=True,                 # True pour voir CUDA dans les logs Django
            # gcps=upload.get_gcps(),     # Optionnel : si vous avez une méthode pour les coins
        )

        upload.status = 'done'
        upload.error_message = ''
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.exception('Le pipeline GPU a échoué pour l\'upload %s', upload_id)
        upload.status = 'error'
        upload.error_message = f"{type(exc).__name__}: {exc}\n\n{tb}"
    finally:
        upload.save()

def enqueue_pipeline(upload: MapUpload) -> None:
    """Lance le pipeline dans un thread détaché (V1/V2 hybride)."""
    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(upload.pk,),
        daemon=True,
    )
    thread.start()