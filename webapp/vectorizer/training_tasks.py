"""
webapp/vectorizer/training_tasks.py -- lance scripts/train.py dans un thread.

Capture stdout dans un fichier .log, met a jour le TrainingJob.status,
extrait le mIoU final du log pour le stocker en base.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Racine du repo (parent du dossier webapp/)."""
    return Path(settings.BASE_DIR).parent


def _weights_dir() -> Path:
    p = _project_root() / "external" / "weight"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _logs_dir() -> Path:
    p = _project_root() / "external" / "weight" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_available_weights() -> list[dict]:
    """
    Retourne la liste des fichiers .pth dans external/weight/.

    Format : [{name, path, size_mb, dataset}, ...]
    Le 'dataset' est devine d'apres le nom du fichier (soduco/semap/other).
    """
    out = []
    for p in sorted(_weights_dir().glob("*.pth")):
        name = p.name.lower()
        if "soduco" in name:
            ds = "soduco"
        elif "semap" in name:
            ds = "semap"
        elif "mask2former" in name or "iter_138828" in name:
            ds = "mask2former"
        else:
            ds = "other"
        out.append({
            "name":    p.name,
            "path":    str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
            "dataset": ds,
        })
    return out


def enqueue_training(job_id: int) -> None:
    """Lance un thread qui execute scripts/train.py pour le job donne."""
    t = threading.Thread(target=_run_training_thread, args=(job_id,),
                          daemon=True, name=f"training-job-{job_id}")
    t.start()
    logger.info("[training] Thread lance pour job %d", job_id)


def _run_training_thread(job_id: int) -> None:
    from django.db import close_old_connections
    close_old_connections()

    from .models import TrainingJob
    try:
        job = TrainingJob.objects.get(pk=job_id)
    except TrainingJob.DoesNotExist:
        logger.error("[training] job %d introuvable", job_id)
        return

    project_root = _project_root()
    train_script = project_root / "scripts" / "train.py"
    log_file = _logs_dir() / f"training_job_{job_id}.log"

    # Construit la commande shell
    cmd = [
        sys.executable, str(train_script),
        "--dataset",     job.dataset,
        "--epochs",      str(job.epochs),
        "--batch-size",  str(job.batch_size),
        "--target-size", str(job.target_size),
        "--lr",          str(job.learning_rate),
        "--encoder",     job.encoder,
    ]
    if job.no_synthetic:
        cmd.append("--no-synthetic")
    if job.no_augment:
        cmd.append("--no-augment")

    job.log_path = str(log_file)
    job.save(update_fields=["log_path"])

    try:
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(f"[CartoVec training] {' '.join(cmd)}\n\n")
            proc = subprocess.Popen(cmd,
                                     stdout=fh, stderr=subprocess.STDOUT,
                                     cwd=str(project_root),
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
            job.mark_running(pid=proc.pid)
            ret = proc.wait()

        # Analyse du log pour extraire le best mIoU et le chemin du checkpoint
        log_txt = log_file.read_text(encoding="utf-8", errors="replace")
        best_miou = None
        for m in re.finditer(r"Best mIoU\s*=\s*([0-9.]+)", log_txt):
            try:
                best_miou = float(m.group(1))
            except ValueError:
                pass

        # Chemin du best.pth attendu (cf. scripts/train.py)
        weights_path = _weights_dir() / f"{job.dataset}_unet_best.pth"

        if ret == 0 and weights_path.exists():
            job.mark_done(weights_path=str(weights_path), best_miou=best_miou)
            logger.info("[training] job %d OK -- mIoU=%s -- %s",
                        job_id, best_miou, weights_path)
        else:
            tail = "\n".join(log_txt.splitlines()[-30:])
            job.mark_failed(f"Return code={ret}.\n\n--- Dernieres lignes ---\n{tail}")
            logger.error("[training] job %d KO -- code %d", job_id, ret)
    except Exception as exc:
        import traceback
        job.mark_failed(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        logger.exception("[training] job %d -- exception", job_id)
    finally:
        close_old_connections()
