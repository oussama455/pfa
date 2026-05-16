"""
webapp/vectorizer/api_training.py -- API REST pour training + selection des poids.

Endpoints :
    GET    /api/weights/                  liste les .pth dans external/weight/
    GET    /api/training/                 liste les TrainingJob
    POST   /api/training/                 cree + demarre un TrainingJob
    GET    /api/training/<pk>/            detail d'un job (statut, best_miou, log_path)
    GET    /api/training/<pk>/log/        contenu du log (texte brut)
    GET    /api/training/<pk>/download/   telecharge le .pth produit
"""
from __future__ import annotations

from pathlib import Path

from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import TrainingJob
from .training_tasks import enqueue_training, list_available_weights


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────────────
# Serializers
# ─────────────────────────────────────────────────────────────────────────────
class TrainingJobSerializer(serializers.ModelSerializer):
    download_url = serializers.SerializerMethodField()
    log_url      = serializers.SerializerMethodField()

    class Meta:
        model = TrainingJob
        fields = [
            "id", "dataset", "epochs", "batch_size", "target_size",
            "learning_rate", "encoder", "no_synthetic", "no_augment",
            "status", "pid", "best_miou", "error_message",
            "log_path", "output_weights_path",
            "created_at", "started_at", "finished_at",
            "download_url", "log_url",
        ]
        read_only_fields = ["status", "pid", "best_miou", "error_message",
                             "log_path", "output_weights_path",
                             "created_at", "started_at", "finished_at",
                             "download_url", "log_url"]

    def get_download_url(self, obj):
        if not obj.output_weights_path:
            return None
        if not Path(obj.output_weights_path).exists():
            return None
        return f"/api/training/{obj.pk}/download/"

    def get_log_url(self, obj):
        if obj.log_path and Path(obj.log_path).exists():
            return f"/api/training/{obj.pk}/log/"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# /api/weights/  -- liste tous les .pth disponibles dans external/weight/
# ─────────────────────────────────────────────────────────────────────────────
class WeightsListView(APIView):
    """
    GET /api/weights/
    Liste tous les fichiers .pth dans external/weight/.
    Permet au frontend de proposer un dropdown "choisir les poids" pour l'upload.
    """
    def get(self, request):
        weights = list_available_weights()
        return Response({"weights": weights, "count": len(weights)})


# ─────────────────────────────────────────────────────────────────────────────
# /api/training/
# ─────────────────────────────────────────────────────────────────────────────
class TrainingJobListCreateView(APIView):
    """
    GET  /api/training/   liste les TrainingJob (plus recent en premier).
    POST /api/training/   cree un job + lance le thread.
    """
    def get(self, request):
        jobs = TrainingJob.objects.all()
        return Response(TrainingJobSerializer(jobs, many=True).data)

    def post(self, request):
        # Validation des champs : dataset obligatoire
        ds = request.data.get("dataset", "semap")
        if ds not in ("soduco", "semap"):
            return Response({"error": "dataset must be 'soduco' or 'semap'"},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            job = TrainingJob.objects.create(
                dataset=ds,
                epochs=int(request.data.get("epochs", 20)),
                batch_size=int(request.data.get("batch_size", 8)),
                target_size=int(request.data.get("target_size", 512)),
                learning_rate=float(request.data.get("learning_rate", 1e-4)),
                encoder=request.data.get("encoder", "resnet34"),
                no_synthetic=_as_bool(request.data.get("no_synthetic", False)),
                no_augment=_as_bool(request.data.get("no_augment", False)),
                status="queued",
            )
        except (ValueError, TypeError) as exc:
            return Response({"error": f"Invalid parameters: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)

        enqueue_training(job.pk)
        return Response(TrainingJobSerializer(job).data,
                        status=status.HTTP_201_CREATED)


# ─────────────────────────────────────────────────────────────────────────────
# /api/training/<pk>/
# ─────────────────────────────────────────────────────────────────────────────
class TrainingJobDetailView(APIView):
    def get(self, request, pk):
        job = get_object_or_404(TrainingJob, pk=pk)
        return Response(TrainingJobSerializer(job).data)


class TrainingJobLogView(APIView):
    """GET /api/training/<pk>/log/  -- contenu brut du log (texte)."""
    def get(self, request, pk):
        job = get_object_or_404(TrainingJob, pk=pk)
        if not job.log_path:
            return HttpResponse("(log non encore disponible)", content_type="text/plain")
        p = Path(job.log_path)
        if not p.exists():
            return HttpResponse(f"(log introuvable : {p})", content_type="text/plain")
        # Limite a 200 KB pour pas saturer le frontend
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > 200_000:
            text = text[:50_000] + "\n\n[... tronque ...]\n\n" + text[-150_000:]
        return HttpResponse(text, content_type="text/plain")


class TrainingJobDownloadView(APIView):
    """GET /api/training/<pk>/download/  -- envoie le .pth produit."""
    def get(self, request, pk):
        job = get_object_or_404(TrainingJob, pk=pk)
        if not job.output_weights_path:
            raise Http404("Aucun fichier de poids produit")
        p = Path(job.output_weights_path)
        if not p.exists():
            raise Http404(f"Poids introuvables : {p}")
        return FileResponse(open(p, "rb"),
                             as_attachment=True,
                             filename=p.name,
                             content_type="application/octet-stream")
