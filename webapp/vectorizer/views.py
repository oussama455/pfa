"""Vues Django pour le vectoriseur."""
from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .forms import MapUploadForm
from .models import MapUpload
from .tasks import enqueue_pipeline


def upload_view(request):
    """Page d'accueil : formulaire d'upload + liste des traitements récents."""
    if request.method == 'POST':
        form = MapUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save()
            enqueue_pipeline(upload)
            return redirect(upload.get_absolute_url())
    else:
        form = MapUploadForm()

    recent = MapUpload.objects.all()[:10]
    return render(request, 'vectorizer/upload.html', {
        'form': form,
        'recent': recent,
    })


def result_view(request, pk):
    """Page résultat : affiche la carte Leaflet + couches extraites."""
    upload = get_object_or_404(MapUpload, pk=pk)
    return render(request, 'vectorizer/result.html', {
        'upload': upload,
        'layers': upload.output_layers,
    })


def status_view(request, pk):
    """Endpoint JSON pour polling côté navigateur."""
    upload = get_object_or_404(MapUpload, pk=pk)
    return JsonResponse({
        'status': upload.status,
        'status_label': upload.get_status_display(),
        'error': upload.error_message,
        'layers': upload.output_layers,
    })
