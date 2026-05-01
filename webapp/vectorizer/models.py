"""
Modèles Django pour le vectoriseur.

Pour la V1, un seul modèle : MapUpload, qui représente une carte
téléversée et les résultats de son traitement.
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.urls import reverse


class MapUpload(models.Model):
    """Représente une carte téléversée et son traitement."""

    STATUS_CHOICES = [
        ('pending',   'En attente'),
        ('processing', 'En cours'),
        ('done',      'Terminé'),
        ('error',     'Erreur'),
    ]

    title = models.CharField('Titre', max_length=200)
    raster = models.ImageField('Carte raster', upload_to='uploads/')
    status = models.CharField('Statut', max_length=20,
                               choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField('Erreur', blank=True)
    created_at = models.DateTimeField('Créé le', auto_now_add=True)
    updated_at = models.DateTimeField('Mis à jour le', auto_now=True)

    class Meta:
        ordering = ('-created_at',)
        verbose_name = 'Carte téléversée'
        verbose_name_plural = 'Cartes téléversées'

    def __str__(self) -> str:
        return f'{self.title} [{self.get_status_display()}]'

    def get_absolute_url(self):
        return reverse('vectorizer:result', kwargs={'pk': self.pk})

    @property
    def output_dir(self) -> Path:
        """Dossier où le pipeline écrit les GeoJSON pour cette carte."""
        return Path(settings.PROCESSED_DIR) / f'map_{self.pk}'

    @property
    def output_layers(self) -> dict:
        """Liste les GeoJSON produits (nom → URL)."""
        if not self.output_dir.exists():
            return {}
        layers = {}
        for geojson in self.output_dir.glob('*.geojson'):
            rel = geojson.relative_to(settings.MEDIA_ROOT)
            layers[geojson.stem] = settings.MEDIA_URL + str(rel).replace('\\', '/')
        return layers
