"""
webapp/vectorizer/models.py — CartoVec database models (V2).

Backwards-compatible with V1 MapUpload. Added:
    - map_name, map_type, confidence_score, qa_passed, retry_count
    - georef_crs, raster_bounds
    - mark_processing / mark_done / mark_failed helpers
    - Correction model (HITL)
    - TrainingPatch model (retraining pipeline)
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone


class MapUpload(models.Model):

    STATUS_CHOICES = [
        ('pending',    'En attente'),
        ('processing', 'En cours'),
        ('done',       'Terminé'),
        ('error',      'Erreur'),    # legacy alias
        ('failed',     'Échec'),
    ]

    # ── Core ─────────────────────────────────────────────────────────────────
    title         = models.CharField('Titre', max_length=200)
    raster        = models.ImageField('Carte raster', upload_to='uploads/')
    map_name      = models.CharField(
        max_length=100, blank=True, null=True,
        help_text="Sheet name for auto GCP lookup (e.g. 'tunis', 'ain_bessem')"
    )

    # ── Pipeline state ────────────────────────────────────────────────────────
    status        = models.CharField('Statut', max_length=20,
                                     choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField('Erreur', blank=True, null=True)
    map_type      = models.CharField(max_length=30, blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    qa_passed     = models.BooleanField(default=False)
    retry_count   = models.PositiveSmallIntegerField(default=0)
    # Chemin du fichier .pth a utiliser pour la segmentation U-Net.
    # None = pas d'U-Net (segmentation couleur seulement).
    unet_weights  = models.CharField(
        'Poids U-Net', max_length=500, blank=True, null=True,
        help_text="Chemin vers le .pth a utiliser. Vide = pas d'U-Net."
    )

    # ── Geo ───────────────────────────────────────────────────────────────────
    georef_crs    = models.CharField(max_length=50, blank=True, null=True)
    raster_bounds = models.JSONField(blank=True, null=True,
        help_text="[[lat_sw, lon_sw], [lat_ne, lon_ne]] for Leaflet ImageOverlay")

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at    = models.DateTimeField('Créé le',       auto_now_add=True)
    updated_at    = models.DateTimeField('Mis à jour le', auto_now=True)
    finished_at   = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ('-created_at',)
        verbose_name = 'Carte téléversée'
        verbose_name_plural = 'Cartes téléversées'

    def __str__(self) -> str:
        return f'{self.title} [{self.get_status_display()}]'

    def get_absolute_url(self):
        return reverse('vectorizer:result', kwargs={'pk': self.pk})

    # ── Computed paths ────────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """Dossier où le pipeline écrit les GeoJSON pour cette carte."""
        try:
            base = Path(settings.PROCESSED_DIR) / f'map_{self.pk}'
        except AttributeError:
            base = Path(settings.MEDIA_ROOT) / 'processed' / f'map_{self.pk}'
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def raster_path(self) -> Path:
        return Path(settings.MEDIA_ROOT) / str(self.raster)

    @property
    def output_layers(self) -> dict:
        """Liste les GeoJSON produits (nom → URL médias)."""
        if not self.output_dir.exists():
            return {}
        layers = {}
        for geojson in sorted(self.output_dir.glob('*.geojson')):
            try:
                rel = geojson.relative_to(Path(settings.MEDIA_ROOT))
                layers[geojson.stem] = settings.MEDIA_URL + str(rel).replace('\\', '/')
            except ValueError:
                layers[geojson.stem] = str(geojson)
        return layers

    # ── Status helpers ────────────────────────────────────────────────────────

    def mark_processing(self):
        self.status = 'processing'
        self.save(update_fields=['status', 'updated_at'])

    def mark_done(self, *, map_type=None, confidence_score=None,
                  qa_passed=False, retry_count=0,
                  georef_crs=None, raster_bounds=None):
        self.status        = 'done'
        self.finished_at   = timezone.now()
        self.error_message = None
        if map_type          is not None: self.map_type          = map_type
        if confidence_score  is not None: self.confidence_score  = confidence_score
        if qa_passed         is not None: self.qa_passed         = qa_passed
        if retry_count       is not None: self.retry_count       = retry_count
        if georef_crs        is not None: self.georef_crs        = georef_crs
        if raster_bounds     is not None: self.raster_bounds     = raster_bounds
        self.save(update_fields=[
            'status', 'finished_at', 'error_message',
            'map_type', 'confidence_score', 'qa_passed',
            'retry_count', 'georef_crs', 'raster_bounds', 'updated_at',
        ])

    def mark_failed(self, error: str):
        self.status        = 'failed'
        self.error_message = str(error)[:2000]
        self.finished_at   = timezone.now()
        self.save(update_fields=['status', 'error_message', 'finished_at', 'updated_at'])


class Correction(models.Model):
    """
    One HITL correction made by a human operator in MapViewer.

    type=delete → false positive removed
    type=edit   → boundary corrected (geometry stored)

    Append-only: corrections are never updated (military audit trail).
    """
    TYPE_CHOICES = [
        ('delete', 'False Positive Removed'),
        ('edit',   'Boundary Corrected'),
    ]

    map_upload      = models.ForeignKey(
        MapUpload, on_delete=models.CASCADE, related_name='corrections'
    )
    correction_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    layer_name      = models.CharField(max_length=100)
    feature_id      = models.CharField(max_length=200)
    geometry        = models.JSONField(blank=True, null=True,
        help_text="New GeoJSON geometry for 'edit'; null for 'delete'")
    created_at      = models.DateTimeField(auto_now_add=True)
    operator        = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='map_corrections'
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'HITL Correction'
        verbose_name_plural = 'HITL Corrections'
        indexes = [
            models.Index(fields=['map_upload', 'layer_name']),
        ]

    def __str__(self):
        return (f'[{self.correction_type.upper()}] '
                f'map={self.map_upload_id} layer={self.layer_name} feat={self.feature_id}')


class TrainingPatch(models.Model):
    """
    512×512 tile extracted from a processed map for U-Net retraining.
    Human corrections update the associated mask → closes the HITL loop.
    """
    STATUS_CHOICES = [
        ('raw',       'Extrait'),
        ('corrected', 'Corrigé'),
        ('validated', 'Validé'),
        ('rejected',  'Rejeté'),
    ]

    map_upload   = models.ForeignKey(
        MapUpload, on_delete=models.CASCADE, related_name='training_patches'
    )
    patch_image  = models.ImageField(upload_to='patches/images/')
    mask_image   = models.ImageField(upload_to='patches/masks/', blank=True, null=True)
    tile_row     = models.PositiveIntegerField()
    tile_col     = models.PositiveIntegerField()
    crop_x       = models.PositiveIntegerField()
    crop_y       = models.PositiveIntegerField()
    layer_name   = models.CharField(max_length=100)
    patch_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='raw')
    correction_count = models.PositiveSmallIntegerField(default=0)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('map_upload', 'layer_name', 'tile_row', 'tile_col')]
        ordering = ['map_upload', 'layer_name', 'tile_row', 'tile_col']
        verbose_name = 'Training Patch'
        verbose_name_plural = 'Training Patches'

    def __str__(self):
        return (f'Patch [{self.tile_row},{self.tile_col}] '
                f'map={self.map_upload_id} layer={self.layer_name}')


# ─────────────────────────────────────────────────────────────────────────────
# TrainingJob -- represente un entrainement U-Net lance depuis la webapp
# ─────────────────────────────────────────────────────────────────────────────
class TrainingJob(models.Model):
    """
    Entrainement U-Net lance par scripts/train.py via la webapp.

    Le thread de fond appelle `python scripts/train.py --dataset <X>` et
    capture stdout dans log_path. Quand le run reussit, output_weights_path
    pointe vers external/weight/<dataset>_unet_best.pth.
    """

    DATASET_CHOICES = [
        ('soduco', 'SODUCO Benchmark (5 classes)'),
        ('semap',  'SEMAP Petitpierre (6 classes)'),
    ]
    STATUS_CHOICES = [
        ('queued',  'En file d\'attente'),
        ('running', 'En cours'),
        ('done',    'Terminé'),
        ('failed',  'Échec'),
        ('aborted', 'Interrompu'),
    ]

    dataset      = models.CharField(max_length=20, choices=DATASET_CHOICES,
                                     default='semap')
    epochs       = models.PositiveSmallIntegerField(default=20)
    batch_size   = models.PositiveSmallIntegerField(default=8)
    target_size  = models.PositiveSmallIntegerField(default=512)
    learning_rate = models.FloatField(default=1e-4)
    encoder      = models.CharField(max_length=32, default='resnet34')
    no_synthetic = models.BooleanField(default=False,
                    help_text="SEMAP only : exclut les 12 122 images synthétiques.")
    no_augment   = models.BooleanField(default=False)

    status       = models.CharField(max_length=20, choices=STATUS_CHOICES,
                                     default='queued')
    pid          = models.IntegerField(blank=True, null=True)
    log_path     = models.CharField(max_length=500, blank=True, null=True)
    output_weights_path = models.CharField(max_length=500, blank=True, null=True)
    best_miou    = models.FloatField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)

    created_at   = models.DateTimeField(auto_now_add=True)
    started_at   = models.DateTimeField(blank=True, null=True)
    finished_at  = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Training Job'
        verbose_name_plural = 'Training Jobs'

    def __str__(self):
        return f'TrainingJob#{self.pk} [{self.dataset} {self.status}]'

    def mark_running(self, pid=None):
        self.status = 'running'
        self.pid = pid
        self.started_at = timezone.now()
        self.save(update_fields=['status', 'pid', 'started_at'])

    def mark_done(self, weights_path: str | None = None, best_miou: float | None = None):
        self.status = 'done'
        self.finished_at = timezone.now()
        if weights_path:
            self.output_weights_path = str(weights_path)
        if best_miou is not None:
            self.best_miou = best_miou
        self.save()

    def mark_failed(self, message: str):
        self.status = 'failed'
        self.finished_at = timezone.now()
        self.error_message = message[:5000]
        self.save()

    @property
    def output_weights_url(self) -> str | None:
        """URL Django pour telecharger le .pth s'il existe."""
        if not self.output_weights_path:
            return None
        p = Path(self.output_weights_path)
        if not p.exists():
            return None
        # Doit etre servi via une vue dediee (cf. api_training.py)
        return reverse('vectorizer:api-training-download',
                       kwargs={'pk': self.pk})
