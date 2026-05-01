"""Formulaires Django pour le vectoriseur."""
from django import forms

from .models import MapUpload


class MapUploadForm(forms.ModelForm):
    """Formulaire d'upload d'une carte."""

    class Meta:
        model = MapUpload
        fields = ['title', 'raster']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Ex. Carte topo Bizerte 1/50000',
            }),
            'raster': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'image/png,image/jpeg,image/tiff',
            }),
        }
        labels = {
            'title': 'Nom de la carte',
            'raster': 'Fichier raster (PNG, JPG, TIFF)',
        }
