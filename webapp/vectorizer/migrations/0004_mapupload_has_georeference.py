"""
Migration 0004 — ajoute MapUpload.has_georeference (BooleanField, default=False).

Cette migration accompagne le refactor "pixel-d'abord, SIG-en-option" :
    - has_georeference=False (défaut)  → la carte est restée en coords pixel ;
                                          le frontend la rend avec CRS.Simple
                                          + ImageOverlay aux dimensions natives.
    - has_georeference=True             → la carte a été géoréférencée vers
                                          WGS84 ; le frontend bascule sur le
                                          rendu SIG legacy.
"""
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vectorizer', '0003_trainingjob_unet_weights'),
    ]

    operations = [
        migrations.AddField(
            model_name='mapupload',
            name='has_georeference',
            field=models.BooleanField(
                default=False,
                help_text='True si la carte a été traitée avec géoréférencement actif.',
            ),
        ),
    ]
