"""
Migration : ajoute MapUpload.unet_weights + nouveau modele TrainingJob.

A appliquer apres 0002.

    cd webapp
    python manage.py migrate
"""
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vectorizer', '0002_mapupload_confidence_score_mapupload_finished_at_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='mapupload',
            name='unet_weights',
            field=models.CharField(
                'Poids U-Net',
                max_length=500,
                blank=True, null=True,
                help_text="Chemin vers le .pth a utiliser. Vide = pas d'U-Net.",
            ),
        ),
        migrations.CreateModel(
            name='TrainingJob',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dataset', models.CharField(
                    choices=[('soduco', 'SODUCO Benchmark (5 classes)'),
                             ('semap',  'SEMAP Petitpierre (6 classes)')],
                    default='semap', max_length=20)),
                ('epochs',       models.PositiveSmallIntegerField(default=20)),
                ('batch_size',   models.PositiveSmallIntegerField(default=8)),
                ('target_size',  models.PositiveSmallIntegerField(default=512)),
                ('learning_rate', models.FloatField(default=0.0001)),
                ('encoder',      models.CharField(default='resnet34', max_length=32)),
                ('no_synthetic', models.BooleanField(
                    default=False,
                    help_text="SEMAP only : exclut les 12 122 images synthetiques.")),
                ('no_augment',   models.BooleanField(default=False)),
                ('status', models.CharField(
                    choices=[('queued',  "En file d'attente"),
                             ('running', 'En cours'),
                             ('done',    'Terminé'),
                             ('failed',  'Échec'),
                             ('aborted', 'Interrompu')],
                    default='queued', max_length=20)),
                ('pid',           models.IntegerField(blank=True, null=True)),
                ('log_path',      models.CharField(blank=True, max_length=500, null=True)),
                ('output_weights_path', models.CharField(blank=True, max_length=500, null=True)),
                ('best_miou',     models.FloatField(blank=True, null=True)),
                ('error_message', models.TextField(blank=True, null=True)),
                ('created_at',    models.DateTimeField(auto_now_add=True)),
                ('started_at',    models.DateTimeField(blank=True, null=True)),
                ('finished_at',   models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'Training Job',
                'verbose_name_plural': 'Training Jobs',
                'ordering': ['-created_at'],
            },
        ),
    ]
