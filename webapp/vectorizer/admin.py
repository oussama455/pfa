from django.contrib import admin

from .models import MapUpload, TrainingJob


@admin.register(MapUpload)
class MapUploadAdmin(admin.ModelAdmin):
    list_display = ('title', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('title',)
    readonly_fields = ('created_at', 'updated_at', 'status', 'error_message')


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'dataset', 'status', 'best_miou', 'created_at', 'finished_at')
    list_filter = ('dataset', 'status', 'created_at')
    readonly_fields = (
        'status', 'pid', 'log_path', 'output_weights_path',
        'best_miou', 'error_message', 'created_at', 'started_at', 'finished_at',
    )
