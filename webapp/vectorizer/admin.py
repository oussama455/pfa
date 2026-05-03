from django.contrib import admin

from .models import MapUpload


@admin.register(MapUpload)
class MapUploadAdmin(admin.ModelAdmin):
    list_display = ('title', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('title',)
    readonly_fields = ('created_at', 'updated_at', 'status', 'error_message')
