"""
webapp/vectorizer/urls.py — URL routing V2 (with Active Learning endpoints).
"""
from django.urls import path
from . import views
from .api import (
    MapListCreateView,
    MapDetailView,
    MapStatusView,
    MapGeoJSONView,
    MapShapefileDownloadView,
)
from .api_v2 import (
    MapCorrectionsV2View,
    CalibrationStatusView,
    CalibrationHistoryView,
    CalibrationResetView,
)
from .api_training import (
    WeightsListView,
    TrainingJobListCreateView,
    TrainingJobDetailView,
    TrainingJobLogView,
    TrainingJobDownloadView,
)

app_name = 'vectorizer'

urlpatterns = [
    # ── Django template views ─────────────────────────────────────────────
    path('',                      views.upload_view, name='upload'),
    path('maps/<int:pk>/',        views.result_view, name='result'),
    path('maps/<int:pk>/status/', views.status_view, name='status'),

    # ── REST API — Maps ───────────────────────────────────────────────────
    path('api/maps/',                       MapListCreateView.as_view(),   name='api-maps'),
    path('api/maps/<int:pk>/',              MapDetailView.as_view(),       name='api-map-detail'),
    path('api/maps/<int:pk>/status/',       MapStatusView.as_view(),       name='api-map-status'),
    path('api/maps/<int:pk>/geojson/',      MapGeoJSONView.as_view(),      name='api-map-geojson'),
    path('api/maps/<int:pk>/shapefiles/',   MapShapefileDownloadView.as_view(), name='api-map-shapefiles'),
    path('api/maps/<int:pk>/corrections/',  MapCorrectionsV2View.as_view(),name='api-corrections'),

    # ── REST API — Active Learning Calibration ────────────────────────────
    path('api/calibration/history/',              CalibrationHistoryView.as_view(), name='api-calib-history'),
    path('api/calibration/<str:series>/',         CalibrationStatusView.as_view(),  name='api-calibration'),
    path('api/calibration/<str:series>/reset/',   CalibrationResetView.as_view(),   name='api-calib-reset'),

    # ── REST API — Training U-Net + selection des poids ───────────────────
    path('api/weights/',                     WeightsListView.as_view(),              name='api-weights'),
    path('api/training/',                    TrainingJobListCreateView.as_view(),    name='api-training'),
    path('api/training/<int:pk>/',           TrainingJobDetailView.as_view(),        name='api-training-detail'),
    path('api/training/<int:pk>/log/',       TrainingJobLogView.as_view(),           name='api-training-log'),
    path('api/training/<int:pk>/download/',  TrainingJobDownloadView.as_view(),      name='api-training-download'),
]
