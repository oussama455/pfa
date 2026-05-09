"""
webapp/vectorizer/api_v2.py — Updated API with Active Learning integration.

Changes from api.py V1:
    - MapCorrectionsView.patch() now calls process_correction() after saving
    - New endpoint: GET /api/calibration/{series}/  → registry status
    - New endpoint: GET /api/calibration/history/   → correction timeline
"""
from __future__ import annotations

import json
from pathlib import Path

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import MapUpload, Correction
from .tasks import enqueue_pipeline

# Reuse serializers from api.py
from .api import (
    MapUploadSerializer,
    CorrectionSerializer,
    CorrectionsPayloadSerializer,
    MapListCreateView,
    MapDetailView,
    MapStatusView,
    MapGeoJSONView,
    _apply_delete_to_geojson,
    _apply_edit_to_geojson,
)


# ─────────────────────────────────────────────────────────────────────────────
# Updated corrections endpoint with Active Learning
# ─────────────────────────────────────────────────────────────────────────────

class MapCorrectionsV2View(APIView):
    """
    PATCH /api/maps/{pk}/corrections/

    Saves HITL corrections AND triggers Active Learning calibration update.

    For each EDIT correction:
        1. Save Correction model to DB
        2. Update GeoJSON file on disk
        3. Call process_correction() → updates HSV registry via EMA

    For each DELETE correction:
        1. Save Correction model to DB
        2. Remove feature from GeoJSON file on disk
        3. Record negative example in registry (for Level 2 retraining)

    Returns:
        {
            "saved": N,
            "calibration_updates": [
                {
                    "layer": "red_roads",
                    "series": "ams_tunisia",
                    "corrections": 4,
                    "active": true,
                    "new_range": {
                        "H": [0, 10], "S": [88, 210], "V": [68, 248]
                    }
                }
            ]
        }
    """

    def patch(self, request: Request, pk: int) -> Response:
        upload = get_object_or_404(MapUpload, pk=pk)

        serializer = CorrectionsPayloadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        corrections_data = serializer.validated_data["corrections"]
        saved = 0
        calibration_updates = []

        # Try to import active learning (graceful fallback if not available)
        try:
            from pipeline.active_learning import (
                process_correction as al_process,
                MIN_CORRECTIONS_TO_ACTIVATE,
            )
            al_available = True
        except ImportError:
            al_available = False

        for item in corrections_data:
            # 1 — Persist to DB
            correction = Correction.objects.create(
                map_upload=upload,
                correction_type=item["type"],
                layer_name=item["layer"],
                feature_id=str(item["feature_id"]),
                geometry=item.get("geometry"),
                operator=request.user if request.user.is_authenticated else None,
            )
            saved += 1

            # 2 — Update GeoJSON on disk
            if item["type"] == "delete":
                _apply_delete_to_geojson(
                    upload.output_dir,
                    item["layer"],
                    str(item["feature_id"]),
                )
            elif item["type"] == "edit" and item.get("geometry"):
                _apply_edit_to_geojson(
                    upload.output_dir,
                    item["layer"],
                    str(item["feature_id"]),
                    item["geometry"],
                )

            # 3 — Active Learning calibration update
            if al_available:
                try:
                    updated_range = al_process(
                        correction=correction,
                        map_upload=upload,
                    )
                    if updated_range is not None:
                        calibration_updates.append({
                            "layer":       updated_range.layer_name,
                            "series":      updated_range.map_series,
                            "corrections": updated_range.correction_count,
                            "active":      updated_range.correction_count >= MIN_CORRECTIONS_TO_ACTIVATE,
                            "new_range": {
                                "H": [round(updated_range.h_min), round(updated_range.h_max)],
                                "S": [round(updated_range.s_min), round(updated_range.s_max)],
                                "V": [round(updated_range.v_min), round(updated_range.v_max)],
                            },
                        })
                except Exception as exc:
                    # AL failure is non-fatal — correction is already saved
                    import logging
                    logging.getLogger(__name__).warning(
                        "[api_v2] Active Learning update failed for layer=%s: %s",
                        item["layer"], exc,
                    )

        return Response({
            "saved":                saved,
            "map_id":               pk,
            "calibration_updates":  calibration_updates,
            "active_learning":      al_available,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Calibration status endpoints
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationStatusView(APIView):
    """
    GET /api/calibration/{series}/

    Returns the current HSV calibration state for a map series.
    Used by the React frontend to show a "Calibration active" badge
    and the current adapted HSV ranges per layer.

    Example response:
        {
            "series": "ams_tunisia",
            "active": true,
            "layers": {
                "red_roads": {
                    "corrections": 5,
                    "active": true,
                    "H": [0, 10], "S": [88, 215], "V": [68, 248],
                    "last_updated": "2025-05-04T21:31:00Z"
                },
                "buildings": {
                    "corrections": 2,
                    "active": false,
                    ...
                }
            }
        }
    """

    def get(self, request: Request, series: str) -> Response:
        try:
            from pipeline.active_learning import (
                load_registry,
                DEFAULT_ADAPTIVE_RANGES,
                MIN_CORRECTIONS_TO_ACTIVATE,
            )
        except ImportError:
            return Response(
                {"detail": "Active Learning module not available."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        registry = load_registry()
        series_data = registry.get(series)

        if series_data is None:
            available = list(DEFAULT_ADAPTIVE_RANGES.keys())
            return Response(
                {
                    "detail": f"Series '{series}' not found.",
                    "available_series": available,
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        import datetime

        layers_response = {}
        for layer_name, r in series_data.items():
            is_active = r.correction_count >= MIN_CORRECTIONS_TO_ACTIVATE
            last_updated = (
                datetime.datetime.fromtimestamp(r.last_updated, tz=datetime.timezone.utc).isoformat()
                if r.last_updated else None
            )
            layers_response[layer_name] = {
                "corrections":  r.correction_count,
                "active":       is_active,
                "H":            [round(r.h_min), round(r.h_max)],
                "S":            [round(r.s_min), round(r.s_max)],
                "V":            [round(r.v_min), round(r.v_max)],
                "last_updated": last_updated,
            }

        overall_active = any(
            r.correction_count >= MIN_CORRECTIONS_TO_ACTIVATE
            for r in series_data.values()
        )

        return Response({
            "series":                  series,
            "active":                  overall_active,
            "min_corrections_needed":  MIN_CORRECTIONS_TO_ACTIVATE,
            "layers":                  layers_response,
        })


class CalibrationHistoryView(APIView):
    """
    GET /api/calibration/history/?map_id={pk}&layer={name}

    Returns the correction history for a specific map or layer.
    Used by the React frontend to show a timeline of HITL corrections
    and their effect on calibration.
    """

    def get(self, request: Request) -> Response:
        map_id     = request.query_params.get("map_id")
        layer_name = request.query_params.get("layer")

        qs = Correction.objects.select_related("map_upload").order_by("-created_at")

        if map_id:
            qs = qs.filter(map_upload_id=map_id)
        if layer_name:
            qs = qs.filter(layer_name=layer_name)

        qs = qs[:100]   # cap at 100 entries

        history = [
            {
                "id":              c.pk,
                "map_id":          c.map_upload_id,
                "map_title":       c.map_upload.title,
                "type":            c.correction_type,
                "layer":           c.layer_name,
                "feature_id":      c.feature_id,
                "has_geometry":    c.geometry is not None,
                "created_at":      c.created_at.isoformat(),
                "operator":        c.operator.username if c.operator else "anonymous",
            }
            for c in qs
        ]

        return Response({
            "count":   len(history),
            "history": history,
        })


class CalibrationResetView(APIView):
    """
    POST /api/calibration/{series}/reset/

    Resets the HSV calibration for a series back to defaults.
    Requires staff permission (military security context).
    """

    def post(self, request: Request, series: str) -> Response:
        if not request.user.is_staff:
            return Response(
                {"detail": "Staff permission required."},
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            from pipeline.active_learning import (
                load_registry,
                save_registry,
                DEFAULT_ADAPTIVE_RANGES,
            )
        except ImportError:
            return Response(
                {"detail": "Active Learning module not available."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        from dataclasses import asdict

        registry = load_registry()
        if series not in DEFAULT_ADAPTIVE_RANGES:
            return Response(
                {"detail": f"Series '{series}' not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Reset to defaults
        from pipeline.active_learning import AdaptiveHSVRange
        registry[series] = {
            k: AdaptiveHSVRange(**asdict(v))
            for k, v in DEFAULT_ADAPTIVE_RANGES[series].items()
        }
        save_registry(registry)

        return Response({
            "reset":  True,
            "series": series,
            "detail": f"Calibration for '{series}' reset to factory defaults.",
        })
