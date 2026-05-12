"""
webapp/vectorizer/api.py — Django REST Framework API for CartoVec.

Endpoints:
    GET  /api/maps/{pk}/geojson/         → all GeoJSON layers for a map
    PATCH /api/maps/{pk}/corrections/    → save HITL corrections
    GET  /api/maps/{pk}/status/          → processing status (polling)
    GET  /api/maps/                      → list all maps
    POST /api/maps/                      → upload + start pipeline
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from django.conf import settings
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import MapUpload, Correction
from .tasks import enqueue_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Serializers
# ─────────────────────────────────────────────────────────────────────────────

class MapUploadSerializer(serializers.ModelSerializer):
    output_layers = serializers.SerializerMethodField()
    raster_url    = serializers.SerializerMethodField()
    status_label  = serializers.SerializerMethodField()

    class Meta:
        model  = MapUpload
        fields = [
            "id", "title", "map_name", "map_type", "raster_url",
            "status", "status_label", "error_message",
            "confidence_score", "qa_passed", "retry_count",
            "georef_crs", "raster_bounds",
            "created_at", "updated_at", "finished_at", "output_layers",
        ]
        read_only_fields = ["id", "status", "created_at", "updated_at"]

    def get_output_layers(self, obj):
        return obj.output_layers   # {layer_name: media_url}

    def get_raster_url(self, obj):
        request = self.context.get("request")
        if obj.raster and request:
            return request.build_absolute_uri(obj.raster.url)
        return obj.raster.url if obj.raster else None

    def get_status_label(self, obj):
        return obj.get_status_display()


class CorrectionSerializer(serializers.Serializer):
    """Single HITL correction entry."""
    type        = serializers.ChoiceField(choices=["delete", "edit"])
    layer       = serializers.CharField(max_length=100)
    feature_id  = serializers.CharField(max_length=200)
    geometry    = serializers.JSONField(required=False, allow_null=True)
    timestamp   = serializers.DateTimeField(required=False)


class CorrectionsPayloadSerializer(serializers.Serializer):
    """Batch of corrections sent from the React frontend."""
    corrections = CorrectionSerializer(many=True)


# ─────────────────────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────────────────────

class MapListCreateView(APIView):
    """
    GET  /api/maps/   — list all maps (most recent first)
    POST /api/maps/   — upload a new raster and start the pipeline
    """
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request: Request) -> Response:
        maps = MapUpload.objects.all()[:50]
        serializer = MapUploadSerializer(maps, many=True, context={"request": request})
        return Response(serializer.data)

    def post(self, request: Request) -> Response:
        # Expect: title (str), raster (file), map_name (str, optional)
        title    = request.data.get("title", "Untitled Map")
        raster   = request.FILES.get("raster")
        map_name = request.data.get("map_name", "")

        if not raster:
            return Response(
                {"detail": "No raster file provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        upload = MapUpload.objects.create(
            title=title,
            raster=raster,
            map_name=map_name or None,
        )

        # Start async pipeline (thread or Celery, depending on tasks.py)
        enqueue_pipeline(upload)

        serializer = MapUploadSerializer(upload, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class MapDetailView(APIView):
    """GET /api/maps/{pk}/ — single map detail."""

    def get(self, request: Request, pk: int) -> Response:
        upload = get_object_or_404(MapUpload, pk=pk)
        serializer = MapUploadSerializer(upload, context={"request": request})
        return Response(serializer.data)


class MapStatusView(APIView):
    """
    GET /api/maps/{pk}/status/
    Lightweight polling endpoint — React frontend polls this every 2 s.
    Returns only status + layers, not the full object.
    """

    def get(self, request: Request, pk: int) -> Response:
        upload = get_object_or_404(MapUpload, pk=pk)
        return Response({
            "id":           upload.pk,
            "status":       upload.status,
            "status_label": upload.get_status_display(),
            "error":        upload.error_message,
            "layers":       upload.output_layers,
        })


class MapGeoJSONView(APIView):
    """
    GET /api/maps/{pk}/geojson/

    Returns all GeoJSON layers for a processed map, merged into a single
    response dict:
        {
            "map_id": 42,
            "crs": "EPSG:4326",
            "layers": {
                "buildings":  { "type": "FeatureCollection", "features": [...] },
                "red_roads":  { ... },
                ...
            }
        }

    The GeoJSON files are read from disk (output_dir property of MapUpload).
    This avoids storing huge JSON blobs in the database.
    """

    def get(self, request: Request, pk: int) -> Response:
        upload = get_object_or_404(MapUpload, pk=pk)

        if upload.status != "done":
            return Response(
                {
                    "detail": f"Map is not yet processed (status={upload.status}).",
                    "status": upload.status,
                },
                status=status.HTTP_202_ACCEPTED,
            )

        output_dir = upload.output_dir
        if not output_dir.exists():
            return Response(
                {"detail": "Output directory not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        layers = {}
        for geojson_file in sorted(output_dir.glob("*.geojson")):
            layer_name = geojson_file.stem
            try:
                with open(geojson_file, "r", encoding="utf-8") as f:
                    layers[layer_name] = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                layers[layer_name] = {"error": str(exc), "features": []}

        return Response({
            "map_id":  upload.pk,
            "title":   upload.title,
            "crs":     "EPSG:4326",
            "layers":  layers,
        })


class MapShapefileDownloadView(APIView):
    """GET /api/maps/{pk}/shapefiles/ - download all layers as a ZIP."""

    def get(self, request: Request, pk: int) -> Response | FileResponse:
        upload = get_object_or_404(MapUpload, pk=pk)

        if upload.status != "done":
            return Response(
                {
                    "detail": f"Map is not yet processed (status={upload.status}).",
                    "status": upload.status,
                },
                status=status.HTTP_409_CONFLICT,
            )

        output_dir = upload.output_dir
        geojson_files = sorted(output_dir.glob("*.geojson"))
        if not geojson_files:
            return Response(
                {"detail": "No GeoJSON layers found for this map."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            import geopandas as gpd
            from pipeline.vectorization import save_shapefile
        except Exception as exc:  # noqa: BLE001
            return Response(
                {"detail": f"Shapefile export dependencies are unavailable: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        shapefile_dir = output_dir / "shapefiles"
        shapefile_dir.mkdir(parents=True, exist_ok=True)

        written_files: list[Path] = []
        for geojson_file in geojson_files:
            layer_dir = shapefile_dir / geojson_file.stem
            layer_dir.mkdir(parents=True, exist_ok=True)
            shp_path = layer_dir / f"{geojson_file.stem}.shp"
            try:
                gdf = gpd.read_file(geojson_file)
                if gdf.empty:
                    continue
                save_shapefile(gdf, shp_path)
                written_files.extend(sorted(layer_dir.glob(f"{geojson_file.stem}.*")))
            except Exception as exc:  # noqa: BLE001
                return Response(
                    {
                        "detail": (
                            f"Failed to export layer '{geojson_file.stem}' "
                            f"as Shapefile: {exc}"
                        )
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        if not written_files:
            return Response(
                {"detail": "No non-empty layers available for Shapefile export."},
                status=status.HTTP_404_NOT_FOUND,
            )

        zip_path = output_dir / f"map_{upload.pk}_shapefiles.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in written_files:
                archive.write(
                    file_path,
                    arcname=f"{file_path.parent.name}/{file_path.name}",
                )

        return FileResponse(
            open(zip_path, "rb"),
            as_attachment=True,
            filename=f"cartovec_map_{upload.pk}_shapefiles.zip",
        )


class MapCorrectionsView(APIView):
    """
    PATCH /api/maps/{pk}/corrections/

    Saves HITL corrections from the React MapViewer.

    Body:
        {
            "corrections": [
                { "type": "delete", "layer": "buildings", "feature_id": "42" },
                { "type": "edit",   "layer": "buildings", "feature_id": "17",
                  "geometry": { "type": "Polygon", "coordinates": [...] } }
            ]
        }

    Each correction is stored in the Correction model for:
        1. Immediate visual feedback (MapViewer reads deletedIds)
        2. Future model fine-tuning (negative examples for U-Net retraining)
        3. Audit trail for military chain-of-custody

    Returns:
        { "saved": N, "map_id": pk }
    """

    def patch(self, request: Request, pk: int) -> Response:
        upload = get_object_or_404(MapUpload, pk=pk)

        serializer = CorrectionsPayloadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        corrections_data = serializer.validated_data["corrections"]
        saved = 0

        for item in corrections_data:
            Correction.objects.create(
                map_upload=upload,
                correction_type=item["type"],
                layer_name=item["layer"],
                feature_id=str(item["feature_id"]),
                geometry=item.get("geometry"),    # None for deletes
            )
            saved += 1

            # For deletes: optionally update the stored GeoJSON on disk
            if item["type"] == "delete":
                _apply_delete_to_geojson(
                    upload.output_dir,
                    item["layer"],
                    str(item["feature_id"]),
                )

            # For edits: update the stored GeoJSON with new geometry
            elif item["type"] == "edit" and item.get("geometry"):
                _apply_edit_to_geojson(
                    upload.output_dir,
                    item["layer"],
                    str(item["feature_id"]),
                    item["geometry"],
                )

        return Response({"saved": saved, "map_id": pk})


# ─────────────────────────────────────────────────────────────────────────────
# GeoJSON file mutation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_delete_to_geojson(
    output_dir: Path,
    layer_name: str,
    feature_id: str,
) -> None:
    """
    Removes a feature from the stored GeoJSON file on disk.

    This keeps the on-disk GeoJSON in sync with HITL corrections so that
    subsequent API calls to /geojson/ reflect the deletions.
    """
    geojson_path = output_dir / f"{layer_name}.geojson"
    if not geojson_path.exists():
        return
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        original_count = len(gj.get("features", []))
        gj["features"] = [
            feat for feat in gj.get("features", [])
            if str(feat.get("properties", {}).get("label_id", "")) != feature_id
            and str(feat.get("id", "")) != feature_id
        ]

        if len(gj["features"]) < original_count:
            with open(geojson_path, "w", encoding="utf-8") as f:
                json.dump(gj, f, indent=2)
    except (json.JSONDecodeError, OSError):
        pass   # non-fatal — correction is still saved to DB


def _apply_edit_to_geojson(
    output_dir: Path,
    layer_name: str,
    feature_id: str,
    new_geometry: dict,
) -> None:
    """
    Updates a feature's geometry in the stored GeoJSON file on disk.
    """
    geojson_path = output_dir / f"{layer_name}.geojson"
    if not geojson_path.exists():
        return
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        for feat in gj.get("features", []):
            fid = str(feat.get("properties", {}).get("label_id", "")) or str(feat.get("id", ""))
            if fid == feature_id:
                feat["geometry"] = new_geometry
                break

        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(gj, f, indent=2)
    except (json.JSONDecodeError, OSError):
        pass
