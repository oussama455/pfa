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


def _parse_bool(value, *, default: bool = False) -> bool:
    """
    Parse les booléens venant d'un payload multipart (str) ou JSON (bool).
    Accepte : True/False, "true"/"false", "1"/"0", "yes"/"no", "on"/"off".
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _epsg_from_crs(crs_str, *, default: int = 4326) -> int:
    """
    Extrait le code EPSG numérique d'une chaîne CRS ("EPSG:4326", "4326", ...).
    Retourne `default` si rien d'exploitable.
    """
    if not crs_str:
        return default
    s = str(crs_str).strip().upper().replace("EPSG:", "")
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def _feature_id(feature: dict) -> str:
    props = feature.get("properties") or {}
    value = props.get("label_id")
    if value is None:
        value = props.get("id")
    if value is None:
        value = feature.get("id", "")
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Serializers
# ─────────────────────────────────────────────────────────────────────────────

class MapUploadSerializer(serializers.ModelSerializer):
    output_layers      = serializers.SerializerMethodField()
    raster_url         = serializers.SerializerMethodField()
    original_image_url = serializers.SerializerMethodField()
    raster_size        = serializers.SerializerMethodField()
    status_label       = serializers.SerializerMethodField()

    class Meta:
        model  = MapUpload
        fields = [
            "id", "title", "map_name", "map_type",
            "raster_url", "original_image_url", "raster_size",
            "unet_weights",
            "status", "status_label", "error_message",
            "confidence_score", "qa_passed", "retry_count",
            "has_georeference", "georef_crs", "raster_bounds",
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

    def get_original_image_url(self, obj):
        """
        Alias dédié au frontend : indique l'URL de l'image d'origine à
        charger comme ImageOverlay sous Leaflet en mode pixel (CRS.Simple).
        Identique à raster_url en pratique.
        """
        return self.get_raster_url(obj)

    def get_raster_size(self, obj):
        """
        Retourne (width, height) en pixels du raster original. Le frontend
        en a besoin pour fixer les bounds de L.CRS.Simple en mode pixel.
        Renvoie None si l'image est introuvable ou Pillow indisponible.
        """
        if not obj.raster:
            return None
        try:
            from PIL import Image
            with Image.open(obj.raster.path) as im:
                return {"width": im.width, "height": im.height}
        except Exception:  # noqa: BLE001
            return None

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
        # Expect: title (str), raster (file), map_name/unet_weights (optional)
        # Also accepts: georeference (bool, default False).
        title        = request.data.get("title", "Untitled Map")
        raster       = request.FILES.get("raster")
        map_name     = request.data.get("map_name", "")
        unet_weights = request.data.get("unet_weights") or None
        georeference = _parse_bool(request.data.get("georeference"), default=False)

        if not raster:
            return Response(
                {"detail": "No raster file provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        upload = MapUpload.objects.create(
            title=title,
            raster=raster,
            map_name=map_name or None,
            unet_weights=unet_weights,
            has_georeference=georeference,
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
            "id":               upload.pk,
            "status":           upload.status,
            "status_label":     upload.get_status_display(),
            "error":            upload.error_message,
            "has_georeference": upload.has_georeference,
            "layers":           upload.output_layers,
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
            "crs":     upload.georef_crs if upload.has_georeference else None,
            "layers":  layers,
        })


class MapShapefileDownloadView(APIView):
    """
    GET /api/maps/{pk}/shapefiles/ — télécharge un bundle QGIS prêt à l'emploi.

    Le ZIP produit contient :
        project.qgs            ← projet QGIS (couches pré-chargées + stylées)
        layers/<name>.shp ...  ← shapefiles LISSÉS (anti-staircase)

    Garde-fou : ce bundle n'a de sens qu'en mode SIG. Si la carte a été
    traitée en pixel pur (has_georeference=False), QGIS ne saurait pas
    positionner les couches → on refuse avec 409 et un message clair.
    """

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

        # ── Garde-fou pixel-only ─────────────────────────────────────────────
        if not getattr(upload, "has_georeference", False):
            return Response(
                {
                    "detail": (
                        "Export QGIS indisponible : cette carte a été traitée "
                        "en espace pixel (sans géoréférencement). QGIS requiert "
                        "un CRS pour positionner les couches. Relance le "
                        "traitement en cochant « Activer le géoréférencement (SIG) »."
                    ),
                    "has_georeference": False,
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

        # ── Construit le bundle QGIS (shapefiles lissés + project.qgs) ───────
        try:
            from pipeline.export import build_qgis_bundle
        except Exception as exc:  # noqa: BLE001
            return Response(
                {"detail": f"Export dependencies unavailable: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        crs_epsg = _epsg_from_crs(getattr(upload, "georef_crs", None))
        zip_path = output_dir / f"cartovec_export_{upload.pk}.zip"
        try:
            build_qgis_bundle(
                output_dir,
                zip_path,
                crs_epsg=crs_epsg,
                title=f"CartoVec — {upload.title}",
                smooth=True,
            )
        except RuntimeError as exc:
            return Response(
                {"detail": f"Aucune couche exportable : {exc}"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as exc:  # noqa: BLE001
            return Response(
                {"detail": f"Échec de génération du bundle QGIS : {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response = FileResponse(
            open(zip_path, "rb"),
            as_attachment=True,
            filename=f"cartovec_export_{upload.pk}.zip",
            content_type="application/zip",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="cartovec_export_{upload.pk}.zip"'
        )
        return response


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
            if _feature_id(feat) != feature_id
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
            if _feature_id(feat) == feature_id:
                feat["geometry"] = new_geometry
                break

        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(gj, f, indent=2)
    except (json.JSONDecodeError, OSError):
        pass
