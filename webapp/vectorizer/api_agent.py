"""
webapp/vectorizer/api_agent.py — Endpoint SSE de streaming de l'agent
"""
from __future__ import annotations

import json
import uuid

from django.conf import settings
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
# IMPORTANT: On utilise les requêtes standard de Django, pas celles de REST Framework
from django.http import HttpRequest 

from .models import MapUpload
from .api import _parse_bool


def _sse(packet: dict) -> str:
    """Encode un dict en ligne SSE `data: {...}\n\n`."""
    return f"data: {json.dumps(packet, ensure_ascii=False)}\n\n"


def _sse_retry(milliseconds: int) -> str:
    """Set EventSource reconnect delay; this stream is an expensive one-shot run."""
    return f"retry: {milliseconds}\n\n"


def agent_stream_view(request: HttpRequest) -> StreamingHttpResponse:
    """
    Vue Django standard (SANS DRF) pour diffuser en SSE
    l'exécution de l'agent sur la carte <map_id>.
    """
    # Remplacement de request.query_params par request.GET pour Django natif
    map_id = request.GET.get("map_id")
    georeference = _parse_bool(request.GET.get("georeference"), default=False)
    thread_id = request.GET.get("thread_id") or f"sess-{uuid.uuid4().hex[:12]}"

    upload = get_object_or_404(MapUpload, pk=map_id) if map_id else None

    def event_stream():
        yield _sse_retry(3_600_000)
        yield _sse({"type": "open", "thread_id": thread_id,
                    "map_id": int(map_id) if map_id else None})

        if upload is None:
            yield _sse({"type": "error",
                        "message": "Paramètre map_id manquant ou carte introuvable."})
            return

        final_packet = None
        error_message = None

        try:
            try:
                upload.mark_processing()
            except Exception:
                pass

            raster_path = str(upload.raster_path)
            output_dir = str(upload.output_dir)
            media_base = getattr(settings, "MEDIA_URL", "/media/")
            
            try:
                rel = upload.output_dir.relative_to(settings.MEDIA_ROOT)
                media_url_base = media_base + str(rel).replace("\\", "/") + "/"
            except Exception:
                media_url_base = ""

            from pipeline.agent_stream import stream_agent
            for packet in stream_agent(
                raster_path,
                output_dir,
                map_name=upload.map_name,
                georeference=georeference,
                thread_id=thread_id,
                media_url_base=media_url_base,
            ):
                if packet.get("type") == "agent_response":
                    final_packet = packet
                elif packet.get("type") == "error":
                    error_message = packet.get("message") or "Erreur agent."
                yield _sse(packet)

            try:
                if error_message:
                    upload.mark_failed(error_message)
                elif final_packet is not None:
                    has_georef = bool(final_packet.get("has_georeference"))
                    upload.mark_done(
                        confidence_score=final_packet.get("confidence"),
                        qa_passed=final_packet.get("qa_passed"),
                        georef_crs="EPSG:4326" if has_georef else None,
                        has_georeference=has_georef,
                    )
            except Exception:
                pass

        except Exception as exc:
            try:
                upload.mark_failed(str(exc))
            except Exception:
                pass
            yield _sse({"type": "error",
                        "message": f"Erreur serveur durant le stream : {exc}"})
        finally:
            yield _sse({"type": "done"})

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )
    # Configuration sécurisée des en-têtes pour WSGI et le dev local
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    
    # RETRAIT CRUCIAL : Ne pas forcer response["Connection"] = "keep-alive"
    
    return response
