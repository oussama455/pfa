"""
pipeline/agent_stream.py — Streaming en direct de l'agent LangGraph
====================================================================

Expose `stream_agent(...)`, un GÉNÉRATEUR qui exécute l'agent map-vectorisation
et produit des paquets JSON structurés au fur et à mesure des transitions de
nœuds du graphe. Conçu pour être branché derrière une vue SSE Django
(webapp/vectorizer/api_agent.py) qui relaie chaque paquet au chat React.

Types de paquets émis :
    {"type": "node_start", "node": "<nom_du_noeud>"}
    {"type": "log",        "message": "..."}
    {"type": "agent_response", "text": "...", "has_georeference": bool,
                               "geojson_url": "...", "qgis_bundle": "..."|None}
    {"type": "error",      "message": "..."}

Isolation de session : `thread_id` est transmis au checkpointer LangGraph
pour que chaque carte / utilisateur garde une mémoire de conversation séparée.

Note : ce module garde les imports lourds (langgraph, agent) en local pour
rester importable même quand langgraph n'est pas installé (les tests de
formatage de paquets fonctionnent alors via un graphe mocké).
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Ordre nominal des nœuds, pour piloter le stepper visuel côté frontend.
NODE_SEQUENCE = [
    "perceive", "preprocess", "vectorize", "qa_check",
    "self_correct", "georef", "export",
]

# Libellés lisibles pour l'UI (FR), par nœud.
NODE_LABELS = {
    "perceive":     "Analyse du type de carte",
    "preprocess":   "Prétraitement & recadrage",
    "vectorize":    "Segmentation & vectorisation",
    "qa_check":     "Contrôle qualité",
    "self_correct": "Auto-correction",
    "georef":       "Géoréférencement",
    "export":       "Export des couches",
}


def _packet(ptype: str, **fields) -> Dict:
    """Construit un paquet horodaté."""
    p = {"type": ptype, "ts": time.time()}
    p.update(fields)
    return p


def _logs_from_update(node_name: str, update: Dict) -> Iterator[Dict]:
    """
    Extrait des paquets `log` lisibles depuis la dernière entrée d'agent_log
    produite par un nœud, plus quelques infos clés selon le nœud.
    """
    agent_log = update.get("agent_log") or []
    last = agent_log[-1] if agent_log else {}

    # Message générique de progression
    label = NODE_LABELS.get(node_name, node_name)
    yield _packet("log", node=node_name, message=f"{label} — terminé.")

    # Infos spécifiques utiles à afficher dans la console
    if node_name == "preprocess":
        bbox = update.get("crop_bbox")
        scale = update.get("downscale_scale")
        if bbox is not None and scale:
            inv = (1.0 / scale) if scale else 1.0
            yield _packet(
                "log", node=node_name,
                message=(f"Recadrage offset ({bbox[0]}, {bbox[1]}) "
                         f"et facteur d'échelle {scale:.3f} "
                         f"(réalignement ×{inv:.2f} vers l'image originale)."),
            )
    elif node_name == "vectorize":
        score = update.get("confidence_score")
        layers = last.get("layers_produced")
        if score is not None:
            yield _packet("log", node=node_name,
                          message=f"Confiance={score:.0%} — couches: {layers}")
    elif node_name == "qa_check":
        fb = last.get("qa_feedback")
        if fb:
            yield _packet("log", node=node_name, message=f"QA: {fb}")
    elif node_name == "georef":
        st = last.get("georef_status")
        if st:
            yield _packet("log", node=node_name, message=f"Géoréf: {st}")


def _build_final_packet(final_state: Dict, *, media_url_base: str = "") -> Dict:
    """
    Construit le paquet agent_response terminal à partir de l'état final.
    media_url_base : préfixe d'URL pour transformer un chemin disque en URL
        servable (laisser vide si le chemin est déjà une URL).
    """
    outputs = final_state.get("output_geojsons") or {}
    has_geo = bool(final_state.get("georef_crs"))
    qgis_bundle = final_state.get("qgis_bundle")

    # Première couche comme aperçu (le frontend liste le reste si besoin)
    geojson_url = None
    if outputs:
        first_path = next(iter(outputs.values()))
        geojson_url = (media_url_base + Path(first_path).name) if media_url_base else first_path

    n_layers = len(outputs)
    mode = "géoréférencé (WGS84)" if has_geo else "espace pixel"
    text = (f"Vectorisation terminée — {n_layers} couche(s) extraite(s) "
            f"en {mode}.")
    if qgis_bundle:
        text += " Un projet QGIS prêt à l'emploi est inclus."

    return _packet(
        "agent_response",
        text=text,
        has_georeference=has_geo,
        geojson_url=geojson_url,
        layers={k: (media_url_base + Path(v).name if media_url_base else v)
                for k, v in outputs.items()},
        qgis_bundle=(media_url_base + Path(qgis_bundle).name
                     if (qgis_bundle and media_url_base) else qgis_bundle),
        confidence=final_state.get("confidence_score"),
        qa_passed=final_state.get("qa_passed"),
    )


def stream_agent(raster_path: str,
                 output_dir: str,
                 *,
                 map_name: Optional[str] = None,
                 weights_path: Optional[str] = None,
                 device: Optional[str] = None,
                 georeference: bool = False,
                 thread_id: Optional[str] = None,
                 media_url_base: str = "") -> Iterator[Dict]:
    """
    Exécute l'agent en streaming et produit des paquets (dicts) au fil des
    transitions de nœuds. À consommer par une vue SSE.

    Chaque `chunk` renvoyé par `graph.stream()` est de la forme
    `{node_name: state_update}`. On émet pour chacun :
        node_start -> log(s) dérivés de l'update.
    À la fin, on récupère l'état complet via get_state(thread_id) (si
    checkpointer) ou via l'accumulation des updates, et on émet agent_response.

    Toute exception (CUDA crash, fichier illisible, etc.) est capturée et
    émise comme paquet `error` — le graphe ne casse jamais le flux SSE.
    """
    try:
        from pipeline.agent import build_agent
    except Exception as exc:  # noqa: BLE001 — langgraph/deps manquants
        yield _packet("error",
                      message=f"Agent indisponible (dépendances manquantes) : {exc}")
        return

    # Validation d'entrée explicite -> erreur claire plutôt que crash plus loin
    if not Path(raster_path).is_file():
        yield _packet("error", message=f"Fichier raster introuvable : {raster_path}")
        return

    try:
        agent = build_agent(with_memory=bool(thread_id))
    except Exception as exc:  # noqa: BLE001
        yield _packet("error", message=f"Échec d'initialisation de l'agent : {exc}")
        return

    initial_state = {
        "raster_path":  str(raster_path),
        "output_dir":   str(output_dir),
        "map_name":     map_name,
        "weights_path": weights_path,
        "device":       device,
        "georeference": georeference,
        "retry_count":  0,
        "agent_log":    [],
    }
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    yield _packet("log", message=f"Démarrage de l'agent (thread={thread_id or 'anonyme'}, "
                                 f"georeference={georeference}).")

    accumulated: Dict = dict(initial_state)
    try:
        for chunk in agent.stream(initial_state, config=config):
            # chunk = {node_name: partial_state_update}
            for node_name, update in chunk.items():
                yield _packet("node_start", node=node_name,
                              label=NODE_LABELS.get(node_name, node_name))
                if isinstance(update, dict):
                    accumulated.update(update)
                    for log_pkt in _logs_from_update(node_name, update):
                        yield log_pkt
        # Stream terminé -> état final = accumulation
        yield _build_final_packet(accumulated, media_url_base=media_url_base)

    except Exception as exc:  # noqa: BLE001 — CUDA, IO, etc.
        logger.exception("[stream_agent] échec : %s", exc)
        yield _packet(
            "error",
            message=f"Échec de l'agent : {type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=4),
        )
