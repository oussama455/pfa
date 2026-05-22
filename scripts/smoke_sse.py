#!/usr/bin/env python3
"""
scripts/smoke_sse.py — Smoke test client pour l'endpoint SSE de l'agent
========================================================================

Mime un client navigateur frappant la vue de streaming SSE de CartoVec
(webapp/vectorizer/api_agent.py → AgentStreamView) et lit le flux ligne par
ligne, exactement comme le ferait un `EventSource` côté React.

But : valider en local, hors navigateur, que le backend émet bien la séquence
de paquets attendue (open → node_start → log* → agent_response → done) avant
de brancher l'UI. C'est un test « fil à fil » : aucune dépendance lourde
(torch/langgraph) côté client, juste un lecteur HTTP en streaming.

Lit le flux avec `requests` (stream=True) si disponible, sinon `httpx`.

Affichage normalisé :
    {"type": "node_start"}     → [NODE START] -> Executing: {node}
    {"type": "log"}            → [LOG] {message}
    {"type": "agent_response"} → [SUCCESS] Final output generated at: {geojson_url}
    {"type": "error"}          → [ERROR] {message}
    autres                     → [.] {type}  (open, done, ...)

Usage :
    python scripts/smoke_sse.py --map-id 1
    python scripts/smoke_sse.py --map-id 1 --georeference
    python scripts/smoke_sse.py --base-url http://localhost:8000 --map-id 3
    python scripts/smoke_sse.py --url "http://localhost:8000/api/agent/stream/?map_id=1"

Codes de sortie :
    0  flux terminé proprement (agent_response reçu)
    1  paquet d'erreur reçu de l'agent
    2  échec de connexion / I/O réseau
    3  aucune librairie HTTP (ni requests ni httpx)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator, Optional, Tuple
from urllib.parse import urlencode

# ── Sélection du backend HTTP (requests prioritaire, httpx en repli) ──────────
_HTTP_BACKEND: Optional[str] = None
try:
    import requests  # type: ignore
    _HTTP_BACKEND = "requests"
except Exception:  # noqa: BLE001
    try:
        import httpx  # type: ignore
        _HTTP_BACKEND = "httpx"
    except Exception:  # noqa: BLE001
        _HTTP_BACKEND = None


def build_url(args: argparse.Namespace) -> str:
    """Construit l'URL SSE finale depuis les arguments CLI."""
    if args.url:
        return args.url
    params = {}
    if args.map_id is not None:
        params["map_id"] = args.map_id
    params["georeference"] = "true" if args.georeference else "false"
    if args.thread_id:
        params["thread_id"] = args.thread_id
    base = args.base_url.rstrip("/")
    return f"{base}/api/agent/stream/?{urlencode(params)}"


def _iter_sse_lines_requests(url: str, timeout: float) -> Iterator[str]:
    """Itère les lignes brutes du flux via requests (stream=True)."""
    headers = {"Accept": "text/event-stream"}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        # iter_lines décode déjà le chunked transfer ; on garde utf-8.
        for raw in resp.iter_lines(decode_unicode=True):
            yield raw if raw is not None else ""


def _iter_sse_lines_httpx(url: str, timeout: float) -> Iterator[str]:
    """Itère les lignes brutes du flux via httpx (stream)."""
    headers = {"Accept": "text/event-stream"}
    with httpx.stream("GET", url, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            yield line


def iter_sse_lines(url: str, timeout: float) -> Iterator[str]:
    """Dispatch vers le backend HTTP disponible."""
    if _HTTP_BACKEND == "requests":
        yield from _iter_sse_lines_requests(url, timeout)
    elif _HTTP_BACKEND == "httpx":
        yield from _iter_sse_lines_httpx(url, timeout)
    else:  # pragma: no cover — garde-fou
        raise RuntimeError("Aucun backend HTTP (requests/httpx) disponible.")


def parse_sse_event(line: str) -> Optional[dict]:
    """
    Extrait le JSON d'une ligne SSE de la forme `data: {...}`.
    Renvoie None pour les lignes vides, commentaires (`:`) ou champs non-data.
    """
    if not line:
        return None
    if line.startswith(":"):  # commentaire / heartbeat SSE
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        print(f"[WARN] paquet JSON illisible : {payload!r}")
        return None


def render(packet: dict) -> Tuple[bool, bool]:
    """
    Affiche un paquet au format normalisé.
    Renvoie (is_terminal, is_error) pour piloter la boucle / le code de sortie.
    """
    ptype = packet.get("type")

    if ptype == "node_start":
        node = packet.get("node", "?")
        print(f"[NODE START] -> Executing: {node}")
        return False, False

    if ptype == "log":
        node = packet.get("node")
        prefix = f"[{node}] " if node else ""
        print(f"[LOG] {prefix}{packet.get('message', '')}")
        return False, False

    if ptype == "agent_response":
        url = packet.get("geojson_url") or "(aucune couche)"
        print(f"[SUCCESS] Final output generated at: {url}")
        layers = packet.get("layers") or {}
        if layers:
            print(f"          {len(layers)} couche(s) : {', '.join(layers.keys())}")
        if packet.get("qgis_bundle"):
            print(f"          Bundle QGIS : {packet['qgis_bundle']}")
        conf = packet.get("confidence")
        if conf is not None:
            print(f"          Confiance : {round(conf * 100)}%")
        return False, False  # 'done' clôt réellement le flux

    if ptype == "error":
        print(f"[ERROR] {packet.get('message', 'erreur inconnue')}")
        if packet.get("traceback"):
            print(packet["traceback"])
        return False, True

    if ptype == "open":
        tid = packet.get("thread_id", "?")
        print(f"[.] open (thread_id={tid}, map_id={packet.get('map_id')})")
        return False, False

    if ptype == "done":
        print("[.] done — flux clos par le serveur.")
        return True, False

    print(f"[.] {ptype}: {json.dumps(packet, ensure_ascii=False)}")
    return False, False


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test du flux SSE de l'agent CartoVec.")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Origine du serveur Django (def: http://localhost:8000).")
    parser.add_argument("--map-id", type=int, default=None,
                        help="PK de la carte (MapUpload) à traiter.")
    parser.add_argument("--georeference", action="store_true",
                        help="Active le mode SIG (def: mode pixel).")
    parser.add_argument("--thread-id", default=None,
                        help="Identifiant de session (def: généré côté serveur).")
    parser.add_argument("--url", default=None,
                        help="URL SSE complète (ignore base-url/map-id si fournie).")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Timeout réseau en secondes (def: 600).")
    args = parser.parse_args(argv)

    if _HTTP_BACKEND is None:
        print("[FATAL] Ni 'requests' ni 'httpx' n'est installé.")
        print("        pip install requests   # (ou httpx)")
        return 3

    url = build_url(args)
    print(f"[*] Backend HTTP : {_HTTP_BACKEND}")
    print(f"[*] Connexion SSE : {url}")
    print("[*] Lecture du flux ligne par ligne (Ctrl-C pour couper)…\n")

    got_error = False
    try:
        for line in iter_sse_lines(url, args.timeout):
            packet = parse_sse_event(line)
            if packet is None:
                continue
            is_terminal, is_error = render(packet)
            if is_error:
                got_error = True
            if is_terminal:
                break
    except KeyboardInterrupt:
        print("\n[*] Interrompu par l'utilisateur.")
        return 2
    except Exception as exc:  # noqa: BLE001 — toute erreur réseau/HTTP
        print(f"\n[FATAL] Échec de connexion / lecture du flux : "
              f"{type(exc).__name__}: {exc}")
        return 2

    print()
    if got_error:
        print("[*] Terminé avec un paquet d'erreur agent.")
        return 1
    print("[*] Terminé proprement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
