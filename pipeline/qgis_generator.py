"""
pipeline/qgis_generator.py — Générateur de projet QGIS (.qgs)
=============================================================

Écrit un fichier projet QGIS minimal mais valide (XML) qui :
    - définit un CRS de projet (par défaut EPSG:4326),
    - référence chaque couche vectorielle par un chemin RELATIF
      (ex. ./layers/red_roads.shp) pour que le projet s'ouvre quel que
      soit le dossier d'extraction (C:\\Users\\... ou /home/...),
    - applique un style de base par couche selon la convention
      cartographique militaire (bleu eau, rouge routes, vert végétation,
      marron courbes, gris bâti).

Ce module ne dépend PAS de QGIS ni de PyQGIS : il génère le XML à la main.
Le format ciblé est compatible QGIS 3.x (versions LTR récentes).

Usage :
    from pipeline.qgis_generator import build_qgs_project
    xml = build_qgs_project(
        layers=[("red_roads", "linestring"), ("water", "polygon")],
        layers_subdir="layers",
        crs_epsg=4326,
        title="CartoVec — Tunis",
    )
    Path("project.qgs").write_text(xml, encoding="utf-8")

Notes :
    - On utilise des QgsVectorLayer "ogr" pointant vers les .shp.
    - Le style est volontairement simple (renderer "singleSymbol").
"""
from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom


# Convention cartographique militaire → (couleur RGBA, type de symbole)
# Couleur QGIS : "R,G,B,A" (0-255).
LAYER_STYLES: Dict[str, Dict[str, str]] = {
    "water":      {"color": "41,128,185,160",  "outline": "31,97,141,255",   "geom": "polygon"},
    "vegetation": {"color": "46,204,113,140",   "outline": "39,174,96,255",   "geom": "polygon"},
    "buildings":  {"color": "127,140,141,180",  "outline": "44,62,80,255",    "geom": "polygon"},
    "red_roads":  {"color": "231,76,60,255",    "outline": "192,57,43,255",   "geom": "linestring"},
    "roads":      {"color": "231,76,60,255",    "outline": "192,57,43,255",   "geom": "linestring"},
    "contours":   {"color": "139,69,19,255",    "outline": "139,69,19,255",   "geom": "linestring"},
    "_default":   {"color": "155,89,182,150",   "outline": "142,68,173,255",  "geom": "polygon"},
}

# Géométries traitées comme des lignes (sinon polygone)
_LINE_GEOMS = {"linestring", "line", "multilinestring"}


def _style_for(layer_name: str, geom_hint: Optional[str] = None) -> Dict[str, str]:
    """Retourne le style pour une couche, avec repli sur _default."""
    style = dict(LAYER_STYLES.get(layer_name, LAYER_STYLES["_default"]))
    if geom_hint:
        style["geom"] = geom_hint
    return style


def _symbol_xml(style: Dict[str, str]) -> str:
    """
    Construit le bloc <symbol> QGIS adapté au type de géométrie.
    Ligne → simpleLine ; Polygone → simpleFill.
    """
    is_line = style["geom"].lower() in _LINE_GEOMS
    color = style["color"]
    outline = style["outline"]

    if is_line:
        return (
            '<symbol alpha="1" type="line" name="0">'
            '<layer class="SimpleLine" enabled="1">'
            f'<prop k="line_color" v="{color}"/>'
            '<prop k="line_width" v="0.45"/>'
            '<prop k="capstyle" v="round"/>'
            '<prop k="joinstyle" v="round"/>'
            '</layer></symbol>'
        )
    return (
        '<symbol alpha="1" type="fill" name="0">'
        '<layer class="SimpleFill" enabled="1">'
        f'<prop k="color" v="{color}"/>'
        f'<prop k="outline_color" v="{outline}"/>'
        '<prop k="outline_width" v="0.26"/>'
        '<prop k="style" v="solid"/>'
        '<prop k="outline_style" v="solid"/>'
        '</layer></symbol>'
    )


def _wkb_type(geom: str) -> str:
    """Mappe un hint de géométrie vers le wkbType QGIS (texte)."""
    g = geom.lower()
    if g in _LINE_GEOMS:
        return "LineString"
    if "point" in g:
        return "Point"
    return "Polygon"


def _maplayer_xml(layer_name: str,
                  rel_path: str,
                  crs_epsg: int,
                  style: Dict[str, str]) -> Tuple[str, str]:
    """
    Construit un bloc <maplayer> et retourne (layer_id, xml).

    rel_path : chemin RELATIF vers le .shp (ex. ./layers/red_roads.shp).
    """
    layer_id = f"{layer_name}_{uuid.uuid4().hex[:12]}"
    geom_type = _wkb_type(style["geom"])
    symbol = _symbol_xml(style)

    xml = (
        f'<maplayer type="vector" geometry="{geom_type}">'
        f'<id>{layer_id}</id>'
        f'<datasource>{rel_path}</datasource>'
        f'<layername>{layer_name}</layername>'
        f'<srs><spatialrefsys>'
        f'<authid>EPSG:{crs_epsg}</authid>'
        f'</spatialrefsys></srs>'
        f'<provider>ogr</provider>'
        f'<renderer-v2 type="singleSymbol">'
        f'<symbols>{symbol}</symbols>'
        f'</renderer-v2>'
        f'</maplayer>'
    )
    return layer_id, xml


def build_qgs_project(layers: List[Tuple[str, str]],
                      *,
                      layers_subdir: str = "layers",
                      crs_epsg: int = 4326,
                      title: str = "CartoVec Export",
                      ext: str = "shp") -> str:
    """
    Génère le contenu XML d'un projet QGIS (.qgs).

    Arguments
    ---------
    layers : liste de (nom_couche, type_geometrie). type_geometrie ∈
             {"polygon", "linestring", ...}. Si inconnu, on déduit du style.
    layers_subdir : sous-dossier (relatif) où sont les shapefiles dans le zip.
    crs_epsg : code EPSG du projet (4326 par défaut).
    title : titre du projet affiché dans QGIS.
    ext : extension des fichiers de couche ("shp" ou "geojson").

    Retourne une chaîne XML prête à écrire en .qgs (encodage UTF-8).
    """
    layer_blocks: List[str] = []
    layer_ids: List[str] = []
    legend_items: List[str] = []
    order_items: List[str] = []

    for name, geom in layers:
        style = _style_for(name, geom)
        rel_path = f"./{layers_subdir}/{name}.{ext}"
        layer_id, block = _maplayer_xml(name, rel_path, crs_epsg, style)
        layer_blocks.append(block)
        layer_ids.append(layer_id)
        legend_items.append(
            f'<legendlayer name="{name}" checked="Qt::Checked">'
            f'<filegroup hidden="false" open="true">'
            f'<legendlayerfile layerid="{layer_id}" visible="1"/>'
            f'</filegroup></legendlayer>'
        )
        order_items.append(f'<item>{layer_id}</item>')

    legend = "<legend>" + "".join(legend_items) + "</legend>"
    maplayers = '<projectlayers>' + "".join(layer_blocks) + '</projectlayers>'
    layer_order = '<layer-tree-canvas><custom-order enabled="0">' \
                  + "".join(order_items) + '</custom-order></layer-tree-canvas>'

    project = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<qgis version="3.34" projectname="{title}">'
        f'<title>{title}</title>'
        '<projectCrs><spatialrefsys>'
        f'<authid>EPSG:{crs_epsg}</authid>'
        '</spatialrefsys></projectCrs>'
        f'{legend}'
        f'{maplayers}'
        f'{layer_order}'
        '</qgis>'
    )

    # Reformate proprement pour lisibilité / robustesse parseur QGIS.
    try:
        pretty = minidom.parseString(project).toprettyxml(indent="  ",
                                                           encoding="UTF-8")
        return pretty.decode("utf-8")
    except Exception:
        # Si le pretty-print échoue, on rend le XML compact (toujours valide).
        return project


def layers_from_geojson_dir(geojson_dir) -> List[Tuple[str, str]]:
    """
    Inspecte un dossier de .geojson et déduit (nom, type_geometrie) pour
    chaque couche, en lisant le premier feature.

    Retourne une liste utilisable directement par build_qgs_project().
    """
    import json
    from pathlib import Path

    geojson_dir = Path(geojson_dir)
    out: List[Tuple[str, str]] = []
    for gj_file in sorted(geojson_dir.glob("*.geojson")):
        name = gj_file.stem
        geom = "polygon"
        try:
            with open(gj_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            feats = data.get("features", [])
            if feats:
                gtype = feats[0].get("geometry", {}).get("type", "").lower()
                if "line" in gtype:
                    geom = "linestring"
                elif "point" in gtype:
                    geom = "point"
                else:
                    geom = "polygon"
        except Exception:  # noqa: BLE001
            # On retombe sur le style par nom de couche
            style = LAYER_STYLES.get(name, LAYER_STYLES["_default"])
            geom = style["geom"]
        out.append((name, geom))
    return out
