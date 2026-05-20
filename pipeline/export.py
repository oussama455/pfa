"""
pipeline/export.py — Assemblage du bundle d'export GIS QGIS
===========================================================

Prend un dossier de couches GeoJSON déjà géoréférencées et produit une
archive ZIP autonome :

    cartovec_export_<id>.zip
    ├── project.qgs                 ← projet QGIS prêt à ouvrir
    └── layers/
        ├── red_roads.shp / .shx / .dbf / .prj
        ├── water.shp / ...
        └── ...

Étapes :
    1. Lit chaque .geojson, le LISSE (smooth_geodataframe) pour éliminer
       l'effet d'escalier pixel.
    2. Écrit les shapefiles lissés dans layers/.
    3. Génère project.qgs avec des chemins RELATIFS (./layers/x.shp).
    4. Zippe le tout.

Ce module n'est utilisé qu'en mode SIG (georeference=True) : il requiert
geopandas + un backend I/O (pyogrio ou fiona). En mode pixel, l'appelant
ne doit pas l'invoquer (cf. garde-fou côté API).
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _smooth_tolerance_for_crs(crs_epsg: int) -> float:
    """
    Choisit une tolérance de lissage adaptée à l'unité du CRS.
        - EPSG:4326 (degrés)      → ~1e-4 (≈ 11 m à l'équateur)
        - CRS projeté (mètres)    → 6 m (demi-pixel à 1:50 000 / 2400 px)
    """
    from pipeline.vectorization import (
        DEFAULT_SMOOTH_TOLERANCE_DEG, DEFAULT_SMOOTH_TOLERANCE_M,
    )
    return DEFAULT_SMOOTH_TOLERANCE_DEG if crs_epsg == 4326 else DEFAULT_SMOOTH_TOLERANCE_M


def build_qgis_bundle(geojson_dir: str | Path,
                      output_zip: str | Path,
                      *,
                      crs_epsg: int = 4326,
                      title: str = "CartoVec Export",
                      smooth: bool = True,
                      smooth_tolerance: Optional[float] = None) -> Path:
    """
    Construit l'archive ZIP QGIS à partir d'un dossier de GeoJSON.

    Arguments
    ---------
    geojson_dir : dossier contenant les .geojson géoréférencés.
    output_zip  : chemin du .zip à produire.
    crs_epsg    : code EPSG des couches (4326 par défaut).
    title       : titre du projet QGIS.
    smooth      : applique le lissage Douglas-Peucker avant export.
    smooth_tolerance : surcharge la tolérance auto (unités du CRS).

    Retourne le Path du zip créé.
    Lève RuntimeError si geopandas/IO backend indisponible, ou si aucune
    couche exploitable n'est trouvée.
    """
    import geopandas as gpd  # import tardif : seulement en mode SIG
    from pipeline.vectorization import save_shapefile, smooth_geodataframe
    from pipeline.qgis_generator import build_qgs_project

    geojson_dir = Path(geojson_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    geojson_files = sorted(geojson_dir.glob("*.geojson"))
    if not geojson_files:
        raise RuntimeError(f"Aucun .geojson trouvé dans {geojson_dir}")

    if smooth_tolerance is None:
        smooth_tolerance = _smooth_tolerance_for_crs(crs_epsg)

    # Dossier temporaire de staging pour les shapefiles
    staging = output_zip.parent / f"_staging_{output_zip.stem}"
    layers_dir = staging / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    written_layers: List[Tuple[str, str]] = []   # (nom, type_geometrie)
    shp_member_files: List[Path] = []

    for gj_file in geojson_files:
        name = gj_file.stem
        try:
            gdf = gpd.read_file(gj_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[export] lecture %s échouée : %s", gj_file.name, exc)
            continue
        if gdf is None or len(gdf) == 0:
            logger.info("[export] couche %s vide — ignorée", name)
            continue

        # Forcer le CRS si absent (les GeoJSON pixel n'en ont pas)
        if gdf.crs is None:
            try:
                gdf = gdf.set_crs(epsg=crs_epsg, allow_override=True)
            except Exception:  # noqa: BLE001
                pass

        if smooth:
            gdf = smooth_geodataframe(gdf, tolerance=smooth_tolerance)
        if gdf is None or len(gdf) == 0:
            continue

        # Type de géométrie pour le style QGIS
        geom_type = "polygon"
        try:
            gt = str(gdf.geom_type.iloc[0]).lower()
            geom_type = "linestring" if "line" in gt else (
                "point" if "point" in gt else "polygon")
        except Exception:  # noqa: BLE001
            pass

        shp_path = layers_dir / f"{name}.shp"
        try:
            save_shapefile(gdf, shp_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[export] écriture shapefile %s échouée : %s", name, exc)
            continue

        written_layers.append((name, geom_type))
        # Collecte tous les fichiers compagnons du shapefile (.shp/.shx/.dbf/.prj/.cpg)
        for member in sorted(layers_dir.glob(f"{name}.*")):
            shp_member_files.append(member)

    if not written_layers:
        raise RuntimeError("Aucune couche non vide à exporter en shapefile.")

    # Génère le projet QGIS
    qgs_xml = build_qgs_project(
        written_layers,
        layers_subdir="layers",
        crs_epsg=crs_epsg,
        title=title,
        ext="shp",
    )
    qgs_path = staging / "project.qgs"
    qgs_path.write_text(qgs_xml, encoding="utf-8")

    # Zippe : project.qgs à la racine, shapefiles sous layers/
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(qgs_path, arcname="project.qgs")
        for member in shp_member_files:
            archive.write(member, arcname=f"layers/{member.name}")

    # Nettoyage du staging
    try:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    logger.info("[export] bundle QGIS créé : %s (%d couches)",
                output_zip, len(written_layers))
    return output_zip
