#!/usr/bin/env bash
# =====================================================================
# Clone les repos open source utilisés par le PFA
# Usage : bash scripts/clone_sources.sh
# =====================================================================
set -e

# Déterminer le dossier racine du projet
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$ROOT_DIR/external"

mkdir -p "$EXT_DIR"
cd "$EXT_DIR"

echo ""
echo "==> Clonage des repos dans : $EXT_DIR"
echo ""

clone_if_missing() {
    local url="$1"
    local dir="$2"
    if [ -d "$dir" ]; then
        echo "  [skip]   $dir existe déjà"
    else
        echo "  [clone]  $url"
        git clone --depth 1 "$url" "$dir"
    fi
}

# 1. Coeur de la vectorisation de cartes (soduco)
clone_if_missing \
    "https://github.com/soduco/Benchmark_historical_map_vectorization.git" \
    "soduco_vectorization"

# 2. Segmentation de routes (U-Net + ResNet)
clone_if_missing \
    "https://github.com/farhad-dalirani/Satellite-Imagery-Road-Segmentation.git" \
    "road_segmentation"

# 3. Référence pour intégration Leaflet dans Django (exemple)
clone_if_missing \
    "https://github.com/makinacorpus/django-leaflet.git" \
    "django-leaflet-ref"

echo ""
echo "==> Terminé. Repos clonés :"
ls -1 "$EXT_DIR"
echo ""
echo "Prochaine étape : consulter les README de chaque repo dans external/"
