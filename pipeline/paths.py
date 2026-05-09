"""
shared/paths.py — Absolute path manager for the CartoVec multi-package project.

WHY THIS MODULE EXISTS:
    In a multi-package project (pipeline/, training/, webapp/, shared/),
    every package resolves relative paths from *its own working directory*.
    This causes "FileNotFoundError" when, e.g., training/ writes a model
    to ../../models/ and pipeline/ reads from ./models/ — they resolve
    to different absolute paths depending on where Python was launched.

    This module anchors *all* project paths to a single PROJECT_ROOT,
    detected automatically from this file's location, and exposes them as
    pathlib.Path objects. Every package imports from shared.paths and uses
    the same absolute references.

USAGE (from any package):
    from shared.paths import Paths

    model_path = Paths.models / "unet_tunis.pth"     # always absolute
    output_dir = Paths.processed / "map_42"
    output_dir.mkdir(parents=True, exist_ok=True)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Project root detection
# ─────────────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """
    Walk up from this file's directory until we find the root marker.

    The root marker is any of:
        - .git/              (git repository root)
        - environment.yml    (conda env definition at project root)
        - pyproject.toml

    Falls back to the parent of shared/ if no marker is found (useful in
    packaged / installed contexts where .git is absent).
    """
    # shared/paths.py → shared/ → project_root
    candidate = Path(__file__).resolve().parent.parent

    markers = [".git", "environment.yml", "environment_cuda.yml", "pyproject.toml"]

    for _ in range(6):  # max 6 levels up — avoids infinite loop
        if any((candidate / m).exists() for m in markers):
            return candidate
        parent = candidate.parent
        if parent == candidate:          # filesystem root reached
            break
        candidate = parent

    # Fallback: two levels above this file
    return Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Core paths class
# ─────────────────────────────────────────────────────────────────────────────

class _ProjectPaths:
    """
    Singleton-style namespace for all project absolute paths.

    All attributes are pathlib.Path objects. They are *defined* here but
    *not* created on disk — call .ensure() or mkdir(parents=True) yourself
    when needed. This avoids polluting the filesystem on import.

    Directory layout expected:
        <project_root>/
        ├── shared/
        ├── data_collection/
        ├── training/
        ├── pipeline/
        ├── webapp/
        │   └── media/
        │       ├── uploads/          ← raw raster uploads
        │       └── processed/        ← GeoJSON / Shapefile outputs
        ├── models/                   ← .pth weight files
        ├── data/
        │   ├── raw/                  ← original raster maps
        │   ├── tiles/                ← 512×512 training tiles
        │   ├── masks/                ← binary ground-truth masks
        │   └── processed/            ← pipeline outputs (mirrors media/)
        └── logs/
    """

    def __init__(self) -> None:
        self._root: Optional[Path] = None

    # ── Root ─────────────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Absolute project root — computed once, then cached."""
        if self._root is None:
            self._root = _find_project_root()
        return self._root

    # ── Package directories ──────────────────────────────────────────────────

    @property
    def shared(self) -> Path:
        return self.root / "shared"

    @property
    def pipeline(self) -> Path:
        return self.root / "pipeline"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def data_collection(self) -> Path:
        return self.root / "data_collection"

    @property
    def webapp(self) -> Path:
        return self.root / "webapp"

    # ── Model weights ────────────────────────────────────────────────────────

    @property
    def models(self) -> Path:
        """Directory for .pth model weight files."""
        return self.root / "models"

    @property
    def unet_weights(self) -> Path:
        """Default U-Net weights path. Override via UNET_WEIGHTS env var."""
        env_override = os.environ.get("UNET_WEIGHTS")
        if env_override:
            p = Path(env_override)
            if not p.is_absolute():
                p = self.root / p
            return p
        return self.models / "unet_tunis.pth"

    # ── Data ─────────────────────────────────────────────────────────────────

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_maps(self) -> Path:
        """Original raster map images (input to pipeline)."""
        return self.data / "raw"

    @property
    def tiles(self) -> Path:
        """512×512 training/inference tiles."""
        return self.data / "tiles"

    @property
    def masks(self) -> Path:
        """Ground-truth binary masks (same tile structure as tiles/)."""
        return self.data / "masks"

    @property
    def processed(self) -> Path:
        """Pipeline outputs: GeoJSON, Shapefiles, visualizations."""
        return self.data / "processed"

    # ── Web app media ────────────────────────────────────────────────────────

    @property
    def media(self) -> Path:
        """Django MEDIA_ROOT."""
        return self.webapp / "media"

    @property
    def uploads(self) -> Path:
        """Uploaded raster maps via Django form."""
        return self.media / "uploads"

    @property
    def media_processed(self) -> Path:
        """Pipeline outputs served via Django MEDIA_URL."""
        return self.media / "processed"

    # ── Logs ─────────────────────────────────────────────────────────────────

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def agent_log(self) -> Path:
        """LangGraph agent execution log."""
        return self.logs / "agent_runs.jsonl"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def ensure_all(self) -> None:
        """
        Create all required directories on disk.
        Call once at application startup (manage.py ready signal, CLI entry).
        Idempotent — safe to call multiple times.
        """
        dirs = [
            self.models,
            self.raw_maps,
            self.tiles,
            self.masks,
            self.processed,
            self.uploads,
            self.media_processed,
            self.logs,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def output_dir_for_map(self, map_id: int | str) -> Path:
        """
        Returns (and creates) the output directory for a specific map upload.

        Example:
            out = Paths.output_dir_for_map(42)
            # → <root>/data/processed/map_42/
            gdf.to_file(out / "buildings.geojson", driver="GeoJSON")
        """
        d = self.media_processed / f"map_{map_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def require_file(self, path: Path, label: str = "") -> Path:
        """
        Assert a file exists and return it.
        Raises FileNotFoundError with a helpful message if not found.

        Usage:
            weights = Paths.require_file(Paths.unet_weights, "U-Net weights")
            model = load_weights(model, weights)
        """
        path = Path(path)
        if not path.is_file():
            hint = ""
            if "unet" in str(path).lower() or "pth" in str(path).lower():
                hint = (
                    "\n  → Download pre-trained weights or train with: "
                    "python -m training.train"
                )
            elif "gdal" in str(path).lower():
                hint = (
                    "\n  → Install GDAL via conda-forge: "
                    "conda install -c conda-forge gdal"
                )
            raise FileNotFoundError(
                f"Required file not found: {path}"
                + (f" [{label}]" if label else "")
                + hint
            )
        return path

    def add_packages_to_sys_path(self) -> None:
        """
        Ensure all top-level packages are importable regardless of where
        Python was launched from.

        Call this at the entry point of CLI scripts:
            from shared.paths import Paths
            Paths.add_packages_to_sys_path()
            from pipeline.pipeline import run_pipeline  # now always works
        """
        packages = [
            self.root,          # project root → imports: pipeline.xxx, shared.xxx
            self.shared,        # shared/ itself → import paths (less common)
        ]
        for p in packages:
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)

    def __repr__(self) -> str:
        return f"<ProjectPaths root={self.root}>"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance — import this throughout the project
# ─────────────────────────────────────────────────────────────────────────────

Paths = _ProjectPaths()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Windows GDAL DLL resolution
# ─────────────────────────────────────────────────────────────────────────────

def configure_gdal_windows() -> None:
    """
    On Windows, add the conda-forge GDAL DLL directory to the DLL search
    path so that rasterio / GDAL imports don't raise:
        OSError: [WinError 126] The specified module could not be found

    Reads CONDA_ENV_PATH from .env or the environment. Falls back to
    auto-detecting the active conda prefix via os.environ["CONDA_PREFIX"].

    Call this at the TOP of settings.py and pipeline entry points:
        from shared.paths import configure_gdal_windows
        configure_gdal_windows()
    """
    import platform
    if platform.system() != "Windows":
        return  # noop on Linux / macOS

    import os
    from dotenv import load_dotenv

    # Load .env from project root (silently skip if absent)
    load_dotenv(Paths.root / ".env")

    conda_env = (
        os.environ.get("CONDA_ENV_PATH")
        or os.environ.get("CONDA_PREFIX")
    )
    if not conda_env:
        import warnings
        warnings.warn(
            "configure_gdal_windows(): CONDA_ENV_PATH not set. "
            "Set it in .env or as an environment variable to avoid GDAL DLL errors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    gdal_bin = Path(conda_env) / "Library" / "bin"
    if not gdal_bin.is_dir():
        import warnings
        warnings.warn(
            f"configure_gdal_windows(): GDAL bin dir not found: {gdal_bin}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    # Register the DLL directory (Python 3.8+ API)
    os.add_dll_directory(str(gdal_bin))

    # Also prepend to PATH for subprocesses (gdalinfo, ogr2ogr, etc.)
    os.environ["PATH"] = str(gdal_bin) + os.pathsep + os.environ.get("PATH", "")

    # Detect the exact GDAL DLL filename (varies: gdal.dll, gdal310.dll, …)
    gdal_dlls = sorted(gdal_bin.glob("gdal*.dll"))
    if gdal_dlls:
        os.environ.setdefault("GDAL_LIBRARY_PATH", str(gdal_dlls[0]))
    geos_dll = gdal_bin / "geos_c.dll"
    if geos_dll.exists():
        os.environ.setdefault("GEOS_LIBRARY_PATH", str(geos_dll))
