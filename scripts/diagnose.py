"""
scripts/diagnose.py — diagnostic unifie de l'environnement CartoVec.

Remplace les anciens scripts/diagnose_env.py (imports systeme) et
scripts/test_gpu.py (benchmark PyTorch).

Usage :
    python scripts/diagnose.py                # equivalent --check all
    python scripts/diagnose.py --check env    # imports systeme uniquement
    python scripts/diagnose.py --check gpu    # GPU/CUDA + benchmark uniquement
    python scripts/diagnose.py --check all    # les deux (defaut)
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback


def header(title):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def test_import(label, code, hint=""):
    try:
        exec(code, {"__name__": "__main__"})
        print(f"  OK   {label}")
        return True
    except Exception as exc:
        print(f"  FAIL {label}")
        print(f"       {type(exc).__name__}: {exc}")
        if hint:
            print(f"       -> {hint}")
        return False


def check_env():
    """Diagnostic des imports systeme (anciennement diagnose_env.py)."""
    header("Systeme")
    print(f"  OS       : {platform.system()} {platform.release()}")
    print(f"  Python   : {platform.python_version()}  ({sys.executable})")
    print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', '(aucun)')}")

    header("Modules critiques")
    test_import("numpy",          "import numpy")
    test_import("matplotlib",     "import matplotlib")
    ok_cv2 = test_import(
        "cv2 (OpenCV)",
        "import cv2; assert cv2.__version__",
        hint="pip uninstall opencv-python opencv-python-headless puis "
             "pip install opencv-python==4.10.0.84")
    test_import("skimage", "import skimage")
    test_import("shapely", "import shapely; from shapely.geometry import Polygon")
    ok_rio = test_import(
        "rasterio",
        "import rasterio; from rasterio import features",
        hint="conda create -n pfa python=3.10 -c conda-forge "
             "--strict-channel-priority -y rasterio")
    ok_pyogrio = test_import(
        "pyogrio",
        "import pyogrio; from pyogrio import _io",
        hint="pip install fiona (le pipeline detecte automatiquement)")
    ok_fiona = test_import("fiona", "import fiona")
    ok_gpd = test_import("geopandas", "import geopandas")
    test_import("torch (PyTorch)",
                 "import torch; print(f'         CUDA dispo : "
                 "{torch.cuda.is_available()}')")
    return {
        "cv2": ok_cv2, "rasterio": ok_rio, "pyogrio": ok_pyogrio,
        "fiona": ok_fiona, "geopandas": ok_gpd,
    }


def check_gpu():
    """Diagnostic GPU + benchmark U-Net (anciennement test_gpu.py)."""
    header("Diagnostic GPU / CPU")
    try:
        import torch
        print(f"  PyTorch       : {torch.__version__}")
        print(f"  CUDA torch    : {torch.version.cuda}")
        cudnn_v = (torch.backends.cudnn.version()
                   if torch.backends.cudnn.is_available() else "indispo")
        print(f"  cuDNN         : {cudnn_v}")
        print(f"  CUDA dispo    : {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch non installe.")
        return

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  GPUs detectees: {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}  (compute {p.major}.{p.minor}, "
                  f"{p.total_memory / 1e9:.2f} Go VRAM)")
    else:
        print("  Aucun GPU CUDA detecte. Le pipeline tournera sur CPU.")

    try:
        import segmentation_models_pytorch as smp
        print(f"  SMP           : {smp.__version__}")
    except ImportError:
        print("  segmentation_models_pytorch absent. Skip benchmark.")
        return

    header("Benchmark U-Net resnet34, batch (1, 3, 512, 512)")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=3, classes=2)
    model.eval()
    x = torch.randn(1, 3, 512, 512)

    def bench(device, repeats=3):
        m = model.to(device); inp = x.to(device)
        with torch.no_grad():
            _ = m(inp)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = m(inp)
                if device == "cuda":
                    torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats

    cpu_t = bench("cpu", 3)
    print(f"  CPU  : {cpu_t * 1000:.1f} ms / forward")
    if torch.cuda.is_available():
        try:
            gpu_t = bench("cuda", 10)
            print(f"  GPU  : {gpu_t * 1000:.1f} ms / forward")
            if gpu_t > 0:
                print(f"\n  Acceleration GPU : x{cpu_t / gpu_t:.1f}")
        except Exception as exc:
            print(f"  GPU echec : {exc}")

    print("\nRecommandation :")
    if torch.cuda.is_available():
        print("  Le pipeline utilisera la GPU automatiquement.")
    else:
        print("  Le pipeline tombera sur CPU. Pour accelerer :")
        print("    pip uninstall torch torchvision")
        print("    pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")


def main():
    p = argparse.ArgumentParser(description="Diagnostic environnement CartoVec")
    p.add_argument("--check", choices=["env", "gpu", "all"], default="all")
    args = p.parse_args()

    if args.check in ("env", "all"):
        check_env()
    if args.check in ("gpu", "all"):
        check_gpu()

    header("Termine")


if __name__ == "__main__":
    main()
