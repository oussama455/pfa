"""
Script de diagnostic GPU/CPU pour le PFA.

Usage :
    python scripts/test_gpu.py

Affiche :
    - Version PyTorch + CUDA
    - GPU detectee (nom, VRAM, capability)
    - Test reel : forward pass d'un U-Net resnet34 sur (1, 3, 512, 512)
      pour comparer la vitesse GPU vs CPU.

Si CUDA n'est pas dispo alors qu'une RTX 2050 est presente, c'est que
PyTorch a ete installe sans support CUDA. Reinstalle avec :
    pip uninstall torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""
from __future__ import annotations

import sys
import time

print("=" * 60)
print("  Diagnostic GPU / CPU pour le PFA")
print("=" * 60)

# 1) Imports
try:
    import torch
    print(f"\n  PyTorch       : {torch.__version__}")
except ImportError as e:
    print("  ERREUR : PyTorch n'est pas installe.")
    print("  Anaconda  : conda install pytorch torchvision -c pytorch")
    print("  Pip+CUDA  : pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

print(f"  CUDA torch    : {torch.version.cuda}")
print(f"  cuDNN         : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'indispo'}")
print(f"  CUDA dispo    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  GPUs detectees: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    [{i}] {p.name}  (compute {p.major}.{p.minor}, "
              f"{p.total_memory / 1e9:.2f} Go VRAM)")
else:
    print("  Aucun GPU CUDA detecte. Le pipeline tournera sur CPU "
          "(plus lent mais fonctionne).")

# 2) Verification SMP
try:
    import segmentation_models_pytorch as smp
    print(f"\n  SMP           : {smp.__version__}")
except ImportError:
    print("\n  WARNING : segmentation_models_pytorch (SMP) non installe.")
    print("  Installe : pip install segmentation-models-pytorch")
    sys.exit(0)

# 3) Mini benchmark forward U-Net
print("\n" + "-" * 60)
print("  Benchmark : forward pass U-Net resnet34, batch (1, 3, 512, 512)")
print("-" * 60)

model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                  in_channels=3, classes=2)
model.eval()
x = torch.randn(1, 3, 512, 512)


def bench(device, repeats=3):
    m = model.to(device)
    inp = x.to(device)
    # Warmup
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


cpu_time = bench("cpu", repeats=3)
print(f"  CPU  : {cpu_time*1000:.1f} ms / forward")

if torch.cuda.is_available():
    try:
        gpu_time = bench("cuda", repeats=10)
        print(f"  GPU  : {gpu_time*1000:.1f} ms / forward")
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n  Acceleration GPU : x{speedup:.1f}")
    except Exception as e:
        print(f"  GPU echec pendant le benchmark : {e}")

print("\nRecommandation :")
if torch.cuda.is_available():
    print("  Le pipeline utilisera automatiquement la GPU. RAS.")
else:
    print("  Le pipeline tombera sur CPU. Pour accelerer :")
    print("    1. Verifier que les drivers NVIDIA sont a jour (nvidia-smi).")
    print("    2. Reinstaller PyTorch avec support CUDA :")
    print("       pip uninstall torch torchvision")
    print("       pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu121")
