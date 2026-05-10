"""
Segmentation sémantique par deep learning (U-Net) — GPU si dispo, sinon CPU.

S'appuie sur segmentation_models_pytorch (SMP) pour charger un U-Net
avec un encodeur ResNet pré-entraîné sur ImageNet.

Toutes les fonctions ont un parametre `device` optionnel. Si tu ne le
precises pas, get_device() detecte automatiquement :
    - 'cuda' si une carte NVIDIA + CUDA sont disponibles (RTX 2050 OK)
    - 'cpu'  sinon (fallback automatique, plus lent mais fonctionne partout)

Usage minimal :
    model  = build_unet(classes=2)
    model  = load_weights(model, "weights.pth")     # device auto
    mask   = predict_mask(model, image_rgb)         # device auto
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    import segmentation_models_pytorch as smp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch et segmentation_models_pytorch sont requis pour ce module. "
            "conda install -c conda-forge pytorch torchvision  (ou avec CUDA : "
            "pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121)"
        )


# ---------------------------------------------------------------------
# Detection automatique GPU / CPU
# ---------------------------------------------------------------------
def get_device(prefer_gpu: bool = True, *, verbose: bool = True) -> str:
    """
    Retourne le meilleur device disponible : 'cuda' si GPU NVIDIA + CUDA
    detectes, sinon 'cpu'.

    Sur l'ASUS TUF A15 Ryzen 5 RTX 2050 de Mohamed avec PyTorch+CUDA bien
    installe -> 'cuda'. Sinon (pas de GPU, ou PyTorch CPU-only) -> 'cpu'.

    prefer_gpu=False force le CPU meme si un GPU est dispo (debug, comparaison).
    verbose=True affiche les infos GPU au premier appel.
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    if prefer_gpu and torch.cuda.is_available():
        if verbose:
            try:
                name = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU detecte : {name}  (CUDA {cap[0]}.{cap[1]}, "
                      f"{mem_gb:.1f} Go VRAM)")
            except Exception:
                print("  GPU detecte (infos indisponibles)")
        return "cuda"
    if verbose:
        if prefer_gpu and not torch.cuda.is_available():
            print("  Pas de GPU CUDA detecte -> CPU (plus lent mais OK)")
        elif not prefer_gpu:
            print("  prefer_gpu=False -> CPU force")
    return "cpu"


def cuda_summary() -> dict:
    """
    Resume des capacites CUDA detectees, pour diagnostic.
    Retourne un dict serializable JSON.
    """
    if not TORCH_AVAILABLE:
        return {"torch_available": False}
    info = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        info["devices"] = []
        for i in range(info["device_count"]):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i, "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "total_memory_gb": round(p.total_memory / 1e9, 2),
            })
    return info


# ---------------------------------------------------------------------
# Modele U-Net
# ---------------------------------------------------------------------
def build_unet(*, encoder_name: str = "resnet34",
               encoder_weights: str = "imagenet",
               classes: int = 2,
               activation: str = "softmax2d",
               device: Optional[str] = None):
    """
    Construit un U-Net avec encodeur pre-entraine et le place sur le device.

    classes = 2 par defaut : fond + route (ou fond + batiment).
    Pour multi-classe (route, batiment, fond) : classes=3.

    Si device=None -> auto-detection (GPU si dispo).
    """
    _require_torch()
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
        activation=activation,
    )
    dev = device or get_device(verbose=False)
    model.to(dev)
    model.eval()
    return model


def load_weights(model, weights_path: str | Path,
                 device: Optional[str] = None):
    """Charge des poids sauvegardes (.pth) avec auto-detection GPU/CPU."""
    _require_torch()
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")
    
    dev = device or get_device(verbose=False)
    
    # التعديل هنا: إضافة weights_only=True لتجنب التحذير الأمني
    try:
        state = torch.load(str(weights_path), map_location=dev, weights_only=True)
    except Exception:
        # في حال فشل التحميل الآمن (بسبب بنية ملف قديمة)، نعود للطريقة العادية مع تنبيه
        print("  Info: Chargement en mode weights_only=False pour compatibilité.")
        state = torch.load(str(weights_path), map_location=dev, weights_only=False)
        
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        
    model.load_state_dict(state)
    model.to(dev).eval()
    return model

# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(image_rgb: np.ndarray, device: str):
    """RGB uint8 HxWx3 -> tensor float CxHxW normalise ImageNet, sur le device."""
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device)


def predict_mask(model, image_rgb: np.ndarray,
                 device: Optional[str] = None,
                 target_class: int = 1) -> np.ndarray:
    """Masque binaire (uint8 0/255) pour la classe target_class."""
    _require_torch()
    dev = device or get_device(verbose=False)
    model.to(dev).eval()
    tensor = _to_tensor(image_rgb, dev)
    with torch.no_grad():
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return ((preds == target_class) * 255).astype(np.uint8)


def predict_multi_class(model, image_rgb: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         device: Optional[str] = None,
                         include_class_zero: bool = True) -> dict:
    """
    Un masque uint8 par classe, retourne {nom: masque}.

    class_names : un nom par classe DANS L'ORDRE DES INDEX du modele
        (0, 1, 2, ...). Si la longueur est num_classes-1, on suppose que
        tu as oublie la classe 0 et on la nomme 'background' (compat
        retro avec l'ancien usage).

    include_class_zero :
        - True (defaut) : on retourne aussi le masque de la classe 0
          (typiquement 'background' — utile pour debug ou pour le filtrer
          dans l'appelant).
        - False : on saute la classe 0 (ancien comportement).
    """
    _require_torch()
    dev = device or get_device(verbose=False)
    model.to(dev).eval()
    tensor = _to_tensor(image_rgb, dev)
    with torch.no_grad():
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    n_classes = logits.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    elif len(class_names) == n_classes - 1:
        # Compat retro : on suppose que la classe 0 = background
        class_names = ["background"] + list(class_names)
    elif len(class_names) != n_classes:
        raise ValueError(
            f"len(class_names)={len(class_names)} ne correspond pas au "
            f"nombre de classes du modele ({n_classes})."
        )
    start = 0 if include_class_zero else 1
    return {name: ((preds == idx) * 255).astype(np.uint8)
            for idx, name in enumerate(class_names) if idx >= start}
