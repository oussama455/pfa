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

import cv2
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


_KNOWN_ENCODERS = (
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "efficientnet-b0", "efficientnet-b1",
    "mobilenet_v2", "vgg11", "vgg16",
)


def _infer_encoder_from_state(state: dict) -> Optional[str]:
    """
    Devine l'encoder ResNet d'après les clés du state_dict.

    Méthode : on regarde la forme de `encoder.conv1.weight` (ResNet) ou
    de `encoder._conv_stem.weight` (EfficientNet) et le nombre de blocs
    dans encoder.layer1/2/3/4.

    Retourne 'resnet18'/'resnet34'/'resnet50'/'resnet101' ou None.
    """
    keys = list(state.keys()) if hasattr(state, "keys") else []
    if not any(k.startswith("encoder.conv1.weight") for k in keys):
        return None

    # Comptage des sous-blocs des layers ResNet
    def count_blocks(prefix: str) -> int:
        seen = set()
        for k in keys:
            if k.startswith(prefix):
                rest = k[len(prefix):]
                head = rest.split(".", 1)[0]
                if head.isdigit():
                    seen.add(int(head))
        return len(seen)

    n1 = count_blocks("encoder.layer1.")
    n2 = count_blocks("encoder.layer2.")
    n3 = count_blocks("encoder.layer3.")
    n4 = count_blocks("encoder.layer4.")
    layout = (n1, n2, n3, n4)

    # Tables de blocs des ResNet officiels
    mapping = {
        (2, 2, 2, 2):  "resnet18",
        (3, 4, 6, 3):  "resnet34",  # ou resnet50 — on désambiguïse par conv3 bottleneck
        (3, 4, 23, 3): "resnet101",
        (3, 8, 36, 3): "resnet152",
    }
    candidate = mapping.get(layout)
    if candidate is None:
        return None

    # ResNet34 (BasicBlock, 2 conv) vs ResNet50 (Bottleneck, 3 conv) :
    # on regarde si la clé `encoder.layer1.0.conv3.weight` existe.
    if candidate == "resnet34":
        for k in keys:
            if k.startswith("encoder.layer1.0.conv3"):
                return "resnet50"
        return "resnet34"
    return candidate


def load_weights(model, weights_path: str | Path,
                 device: Optional[str] = None,
                 *,
                 strict: bool = False,
                 auto_rebuild: bool = True):
    """
    Charge des poids sauvegardes (.pth) avec auto-détection GPU/CPU.

    Robustesse :
        - Tente d'abord `weights_only=True` (sécurité PyTorch 2.4+),
          retombe sur `weights_only=False` si le fichier est plus ancien.
        - Extrait `state_dict` si le checkpoint est un dict imbriqué
          (formats Lightning, custom training scripts, ...).
        - strict=False par défaut : un mismatch n'écrit qu'un log clair
          (missing / unexpected / shape mismatch) au lieu de crasher
          le worker. Le modèle revient même si quelques têtes diffèrent.
        - auto_rebuild=True : si le state_dict pointe clairement vers un
          autre encoder (ex. checkpoint resnet50 chargé sur smp.Unet
          resnet34), on tente de rebuild le modèle avec le bon encoder.

    Lève FileNotFoundError uniquement si le fichier manque ; tout autre
    problème (clé manquante, shape) est logué et retourné gracieusement.
    """
    _require_torch()
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    dev = device or get_device(verbose=False)

    # ── 1) Lire le checkpoint ────────────────────────────────────────────────
    try:
        state = torch.load(str(weights_path), map_location=dev, weights_only=True)
    except Exception:
        # Checkpoint plus ancien ou contient des objets pickled non-tensor.
        # On retombe sur weights_only=False, avec avertissement.
        print(f"  [load_weights] Chargement non sécurisé pour compat : {weights_path.name}")
        state = torch.load(str(weights_path), map_location=dev, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(
            f"Le checkpoint {weights_path.name} n'est pas un state_dict "
            f"(type={type(state).__name__})."
        )

    # ── 2) Auto-rebuild si l'encoder du checkpoint diffère du modèle courant ─
    if auto_rebuild:
        ckpt_encoder = _infer_encoder_from_state(state)
        current_encoder = getattr(getattr(model, "encoder", None),
                                    "_get_name", lambda: None)()
        # Normalisation case-insensitive
        current_enc_str = (current_encoder or "").lower() if current_encoder else ""
        if (ckpt_encoder and ckpt_encoder in _KNOWN_ENCODERS
                and ckpt_encoder.lower() not in current_enc_str):
            print(f"  [load_weights] Encoder du checkpoint = '{ckpt_encoder}', "
                  f"modèle courant = '{current_encoder}' → rebuild.")
            # Récupère le nombre de classes du modèle courant pour le rebuild
            try:
                # smp.Unet expose model.segmentation_head[0].out_channels
                n_classes = model.segmentation_head[0].out_channels
            except Exception:
                n_classes = 2  # fallback raisonnable
            model = smp.Unet(
                encoder_name=ckpt_encoder,
                encoder_weights=None,   # on va charger les poids
                in_channels=3,
                classes=n_classes,
                activation=None,        # cohérent avec inference argmax
            )
            model.to(dev)

    # ── 3) Chargement défensif ───────────────────────────────────────────────
    try:
        result = model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        # Capture les shape mismatch (typiquement classes différentes)
        print(f"  [load_weights] RuntimeError lors du chargement : {exc}")
        print(f"  [load_weights] Repli sur strict=False pour ne pas crasher.")
        result = model.load_state_dict(state, strict=False)

    # PyTorch retourne NamedTuple (missing_keys, unexpected_keys) en mode
    # non-strict. On log un résumé exploitable plutôt que le mur de clés brut.
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        # Regroupe les missing/unexpected par préfixe pour rester lisible.
        def head(k: str) -> str:
            parts = k.split(".")
            return ".".join(parts[:3]) if len(parts) >= 3 else k

        from collections import Counter
        if missing:
            top_missing = Counter(head(k) for k in missing).most_common(5)
            print(f"  [load_weights] {len(missing)} clés MANQUANTES dans le checkpoint "
                  f"(top préfixes) : {top_missing}")
        if unexpected:
            top_unexpected = Counter(head(k) for k in unexpected).most_common(5)
            print(f"  [load_weights] {len(unexpected)} clés EN TROP dans le checkpoint "
                  f"(top préfixes) : {top_unexpected}")
        print(f"  [load_weights] Le modèle continue avec des poids partiels — "
              f"les couches manquantes restent à leur init aléatoire/ImageNet.")

    model.to(dev).eval()
    return model

# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# U-Net SMP a 5 niveaux de downsample → l'entrée doit être un multiple de 32
# en H ET en W, sinon device-side assert dans le bloc de pooling.
UNET_DOWNSAMPLE = 32


def _pad_to_multiple(image: np.ndarray, multiple: int = UNET_DOWNSAMPLE
                      ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pad une image RGB (H×W×3) en bas et à droite pour que H et W soient
    multiples de `multiple`. Retourne (image_paddée, (top, bottom, left, right))
    pour pouvoir recropper le masque de sortie.
    """
    h, w = image.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    bottom = new_h - h
    right  = new_w - w
    if bottom == 0 and right == 0:
        return image, (0, 0, 0, 0)
    padded = cv2.copyMakeBorder(image, 0, bottom, 0, right,
                                  borderType=cv2.BORDER_REFLECT_101)
    return padded, (0, bottom, 0, right)


def _to_tensor(image_rgb: np.ndarray, device: str):
    """RGB uint8 HxWx3 -> tensor float CxHxW normalise ImageNet, sur le device."""
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device)


def _safe_forward(model, image_rgb: np.ndarray, device: str):
    """
    Forward défensif : pad l'image à multiple de 32, push sur device, gère
    les RuntimeError CUDA (OOM, kernel asynchrone, unknown error) en faisant :
        1. torch.cuda.empty_cache()
        2. log warning
        3. retry sur CPU
    Retourne (logits, padding_info, device_used).
    """
    import cv2 as _cv2  # local pour éviter circular si torch absent

    padded, padding = _pad_to_multiple(image_rgb, UNET_DOWNSAMPLE)
    tensor = _to_tensor(padded, device)

    try:
        with torch.no_grad():
            logits = model(tensor)
        return logits, padding, device
    except RuntimeError as exc:
        msg = str(exc)
        is_cuda_err = ("CUDA" in msg or "cuda" in msg
                        or "device-side" in msg or "out of memory" in msg)
        print(f"  [_safe_forward] RuntimeError ({type(exc).__name__}): {msg[:200]}")
        if is_cuda_err and device != "cpu":
            print("  [_safe_forward] → empty_cache + repli sur CPU")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            model.to("cpu").eval()
            tensor = _to_tensor(padded, "cpu")
            with torch.no_grad():
                logits = model(tensor)
            return logits, padding, "cpu"
        raise


def _crop_padding(arr: np.ndarray, padding: tuple[int, int, int, int]
                   ) -> np.ndarray:
    """Recrop d'une image 2D selon (top, bottom, left, right)."""
    top, bottom, left, right = padding
    h, w = arr.shape[:2]
    return arr[top:h - bottom if bottom else h,
               left:w - right if right else w]


def predict_mask(model, image_rgb: np.ndarray,
                 device: Optional[str] = None,
                 target_class: int = 1) -> np.ndarray:
    """Masque binaire (uint8 0/255) pour la classe target_class."""
    _require_torch()
    dev = device or get_device(verbose=False)
    model.to(dev).eval()
    logits, padding, _used = _safe_forward(model, image_rgb, dev)
    preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    preds = _crop_padding(preds, padding)
    return ((preds == target_class) * 255).astype(np.uint8)


def predict_multi_class(model, image_rgb: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         device: Optional[str] = None,
                         include_class_zero: bool = True) -> dict:
    """
    Un masque uint8 par classe, retourne {nom: masque}.

    Robustesse :
        - L'entrée est paddée à multiple de 32 (refléxion sur les bords)
          AVANT d'aller sur le device — évite les device-side assert sur
          les blocs MaxPool/Upsample du U-Net.
        - L'inférence est wrappée par _safe_forward qui retombe sur CPU
          si une RuntimeError CUDA est levée.

    class_names : un nom par classe DANS L'ORDRE DES INDEX du modele.
    """
    _require_torch()
    dev = device or get_device(verbose=False)
    model.to(dev).eval()

    # Garde-fou : downscale agressif si l'image est trop grande pour le GPU.
    # 2400 px reste l'idéal sur 4 Go VRAM ; au-delà on retombe sur CPU/OOM.
    h, w = image_rgb.shape[:2]
    if max(h, w) > 2400 and dev == "cuda":
        import cv2 as _cv2
        scale = 2400 / max(h, w)
        image_rgb = _cv2.resize(
            image_rgb, (int(round(w * scale)), int(round(h * scale))),
            interpolation=_cv2.INTER_AREA,
        )
        print(f"  [predict_multi_class] downscale GPU {w}×{h} -> "
              f"{image_rgb.shape[1]}×{image_rgb.shape[0]}")

    logits, padding, _used = _safe_forward(model, image_rgb, dev)
    preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    preds = _crop_padding(preds, padding)

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
