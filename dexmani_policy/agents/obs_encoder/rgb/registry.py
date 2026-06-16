import warnings
import torchvision
import torch.nn as nn
from typing import Dict, Literal, Optional, Tuple
from .image_processor import ImageProcessor
from .utils import get_interpolation, to_hw

BackboneName = Literal["resnet", "clip", "dino", "siglip", "r3m"]

RGB_BACKBONE_CONFIGS: Dict[BackboneName, Dict[str, object]] = {
    "resnet": {
        "model_name": "resnet18",
        "tune_mode": "freeze",
        "norm_mode": "group_norm",
        "global_token_type": "avg",
        "weights": None,
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "tune_mode": "freeze",
        "global_token_type": "avg",
    },
    "dino": {
        "model_name": "facebook/dinov2-base",  # shorthand: small | base
        "tune_mode": "freeze",
        "global_token_type": "avg",
    },
    "siglip": {
        "model_name": "google/siglip-base-patch16-224",
        "tune_mode": "freeze",
        "global_token_type": "avg",
    },
    "r3m": {
        "model_name": "resnet18",
        "tune_mode": "freeze",
        "norm_mode": "group_norm",
        "global_token_type": "avg",
    },
}


def resolve_resnet_weights(cfg: Dict) -> Dict:
    model_name = str(cfg["model_name"])
    weights_value = cfg.pop("weights")
    if isinstance(weights_value, str):
        weights_enum = torchvision.models.get_model_weights(model_name)
        cfg["weights"] = getattr(weights_enum, weights_value)
    else:
        cfg["weights"] = weights_value
    return cfg


def build_backbone(
    name: BackboneName,
    config: Optional[Dict] = None,
) -> Tuple[nn.Module, ImageProcessor]:

    if name not in RGB_BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported backbone name: {name}")

    base_cfg = dict(RGB_BACKBONE_CONFIGS[name])
    cfg = dict(base_cfg)
    if config:
        cfg.update(config)

    # Pop ImageProcessor-level overrides before passing to backbone constructor.
    # The backbone classes (ResNet, DINO, CLIP, SigLIP) don't accept these params.
    image_size = cfg.pop("image_size", None)
    center_crop_size = cfg.pop("center_crop_size", None)
    interpolation = cfg.pop("interpolation", None)

    if name == "resnet":
        from .resnet import ResNet
        cfg = resolve_resnet_weights(cfg)
        backbone = ResNet(**cfg)
    elif name == "clip":
        from .clip import CLIP
        backbone = CLIP(**cfg)
    elif name == "dino":
        from .dino import DINO
        backbone = DINO(**cfg)
    elif name == "siglip":
        from .siglip import SigLIP
        backbone = SigLIP(**cfg)
    elif name == "r3m":
        from .r3m import R3M
        backbone = R3M(**cfg)
    else:
        raise ValueError(f"Unsupported backbone name: {name}")

    if str(cfg.get("model_name")) != str(base_cfg.get("model_name")):
        warnings.warn(
            f"build_backbone(name='{name}') is using model_name='{cfg.get('model_name')}', "
            f"but ImageProcessor is still selected by preset name='{name}'. "
            "Please make sure the processor matches the checkpoint's resize / crop / normalization."
        )

    image_processor = ImageProcessor.from_preset(name)

    # Override preset defaults with user-specified image processing params.
    # This lets the config align the processor's target size with the dataset's
    # CPU-side preprocess output (e.g., rgb_random_crop_size), avoiding a
    # second redundant resize on GPU.
    if image_size is not None:
        image_processor.image_size = to_hw(image_size)
    if center_crop_size is not None:
        image_processor.center_crop_size = to_hw(center_crop_size)
    if interpolation is not None:
        image_processor.interpolation = get_interpolation(interpolation)

    return backbone, image_processor