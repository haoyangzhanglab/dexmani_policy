from __future__ import annotations

import warnings
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

from .image_processor import ImageProcessor
from .utils import get_interpolation, to_hw

BackboneName = Literal["resnet", "clip", "dino", "siglip", "r3m"]

RGB_BACKBONE_CONFIGS: Dict[BackboneName, Dict[str, object]] = {
    "resnet": {
        "model_name": "resnet18",
        "tune_mode": "full",
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
        "model_name": "facebook/dinov2-base",
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
        "norm_mode": "frozen_bn",
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
    effective_config = dict(cfg)

    # Pop ImageProcessor-level overrides before passing to backbone constructor.
    # The backbone classes (ResNet, DINO, CLIP, SigLIP) don't accept these params.
    image_size = cfg.pop("image_size", None)
    interpolation = cfg.pop("interpolation", None)
    image_mean = cfg.pop("image_mean", None)
    image_std = cfg.pop("image_std", None)

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

    resolved_name = getattr(backbone, "model_name", cfg.get("model_name"))
    if str(resolved_name) != str(base_cfg.get("model_name")):
        warnings.warn(
            f"build_backbone(name='{name}') resolved to model_name='{resolved_name}' "
            f"(preset default is '{base_cfg.get('model_name')}'), "
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
    if interpolation is not None:
        image_processor.interpolation = get_interpolation(interpolation)
    if image_mean is not None:
        image_processor.image_mean = torch.as_tensor(image_mean, dtype=torch.float32)
    if image_std is not None:
        image_processor.image_std = torch.as_tensor(image_std, dtype=torch.float32)
    for name in ('image_mean', 'image_std'):
        value = getattr(image_processor, name)
        if value.shape != (3,) or not torch.isfinite(value).all() or (name == 'image_std' and (value <= 0).any()):
            raise ValueError(f'{name} requires three finite values (std must be positive)')
        effective_config[name] = value.tolist()
    effective_config['image_size'] = image_processor.image_size
    effective_config['interpolation'] = image_processor.interpolation

    import inspect
    for key, parameter in inspect.signature(type(backbone)).parameters.items():
        if key not in effective_config and parameter.default is not inspect.Parameter.empty:
            effective_config[key] = parameter.default
    backbone.effective_config = effective_config
    return backbone, image_processor
