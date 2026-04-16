import torchvision
import torch.nn as nn
from typing import Dict, Literal, Optional, Tuple

from .clip import CLIP
from .dino import DINO
from .resnet import ResNet
from .siglip import SigLIP
from .common.image_processor import ImageProcessor, build_image_processor

BackboneName = Literal["resnet", "clip", "dino", "siglip"]

RGB_BACKBONE_CONFIGS: Dict[BackboneName, Dict[str, object]] = {
    "resnet": {
        "model_name": "resnet18",
        "tune_mode": "full",
        "norm_mode": "group_norm",
        "global_token_type": "avg",
        "out_dim": 512,
        "weights": None,
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "tune_mode": "freeze",
        "global_token_type": "avg",
        "out_dim": 512,
    },
    "dino": {
        "model_name": "facebook/dinov2-base",
        "tune_mode": "freeze",
        "global_token_type": "avg",
        "out_dim": 512,
    },
    "siglip": {
        "model_name": "google/siglip-base-patch16-224",
        "tune_mode": "freeze",
        "global_token_type": "avg",
        "out_dim": 512,
    },
}

RGB_BACKBONE_CLASSES: Dict[str, type] = {
    "resnet": ResNet,
    "clip": CLIP,
    "dino": DINO,
    "siglip": SigLIP,
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
    name: str, config: Optional[Dict] = None
) -> Tuple[nn.Module, ImageProcessor]:
    """用 type 指定 RGB backbone，返回 (backbone, image_processor)。"""
    cfg = dict(RGB_BACKBONE_CONFIGS[name])
    if config:
        cfg.update(config)

    backbone_cls = RGB_BACKBONE_CLASSES[name]
    if name == "resnet":
        cfg = resolve_resnet_weights(cfg)

    backbone = backbone_cls(**cfg)
    image_processor = build_image_processor(name)
    return backbone, image_processor
