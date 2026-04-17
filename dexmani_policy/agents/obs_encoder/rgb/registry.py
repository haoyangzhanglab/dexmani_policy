import warnings
import torchvision
import torch.nn as nn
from typing import Dict, Literal, Optional, Tuple, Type
from .common.image_processor import ImageProcessor

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


def get_backbone_cls(name: BackboneName) -> Type[nn.Module]:
    if name == "resnet":
        from .resnet import ResNet
        return ResNet
    if name == "clip":
        from .clip import CLIP
        return CLIP
    if name == "dino":
        from .dino import DINO
        return DINO
    if name == "siglip":
        from .siglip import SigLIP
        return SigLIP
    raise ValueError(f"Unsupported backbone name: {name}")


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

    backbone_cls = get_backbone_cls(name)
    if name == "resnet":
        cfg = resolve_resnet_weights(cfg)

    backbone = backbone_cls(**cfg)

    if str(cfg.get("model_name")) != str(base_cfg.get("model_name")):
        warnings.warn(
            f"build_backbone(name='{name}') is using model_name='{cfg.get('model_name')}', "
            f"but ImageProcessor is still selected by preset name='{name}'. "
            "Please make sure the processor matches the checkpoint's resize / crop / normalization."
        )

    image_processor = ImageProcessor.from_preset(name)
    return backbone, image_processor