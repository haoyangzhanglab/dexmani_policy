from typing import Literal

from torch import nn

from .image_processor import ImageProcessor

def prepare_initialization(cfg) -> None:
    """Resolve HF architecture once at the fresh-training boundary."""
    if cfg.policy_name not in {"dp", "sat", "dqrise"}:
        return
    name = cfg.agent.get("rgb_backbone_name", "dino")
    if name == "resnet":
        return
    source = cfg.initialization.get("rgb")
    rgb = dict(cfg.agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is None:
        if source is None:
            raise ValueError("Fresh HF RGB requires initialization.rgb or an explicit architecture")
        from transformers import AutoConfig, SiglipVisionConfig

        config_class = SiglipVisionConfig if name == "siglip" else AutoConfig
        rgb["architecture"] = config_class.from_pretrained(source).to_dict()
    from omegaconf import open_dict

    with open_dict(cfg.agent):
        cfg.agent.rgb_backbone_config = rgb


def build_rgb_encoder(
    name: Literal["resnet", "dino", "siglip"], config: dict | None = None
) -> tuple[nn.Module, ImageProcessor]:
    config = dict(config or {})
    processor = ImageProcessor.from_preset(name, config)
    preprocessing = processor.config
    encoder_config = {key: value for key, value in config.items() if key not in preprocessing}

    encoder: nn.Module
    if name == "resnet":
        from .resnet import ResNetEncoder

        encoder = ResNetEncoder(**encoder_config)
    else:
        from .vit import ViTEncoder

        encoder = ViTEncoder(name, **encoder_config)

    return encoder, processor
