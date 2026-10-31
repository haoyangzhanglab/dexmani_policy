from typing import Literal

from torch import nn

from .image_processor import ImageProcessor


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
