from typing import Literal, cast

import torch
from torch import nn

from .image_processor import ImageProcessor

RGB_ENCODER_DEFAULTS = {
    "resnet": {},
    "dino": {"model_name": "facebook/dinov2-base"},
    "siglip": {"model_name": "google/siglip-base-patch16-224"},
}


def build_rgb_encoder(
    name: Literal["resnet", "dino", "siglip"], config: dict | None = None
) -> tuple[nn.Module, ImageProcessor]:
    config = {**RGB_ENCODER_DEFAULTS[name], **(config or {})}
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


def example(name: Literal["resnet", "dino", "siglip"] = "resnet") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, processor = build_rgb_encoder(name)
    encoder.to(device).eval()
    images = torch.randint(256, (2, 2, 240, 320, 3), dtype=torch.uint8)
    rgb = cast(torch.Tensor, processor.process_images(images)).to(
        device=device, dtype=next(encoder.backbone.parameters()).dtype
    )
    with torch.no_grad():
        features = encoder(rgb)
        feature_map = encoder.tokens_to_feature_map(features["patch_tokens"], rgb.shape[-2:])
    for key, value in {**features, "feature_map": feature_map}.items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract spatial and global RGB features.")
    parser.add_argument("encoder", nargs="?", choices=RGB_ENCODER_DEFAULTS, default="resnet")
    example(parser.parse_args().encoder)
