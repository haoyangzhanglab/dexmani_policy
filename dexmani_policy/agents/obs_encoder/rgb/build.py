from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .image_processor import ImageProcessor

if TYPE_CHECKING:
    from .resnet import ResNetEncoder
    from .vit import ViTEncoder


@overload
def build_rgb_encoder(
    name: Literal["resnet"], config: dict | None = None,
) -> tuple[ResNetEncoder, ImageProcessor]: ...


@overload
def build_rgb_encoder(
    name: Literal["dino", "dinov3", "siglip"], config: dict | None = None,
) -> tuple[ViTEncoder, ImageProcessor]: ...


def build_rgb_encoder(
    name: Literal["resnet", "dino", "dinov3", "siglip"], config: dict | None = None
) -> tuple[ResNetEncoder | ViTEncoder, ImageProcessor]:
    config = config or {}
    if name in ("dinov3", "siglip") and config.get("image_size") is None:
        architecture = config.get("architecture") or {}
        size_key = "img_size" if name == "dinov3" else "image_size"
        config = {**config, "image_size": architecture.get(size_key)}
    processor = ImageProcessor.from_preset(name, config)
    preprocessing = processor.config
    encoder_config = {key: value for key, value in config.items() if key not in preprocessing}

    if name == "resnet":
        from .resnet import ResNetEncoder

        return ResNetEncoder(**encoder_config), processor

    from .vit import ViTEncoder

    return ViTEncoder(name, **encoder_config), processor
