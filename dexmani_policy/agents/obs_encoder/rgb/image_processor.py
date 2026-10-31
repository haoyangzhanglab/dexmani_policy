from collections.abc import Sequence

import torch
import torch.nn.functional as F

from dexmani_policy.common.image_util import resize_image_transform

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_PROCESSOR_PRESETS: dict[str, dict] = {
    "dino": {
        "image_size": (224, 224),
        "image_mean": IMAGENET_MEAN,
        "image_std": IMAGENET_STD,
        "interpolation": "bilinear",
    },
    "resnet": {
        "image_size": (224, 224),
        "image_mean": IMAGENET_MEAN,
        "image_std": IMAGENET_STD,
        "interpolation": "bilinear",
    },
    "siglip": {
        "image_size": (224, 224),
        "image_mean": (0.5, 0.5, 0.5),
        "image_std": (0.5, 0.5, 0.5),
        "interpolation": "bilinear",
    },
}


class ImageProcessor(torch.nn.Module):
    """Resize and normalize CHW float32 [0,1] RGB batches."""

    def __init__(
        self,
        image_size: int | Sequence[int] | None = None,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
        interpolation: str = "bilinear",
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(map(int, image_size)) if image_size is not None else None
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32), persistent=False)
        for name, value in (
            ("image_mean", self.image_mean),
            ("image_std", self.image_std),
        ):
            if value.shape != (3,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} requires three finite values")
        if (self.image_std <= 0).any():
            raise ValueError("image_std must be positive")

        self.interpolation = interpolation.lower()
        if self.interpolation not in ("nearest", "bilinear", "bicubic"):
            raise ValueError(f"Unsupported interpolation: {interpolation}")

    @classmethod
    def from_preset(cls, name: str, config: dict | None = None) -> "ImageProcessor":
        """Select preprocessing settings from the combined RGB encoder config."""
        config = config or {}
        return cls(
            **{
                key: config[key] if config.get(key) is not None else default
                for key, default in IMAGE_PROCESSOR_PRESETS[name].items()
            }
        )

    @property
    def config(self) -> dict:
        return {
            "image_size": self.image_size,
            "image_mean": self.image_mean.tolist(),
            "image_std": self.image_std.tolist(),
            "interpolation": self.interpolation,
        }

    def normalize(self, image_batch: torch.Tensor) -> torch.Tensor:
        device, dtype = image_batch.device, image_batch.dtype
        mean = self.image_mean.to(device=device, dtype=dtype).view(1, 3, 1, 1)
        std = self.image_std.to(device=device, dtype=dtype).view(1, 3, 1, 1)
        return image_batch.sub(mean).div_(std)

    def process_images(
        self,
        images: torch.Tensor,
        *,
        return_transform: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Consume [BT,3,H,W]; transforms use half-pixel resizing."""
        if return_transform:
            output_hw = self.image_size or images.shape[-2:]
            if self.interpolation == "nearest" and tuple(output_hw) != images.shape[-2:]:
                raise ValueError("tracked nearest resize does not use the half-pixel convention")
            transform = resize_image_transform(images.shape[-2:], output_hw, device=images.device)
            transform = torch.broadcast_to(transform, (images.shape[0], 3, 3))

        if self.image_size is not None and self.image_size != images.shape[-2:]:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode=self.interpolation,
                align_corners=None if self.interpolation == "nearest" else False,
            )
        images = self.normalize(images)
        return (images, transform) if return_transform else images
