from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

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


def to_rgb_tensor(images: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert integer [0, 255] or floating-point [0, 1] RGB to contiguous CHW float32."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    elif not isinstance(images, torch.Tensor):
        raise TypeError(f"Unsupported input type: {type(images)}")
    if images.ndim < 3:
        raise ValueError("images should have at least 3 dims.")

    if images.shape[-3] != 3:
        if images.shape[-1] != 3:
            raise ValueError("Cannot infer RGB channel dimension.")
        images = images.movedim(-1, -3)

    if images.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        images = images.to(torch.float32).div_(255.0)
    else:
        images = images.to(torch.float32)

    return images.contiguous()


class ImageProcessor:
    """Resize and normalize uint8 or floating-point RGB for policy inputs."""

    def __init__(
        self,
        image_size: int | Sequence[int] | None = None,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
        interpolation: str = "bilinear",
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = (
            tuple(map(int, image_size)) if image_size is not None else None
        )
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32)
        self.image_std = torch.tensor(image_std, dtype=torch.float32)
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
        self.normalization_cache = {}

    def __getstate__(self):
        # Device copies are disposable constants, not serialized model state.
        return {**self.__dict__, "normalization_cache": {}}

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
        key = (device, dtype)
        if key not in self.normalization_cache:
            self.normalization_cache[key] = (
                self.image_mean.to(device=device, dtype=dtype).view(1, 3, 1, 1),
                self.image_std.to(device=device, dtype=dtype).view(1, 3, 1, 1),
            )
        mean, std = self.normalization_cache[key]
        return image_batch.sub(mean).div_(std)

    def process_images(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize RGB; floating-point inputs must already be finite and in [0,1]."""
        images = to_rgb_tensor(images)
        leading_shape = images.shape[:-3]
        flat_images = images.reshape(-1, *images.shape[-3:])

        if self.image_size is not None and self.image_size != images.shape[-2:]:
            flat_images = F.interpolate(
                flat_images,
                size=self.image_size,
                mode=self.interpolation,
                align_corners=None if self.interpolation == "nearest" else False,
            )
        flat_images = self.normalize(flat_images)
        return flat_images.reshape(*leading_shape, *flat_images.shape[1:])
