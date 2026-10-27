from collections.abc import Sequence

import numpy as np
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


class ImageProcessor(torch.nn.Module):
    """Resize and normalize uint8 or floating-point RGB for policy inputs."""

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
        images: np.ndarray | torch.Tensor,
        *,
        return_transform: bool = False,
        input_transform: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Normalize RGB; float inputs must be finite and in [0,1].

        return_transform adds raw->final matrices [...,3,3], matching image leading dimensions.
        input_transform broadcasts to [...,3,3]; omission treats input pixels as raw.
        Tracked resizing uses bilinear/bicubic with align_corners=False.
        """
        if input_transform is not None and not return_transform:
            raise ValueError("input_transform requires return_transform=True")
        images = to_rgb_tensor(images)
        leading_shape = images.shape[:-3]
        flat_images = images.reshape(-1, *images.shape[-3:])
        if return_transform:
            output_hw = self.image_size or images.shape[-2:]
            if self.interpolation == "nearest" and tuple(output_hw) != images.shape[-2:]:
                raise ValueError("tracked nearest resize does not use the half-pixel convention")
            transform = resize_image_transform(images.shape[-2:], output_hw, device=images.device)
            if input_transform is not None:
                if input_transform.shape[-2:] != (3, 3):
                    raise ValueError("input_transform must end in [3,3]")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=images.device.type, enabled=False),
                ):
                    prior = input_transform.to(device=images.device, dtype=torch.float32)
                    # Geometry must not inherit the trainer's global TF32 matmul precision.
                    transform = (transform[:, :, None] * prior[..., None, :, :]).sum(-2)
            transform = torch.broadcast_to(transform, (*leading_shape, 3, 3))

        if self.image_size is not None and self.image_size != images.shape[-2:]:
            flat_images = F.interpolate(
                flat_images,
                size=self.image_size,
                mode=self.interpolation,
                align_corners=None if self.interpolation == "nearest" else False,
            )
        flat_images = self.normalize(flat_images)
        images = flat_images.reshape(*leading_shape, *flat_images.shape[1:])
        return (images, transform) if return_transform else images
