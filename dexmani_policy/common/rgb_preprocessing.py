"""Training and deterministic evaluation RGB recipes; Torch loads only on use."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def rgb_preprocessing_kwargs(dataset_config):
    """The exact deterministic RobotDataset validation recipe."""
    validate_rgb_preprocessing(
        dataset_config.get("rgb_preprocess_size"),
        dataset_config.get("rgb_random_crop_size"),
        dataset_config.get("rgb_color_aug"),
    )
    return {
        "resize_hw": dataset_config.get("rgb_preprocess_size"),
        "center_crop_hw": dataset_config.get("rgb_random_crop_size"),
        "keep_uint8": bool(dataset_config.get("rgb_keep_uint8", False)),
        "float_spatial_before_uint8": dataset_config.get("rgb_color_aug") is not None,
    }


def validate_rgb_preprocessing(resize_hw, crop_hw, color_augmentation=None):
    if resize_hw is None and (crop_hw is not None or color_augmentation is not None):
        raise ValueError("RGB crop and color augmentation require rgb_preprocess_size")
    if crop_hw is not None and (
        len(crop_hw) != 2
        or len(resize_hw) != 2
        or any(not 0 < crop <= size for crop, size in zip(crop_hw, resize_hw))
    ):
        raise ValueError("RGB crop must fit within the resize dimensions")


def raw_rgb_tensor(rgb, *, ndim=None):
    """Validate raw uint8 HWC input without copying its storage."""
    import torch

    value = torch.from_numpy(rgb) if isinstance(rgb, np.ndarray) else rgb
    if not torch.is_tensor(value):
        raise TypeError("raw RGB must be a NumPy array or torch tensor")
    if (
        value.ndim < 3
        or value.shape[-1] != 3
        or value.dtype != torch.uint8
        or (ndim is not None and value.ndim != ndim)
    ):
        shape = "(T, H, W, 3)" if ndim == 4 else "[..., H, W, 3]"
        raise ValueError(f"raw RGB must have shape {shape} and dtype uint8")
    return value


def resize_image_transform(input_hw, output_hw, *, device=None) -> torch.Tensor:
    """Map pixel centers by an FP32 3x3 affine with (H, W) sizes and align_corners=False."""
    import torch

    if len(input_hw) != 2 or len(output_hw) != 2 or min(*input_hw, *output_hw) <= 0:
        raise ValueError("resize requires positive (height,width) sizes")
    sy, sx = (dst / src for src, dst in zip(input_hw, output_hw))
    return torch.tensor(
        ((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)),
        dtype=torch.float32,
        device=device,
    )


def convert_rgb_output(value: torch.Tensor, *, keep_uint8: bool) -> torch.Tensor:
    import torch

    if not value.dtype.is_floating_point:
        return value.contiguous()
    value = value.clamp_(0, 1)
    if keep_uint8:
        value = value.mul_(255).round_().to(torch.uint8)
    return value


def preprocess_training_rgb(
    rgb: np.ndarray | torch.Tensor,
    *,
    resize_hw: tuple[int, int],
    random_crop_hw: tuple[int, int] | None,
    keep_uint8: bool,
    color_augmentation: Callable | None = None,
    return_transform: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Share crop/color across time and track raw->output geometry; color must preserve it."""
    import torch
    import torchvision.transforms.functional as TVF

    value = raw_rgb_tensor(rgb, ndim=4).permute(0, 3, 1, 2).contiguous()
    if return_transform:
        validate_rgb_preprocessing(resize_hw, random_crop_hw)
        transform = resize_image_transform(value.shape[-2:], resize_hw, device=value.device)
    if not keep_uint8 or color_augmentation is not None:
        value = value.float().div_(255.0)
    value = TVF.resize(value, list(resize_hw), antialias=True)
    if random_crop_hw is not None:
        height, width = random_crop_hw
        top = torch.randint(0, value.shape[-2] - height + 1, (1,)).item()
        left = torch.randint(0, value.shape[-1] - width + 1, (1,)).item()
        value = TVF.crop(value, top, left, height, width)
        if return_transform:
            transform[0, 2] -= left
            transform[1, 2] -= top
    if color_augmentation is not None:
        value = color_augmentation(value)
    value = convert_rgb_output(value, keep_uint8=keep_uint8)
    return (value, transform) if return_transform else value


def preprocess_validation_rgb(
    rgb: np.ndarray | torch.Tensor,
    *,
    resize_hw: tuple[int, int] | None,
    center_crop_hw: tuple[int, int] | None,
    keep_uint8: bool,
    float_spatial_before_uint8: bool = False,
    return_transform: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Keep HWC without resize; otherwise return CHW with optional raw->output geometry."""
    import torchvision.transforms.functional as TVF
    from torchvision.transforms import InterpolationMode

    value = raw_rgb_tensor(rgb)
    if return_transform:
        validate_rgb_preprocessing(resize_hw, center_crop_hw)
        transform = resize_image_transform(
            value.shape[-3:-1],
            resize_hw or value.shape[-3:-1],
            device=value.device,
        )
    if resize_hw is None:
        if center_crop_hw is not None:
            raise ValueError("validation RGB center crop requires a resize")
        value = value.contiguous()
        return (value, transform) if return_transform else value

    leading_shape = tuple(value.shape[:-3])
    value = value.movedim(-1, -3).contiguous()
    value = value.reshape(-1, *value.shape[-3:])
    if not keep_uint8 or float_spatial_before_uint8:
        value = value.float().div_(255.0)
    value = TVF.resize(
        value,
        list(resize_hw),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    if center_crop_hw is not None:
        if return_transform:
            # Match torchvision center_crop, including round-to-even for odd gaps.
            transform[0, 2] -= int(round((value.shape[-1] - center_crop_hw[1]) / 2))
            transform[1, 2] -= int(round((value.shape[-2] - center_crop_hw[0]) / 2))
        value = TVF.center_crop(value, list(center_crop_hw))
    value = convert_rgb_output(value, keep_uint8=keep_uint8)
    value = value.reshape(*leading_shape, *value.shape[-3:]).contiguous()
    return (value, transform) if return_transform else value
