"""Saved deterministic RGB validation preprocessing; Torch loads only on use."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def rgb_preprocessing_kwargs(dataset_config):
    """The exact deterministic BaseDataset validation recipe."""
    return {
        "resize_hw": dataset_config.get("rgb_preprocess_size"),
        "center_crop_hw": dataset_config.get("rgb_random_crop_size"),
        "keep_uint8": bool(dataset_config.get("rgb_keep_uint8", False)),
        "float_spatial_before_uint8": dataset_config.get("rgb_color_aug") is not None,
    }


def raw_rgb_tensor(rgb, *, ndim=None):
    """Validate raw uint8 HWC input without copying its storage."""
    import torch

    value = torch.from_numpy(rgb) if isinstance(rgb, np.ndarray) else rgb
    if not torch.is_tensor(value):
        raise TypeError("raw RGB must be a NumPy array or torch tensor")
    if (value.ndim < 3 or value.shape[-1] != 3 or value.dtype != torch.uint8
            or (ndim is not None and value.ndim != ndim)):
        shape = "(T, H, W, 3)" if ndim == 4 else "[..., H, W, 3]"
        raise ValueError(f"raw RGB must have shape {shape} and dtype uint8")
    return value


def preprocess_validation_rgb(
    rgb: np.ndarray | torch.Tensor,
    *,
    resize_hw: tuple[int, int] | None,
    center_crop_hw: tuple[int, int] | None,
    keep_uint8: bool,
    float_spatial_before_uint8: bool = False,
) -> torch.Tensor:
    """Apply the deterministic validation RGB path to raw HWC uint8 frames.

    ``float_spatial_before_uint8`` preserves the training color-augmentation
    recipe's float interpolation, even though validation skips augmentation.
    """
    import torch
    import torchvision.transforms.functional as TVF
    from torchvision.transforms import InterpolationMode

    value = raw_rgb_tensor(rgb)
    if resize_hw is None:
        if center_crop_hw is not None:
            raise ValueError("validation RGB center crop requires a resize")
        return value.contiguous()

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
        value = TVF.center_crop(value, list(center_crop_hw))
    if value.dtype.is_floating_point:
        value = value.clamp_(0, 1)
        if keep_uint8:
            value = value.mul(255).round_().clamp_(0, 255).to(torch.uint8)
    return value.reshape(*leading_shape, *value.shape[-3:]).contiguous()
