from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


def rgb_preprocessing_kwargs(dataset_config):
    return {
        "resize_hw": dataset_config.get("rgb_preprocess_size"),
        "center_crop_hw": dataset_config.get("rgb_random_crop_size"),
        "keep_uint8": dataset_config.get("rgb_keep_uint8", False),
        "float_spatial_before_uint8": dataset_config.get("rgb_color_aug") is not None,
    }


def validate_rgb_preprocessing(resize_hw, crop_hw, color_augmentation=None):
    if resize_hw is None and (crop_hw is not None or color_augmentation is not None):
        raise ValueError("RGB crop and color augmentation require rgb_preprocess_size")
    if crop_hw is not None:
        crop_h, crop_w = crop_hw
        resize_h, resize_w = resize_hw
        if not (0 < crop_h <= resize_h and 0 < crop_w <= resize_w):
            raise ValueError("RGB crop must fit within the resize dimensions")


def resize_image_transform(input_hw, output_hw, *, device=None) -> torch.Tensor:
    """FP32 resize affine for integer pixel centers and align_corners=False."""
    import torch

    sy, sx = (dst / src for src, dst in zip(input_hw, output_hw))
    return torch.tensor(
        ((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)),
        dtype=torch.float32,
        device=device,
    )


def convert_rgb_output(value: torch.Tensor, *, keep_uint8: bool) -> torch.Tensor:
    import torch

    if value.dtype.is_floating_point:
        value = value.clamp_(0, 1)
        if keep_uint8:
            value = value.mul_(255).round_().to(torch.uint8)
    if value.dtype == torch.uint8:
        value = value.to(torch.float32).div_(255.0)
    return value.contiguous()


def preprocess_rgb_train(
    rgb: np.ndarray | torch.Tensor,
    *,
    resize_hw: tuple[int, int],
    random_crop_hw: tuple[int, int] | None,
    keep_uint8: bool,
    color_augmentation: Callable | None = None,
) -> torch.Tensor:
    import torch
    import torchvision.transforms.functional as TVF

    value = torch.as_tensor(rgb).permute(0, 3, 1, 2)
    if not keep_uint8 or color_augmentation is not None:
        value = value.to(dtype=torch.float32, memory_format=torch.contiguous_format).div_(255.0)
    value = TVF.resize(value.contiguous(), list(resize_hw), antialias=True)

    if random_crop_hw is not None:
        # Share crop coordinates across the observation window.
        height, width = random_crop_hw
        top = torch.randint(0, value.shape[-2] - height + 1, (1,)).item()
        left = torch.randint(0, value.shape[-1] - width + 1, (1,)).item()
        value = TVF.crop(value, top, left, height, width)
    if color_augmentation is not None:
        value = color_augmentation(value)
    return convert_rgb_output(value, keep_uint8=keep_uint8)


def preprocess_rgb_eval(
    rgb: np.ndarray | torch.Tensor,
    *,
    resize_hw: tuple[int, int] | None,
    center_crop_hw: tuple[int, int] | None,
    keep_uint8: bool,
    float_spatial_before_uint8: bool = False,
) -> torch.Tensor:
    import torch
    import torchvision.transforms.functional as TVF

    value = torch.as_tensor(rgb)
    leading_shape = value.shape[:-3]
    value = value.reshape(-1, *value.shape[-3:]).permute(0, 3, 1, 2)
    if not keep_uint8 or float_spatial_before_uint8:
        value = value.to(dtype=torch.float32, memory_format=torch.contiguous_format).div_(255.0)
    if resize_hw is not None:
        value = TVF.resize(value.contiguous(), list(resize_hw), antialias=True)

    if center_crop_hw is not None:
        value = TVF.center_crop(value, list(center_crop_hw))
    value = convert_rgb_output(value, keep_uint8=keep_uint8)
    return value.reshape(*leading_shape, *value.shape[-3:]).contiguous()
