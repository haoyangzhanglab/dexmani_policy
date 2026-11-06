"""Frustum projection into the final RGB input; no occlusion inference."""

from typing import NamedTuple

import torch


class ImageProjection(NamedTuple):
    """FP32 align_corners=False grids [BT,N,2] and Bool validity [BT,N]."""

    grid_xy: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def pixels_to_grid(
    pixels: torch.Tensor, valid: torch.Tensor, image_hw: tuple[int, int]
) -> ImageProjection:
    """Normalize pixel centers to FP32 grids with support in [-1,1)."""
    height, width = image_hw
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive (height, width)")

    with torch.autocast(device_type=pixels.device.type, enabled=False):
        pixels = pixels.float()
        size = pixels.new_tensor((width, height))
        valid = (
            ((pixels >= -0.5) & (pixels < size - 0.5)).all(-1)
            & valid
        )
        grid = (pixels + 0.5).div_(size).mul_(2).sub_(1)
        # A valid pixel just inside the right edge can round to exactly +1.
        grid.clamp_(-1, 1 - torch.finfo(torch.float32).eps / 2)
        return ImageProjection(
            torch.where(valid[..., None], grid, 0.0),
            valid,
        )


@torch.no_grad()
def project_to_image(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    T_camera_from_world: torch.Tensor,
    image_transform: torch.Tensor,
    image_hw: tuple[int, int],
) -> ImageProjection:
    """Project world XYZ [BT,N,3] through OpenCV cameras into the processed image."""
    with torch.autocast(device_type=points_world.device.type, enabled=False):
        points = points_world.float()
        transform = T_camera_from_world.float()
        # Explicit FP32 products avoid TF32 rounding at image boundaries.
        camera = (points[:, :, None, :] * transform[..., None, :3, :3]).sum(-1)
        camera.add_(transform[..., None, :3, 3])

        homogeneous = (camera[..., None, :] * intrinsics.float()[..., None, :, :]).sum(-1)
        depth = camera[..., 2]
        denominator = homogeneous[..., 2:3]
        pixels = homogeneous[..., :2] / torch.where(denominator != 0, denominator, 1)

        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        pixels = (pixels[..., None, :] * image_transform.float()[..., None, :2, :]).sum(-1)

        finite_camera = (
            torch.isfinite(transform).all(dim=(-1, -2))
            & torch.isfinite(intrinsics).all(dim=(-1, -2))
            & torch.isfinite(image_transform).all(dim=(-1, -2))
        )
        valid = (
            finite_camera[..., None]
            & torch.isfinite(homogeneous).all(-1)
            & (depth > 0)
            & (denominator[..., 0] != 0)
        )
        return pixels_to_grid(pixels, valid, image_hw)
