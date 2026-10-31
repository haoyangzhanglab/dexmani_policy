"""Frustum projection into the final RGB input; no occlusion inference."""

from typing import NamedTuple

import torch


class PointProjection(NamedTuple):
    """align_corners=False grids [BT,N,2] and FP32 confidence [BT,N]."""

    grid_xy: torch.Tensor
    confidence: torch.Tensor


@torch.no_grad()
def projection_from_pixels(
    pixels: torch.Tensor, confidence: torch.Tensor, image_hw: tuple[int, int]
) -> PointProjection:
    """Normalize pixel centers to FP32 grids with support in [-1,1)."""
    height, width = image_hw
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive (height, width)")

    with torch.autocast(device_type=pixels.device.type, enabled=False):
        pixels, confidence = pixels.float(), confidence.float()
        size = pixels.new_tensor((width, height))
        valid = (
            ((pixels >= -0.5) & (pixels < size - 0.5)).all(-1)
            & torch.isfinite(confidence)
        )
        grid = (pixels + 0.5).div_(size).mul_(2).sub_(1)
        # A valid pixel just inside the right edge can round to exactly +1.
        grid.clamp_(-1, 1 - torch.finfo(torch.float32).eps / 2)
        return PointProjection(
            torch.where(valid[..., None], grid, 0.0),
            torch.where(valid, confidence.clamp(0, 1), 0.0),
        )


@torch.no_grad()
def project_points(
    points_source: torch.Tensor,
    intrinsics: torch.Tensor,
    T_camera_from_source: torch.Tensor,
    image_transform: torch.Tensor,
    image_hw: tuple[int, int],
) -> PointProjection:
    """Project source-frame XYZ [BT,N,3] through cameras [BT,...] into FP32 grids."""
    with torch.autocast(device_type=points_source.device.type, enabled=False):
        points = points_source.float()
        transform = T_camera_from_source.float()
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
        weight = valid.float()
        return projection_from_pixels(pixels, weight, image_hw)
