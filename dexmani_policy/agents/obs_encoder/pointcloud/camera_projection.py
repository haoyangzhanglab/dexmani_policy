"""Frustum projection into the final RGB input; no occlusion inference."""

from typing import NamedTuple

import torch


class ImageProjection(NamedTuple):
    """FP32 align_corners=False grids [...,2] and Bool validity [...]."""

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
        if pixels.dtype != torch.float32:
            pixels = pixels.float()
        size = pixels.new_tensor((width, height))
        valid = ((pixels >= -0.5) & (pixels < size - 0.5)).all(-1) & valid
        grid = (pixels + 0.5).div_(size).mul_(2).sub_(1)
        # A valid pixel just inside the right edge can round to exactly +1.
        grid.clamp_(-1, 1 - torch.finfo(torch.float32).eps / 2)
        return ImageProjection(torch.where(valid[..., None], grid, 0.0), valid)


@torch.no_grad()
def project_to_image(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    T_camera_from_world: torch.Tensor,
    image_transform: torch.Tensor,
    image_hw: tuple[int, int],
) -> ImageProjection:
    """Project world XYZ [BT,N,3] through OpenCV cameras into the processed image."""
    if points_world.ndim != 3 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape [BT,N,3]")
    bt = points_world.shape[0]
    for name, value, shape in (
        ("intrinsics", intrinsics, (bt, 3, 3)),
        ("T_camera_from_world", T_camera_from_world, (bt, 3, 4)),
        ("image_transform", image_transform, (bt, 3, 3)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    with torch.autocast(device_type=points_world.device.type, enabled=False):
        points = points_world.float()
        transform = T_camera_from_world.float()
        camera_intrinsics = intrinsics.float()
        image_affine = image_transform.float()
        # Explicit FP32 products avoid TF32 rounding at image boundaries.
        camera = (points[:, :, None, :] * transform[..., None, :3, :3]).sum(-1)
        camera.add_(transform[..., None, :3, 3])

        homogeneous = (camera[..., None, :] * camera_intrinsics[..., None, :, :]).sum(-1)
        depth = camera[..., 2]
        denominator = homogeneous[..., 2:3]
        valid = (
            torch.isfinite(image_transform).all(dim=(-1, -2))[..., None]
            & torch.isfinite(homogeneous).all(-1)
            & (depth > 1e-6)
            & (denominator[..., 0].abs() > 1e-6)
        )
        # Record validity before division overwrites the homogeneous coordinates.
        pixels = homogeneous[..., :2].div_(torch.where(valid[..., None], denominator, 1))
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        pixels = (pixels[..., None, :] * image_affine[..., None, :2, :]).sum(-1)
        return pixels_to_grid(pixels, valid, image_hw)
