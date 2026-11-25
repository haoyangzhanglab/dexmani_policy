"""Frustum projection into the final RGB input; no occlusion inference."""

from typing import NamedTuple

import torch


class ImageProjection(NamedTuple):
    """FP32 align_corners=False grids [...,2] and Bool validity [...]."""

    grid_xy: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def project_to_image(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_transform: torch.Tensor,
    image_hw: tuple[int, int],
) -> ImageProjection:
    """Project XYZ [BT,N,3] using extrinsics from the point frame to OpenCV cameras."""
    height, width = image_hw
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive (height, width)")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [BT,N,3]")
    bt = points.shape[0]
    for name, value, shape in (
        ("intrinsics", intrinsics, (bt, 3, 3)),
        ("extrinsics", extrinsics, (bt, 3, 4)),
        ("image_transform", image_transform, (bt, 3, 3)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.float()
        transform = extrinsics.float()
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
        size = pixels.new_tensor((width, height))
        valid &= ((pixels >= -0.5) & (pixels < size - 0.5)).all(-1)
        grid = (pixels + 0.5).div_(size).mul_(2).sub_(1)
        # A valid pixel just inside the right edge can round to exactly +1.
        grid.clamp_(-1, 1 - torch.finfo(torch.float32).eps / 2)
        return ImageProjection(torch.where(valid[..., None], grid, 0.0), valid)
