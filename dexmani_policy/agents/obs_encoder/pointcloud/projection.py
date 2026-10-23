"""Frustum projection into the final RGB input; no occlusion inference."""

from typing import NamedTuple

import torch


class PointProjection(NamedTuple):
    """Final image coordinates [BT,V,N,2] and FP32 confidence [BT,V,N].

    Coordinates use align_corners=False and pixel centers at integer positions.
    Valid image support is [-1,1); invalid points have zero confidence.
    The point order must match the exact cloud passed to Sparse3DEncoder.
    """

    grid_xy: torch.Tensor
    confidence: torch.Tensor


@torch.no_grad()
def projection_from_pixels(
    pixels: torch.Tensor, confidence: torch.Tensor, image_hw: tuple[int, int]
) -> PointProjection:
    """Convert final-image pixel centers into the FP32 projection contract."""
    height, width = image_hw
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive (height, width)")
    if (
        pixels.ndim != 4
        or pixels.shape[-1] != 2
        or confidence.shape != pixels.shape[:-1]
    ):
        raise ValueError("pixels/confidence require [BT,V,N,2] / [BT,V,N]")
    if pixels.device != confidence.device:
        raise ValueError("pixels and confidence must share a device")
    with torch.autocast(device_type=pixels.device.type, enabled=False):
        pixels, confidence = pixels.float(), confidence.float()
        size = pixels.new_tensor((width, height))
        valid = (
            torch.isfinite(pixels).all(-1)
            & torch.isfinite(confidence)
            & (pixels >= -0.5).all(-1)
            & (pixels < size - 0.5).all(-1)
        )
        grid = 2 * ((pixels + 0.5) / size) - 1
        # A valid pixel just inside the right edge can round to exactly +1.
        grid = grid.clamp(-1, 1 - torch.finfo(torch.float32).eps / 2)
        return PointProjection(
            torch.where(valid[..., None], grid, 0.0),
            torch.where(valid, confidence.clamp(0, 1), 0.0),
        )


@torch.no_grad()
def project_points(
    points_base: torch.Tensor,
    intrinsics: torch.Tensor,
    T_camera_from_base: torch.Tensor,
    image_transform: torch.Tensor,
    image_hw: tuple[int, int],
    *,
    confidence: torch.Tensor | None = None,
) -> PointProjection:
    """Project metric Base XYZ [BT,N,3] into final RGB input coordinates.

    intrinsics [BT,V,3,3] describes the original, undistorted pinhole image.
    T_camera_from_base [BT,V,4,4] acts on column vectors: p_cam = R p_base + t.
    Camera x/y/z point right/down/forward; points and translation use meters.
    image_transform [BT,V,3,3] is an affine raw-pixel -> final-pixel transform
    from recorded resize/in-bounds crop operations (no padding). image_hw is
    (height,width). Integer pixels are centers, normalized with align_corners=False.
    Optional confidence [BT,V,N] multiplies frustum validity; this function does
    not perform depth_consistency or infer calibration, distortion, or occlusion.
    Nonfinite cameras/points, nonpositive depth and out-of-image points are invalid.
    Geometry is FP32 and detached, including under BF16 autocast.
    """
    if points_base.ndim != 3 or points_base.shape[-1] != 3:
        raise ValueError("points_base must be [BT,N,3]")
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must be [BT,V,3,3]")
    batch, views = intrinsics.shape[:2]
    if points_base.shape[0] != batch or views == 0:
        raise ValueError("camera and point batches must match, with at least one view")
    if T_camera_from_base.shape != (batch, views, 4, 4):
        raise ValueError("T_camera_from_base must be [BT,V,4,4]")
    if image_transform.shape != (batch, views, 3, 3):
        raise ValueError("image_transform must be [BT,V,3,3]")
    if confidence is not None and confidence.shape != (
        batch,
        views,
        points_base.shape[1],
    ):
        raise ValueError("confidence must be [BT,V,N]")
    values = (points_base, intrinsics, T_camera_from_base, image_transform, confidence)
    if any(
        t is not None and (not t.is_floating_point() or t.device != points_base.device)
        for t in values
    ):
        raise ValueError("geometry must be floating point on one device")

    with torch.autocast(device_type=points_base.device.type, enabled=False):
        points = points_base.float()
        transform = T_camera_from_base.float()
        # Explicit FP32 products avoid the trainer's global high/TF32 matmul mode.
        camera = (points[:, None, :, None, :] * transform[..., None, :3, :3]).sum(-1)
        camera = camera + transform[..., None, :3, 3]
        homogeneous = (camera[..., None, :] * intrinsics.float()[..., None, :, :]).sum(
            -1
        )
        depth = camera[..., 2]
        denominator = homogeneous[..., 2:3]
        pixels = homogeneous[..., :2] / torch.where(denominator != 0, denominator, 1)
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        pixels = (pixels[..., None, :] * image_transform.float()[..., None, :2, :]).sum(
            -1
        )
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
        weight = torch.ones_like(depth) if confidence is None else confidence.float()
        weight = torch.where(valid, weight, 0.0)
        return projection_from_pixels(pixels, weight, image_hw)
