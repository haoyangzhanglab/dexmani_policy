import math

import torch
import torch.nn as nn


def resize_crop_transform(
    original_hw: tuple[int, int],
    resized_hw: tuple[int, int],
    crop_top_left: tuple[int, int] = (0, 0),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    scale_y = resized_hw[0] / original_hw[0]
    scale_x = resized_hw[1] / original_hw[1]
    top, left = crop_top_left
    return torch.tensor(
        [
            [scale_x, 0, (scale_x - 1) * 0.5 - left],
            [0, scale_y, (scale_y - 1) * 0.5 - top],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def project_points_to_images(
    point_xyz: torch.Tensor,
    depth_m: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    pixel_transform: torch.Tensor | None = None,
    depth_atol: float = 0.01,
    depth_rtol: float = 0.01,
    return_intermediate: bool = False,
) -> dict[str, torch.Tensor]:
    """米制 XYZ 投影到整数像素中心；先检查原图深度，再做像素变换。"""
    if not depth_m.is_floating_point():
        raise TypeError("convert raw depth to floating-point meters before projection")
    height, width = depth_m.shape[-2:]

    with torch.autocast(device_type=point_xyz.device.type, enabled=False):
        extrinsic = world_to_camera.float()
        camera_xyz = point_xyz.float()[:, None] @ extrinsic[..., :3, :3].transpose(
            -1, -2
        ) + extrinsic[..., :3, 3].unsqueeze(-2)
        projected = camera_xyz @ intrinsics.float().transpose(-1, -2)
        denominator = projected[..., 2:3]
        finite = torch.isfinite(projected).all(-1) & torch.isfinite(camera_xyz).all(-1)
        valid = (
            finite & (camera_xyz[..., 2] > 1e-6) & (denominator[..., 0].abs() > 1e-6)
        )
        uv = projected[..., :2] / torch.where(denominator.abs() > 1e-6, denominator, 1)
        uv = torch.nan_to_num(uv)
        u, v = uv.unbind(-1)
        valid &= (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        in_image = valid
        col = (u.clamp(0, width - 1) + 0.5).floor().long()
        row = (v.clamp(0, height - 1) + 0.5).floor().long()
        observed_depth = depth_m.float().flatten(-2).gather(-1, row * width + col)
        depth_valid = in_image & torch.isfinite(observed_depth) & (observed_depth > 0)
        residual = (observed_depth - camera_xyz[..., 2]).abs()
        tolerance = depth_atol + depth_rtol * observed_depth
        valid = depth_valid & (residual <= tolerance)
        scaled_residual = residual / tolerance.clamp_min(1e-8)
        weight = torch.where(valid, torch.exp(-0.5 * scaled_residual.square()), 0)
        raw_uv = uv

        if pixel_transform is not None:
            homogeneous_uv = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
            transformed = homogeneous_uv @ pixel_transform.float().transpose(-1, -2)
            divisor = transformed[..., 2:3]
            transform_valid = torch.isfinite(transformed).all(-1) & (
                divisor[..., 0].abs() > 1e-6
            )
            uv = transformed[..., :2] / torch.where(divisor.abs() > 1e-6, divisor, 1)
            weight = torch.where(transform_valid, weight, 0)
        outputs = {"point_uv": torch.nan_to_num(uv), "view_weight": weight}
        if return_intermediate:
            outputs.update(
                raw_uv=raw_uv,
                camera_depth=camera_xyz[..., 2],
                sampled_depth=observed_depth,
                in_image=in_image,
                depth_valid=depth_valid,
                depth_residual=residual,
            )
        return outputs


def gather_members(value: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = value.shape[:2]
    flat = value.reshape(batch_size, num_points, -1)
    index = neighbor_idx.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, index).reshape(*neighbor_idx.shape, *value.shape[2:])


@torch.no_grad()
def build_patch_image_weights(
    neighbor_idx: torch.Tensor,
    point_uv: torch.Tensor,
    view_weight: torch.Tensor,
    image_hw: tuple[int, int],
    patch_grid_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    batch_size, num_views = point_uv.shape[:2]
    height, width = image_hw
    grid_h, grid_w = patch_grid_size
    num_patches, group_size = neighbor_idx.shape[1:]
    with torch.autocast(device_type=point_uv.device.type, enabled=False):
        finite = torch.isfinite(point_uv).all(-1) & torch.isfinite(view_weight)
        uv = torch.nan_to_num(point_uv.float())
        u, v = uv.unbind(-1)
        valid = (
            finite & (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        )
        confidence = torch.where(valid, view_weight.float().clamp(0, 1), 0)
        view_sum = confidence.sum(dim=1, keepdim=True)
        normalized = confidence / torch.where(view_sum > 0, view_sum, 1)
        point_valid = view_sum[:, 0] > 0

        x = ((u + 0.5) * (grid_w / width) - 0.5).clamp(0, grid_w - 1)
        y = ((v + 0.5) * (grid_h / height) - 0.5).clamp(0, grid_h - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp_max(grid_w - 1), (y0 + 1).clamp_max(grid_h - 1)
        dx, dy = x - x0, y - y0
        token_idx = torch.stack(
            (y0 * grid_w + x0, y0 * grid_w + x1, y1 * grid_w + x0, y1 * grid_w + x1),
            dim=-1,
        )
        token_idx += torch.arange(num_views, device=uv.device)[None, :, None, None] * (
            grid_h * grid_w
        )
        bilinear = torch.stack(
            ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), dim=-1
        )
        point_weight = normalized.unsqueeze(-1) * bilinear

        member_valid = gather_members(point_valid, neighbor_idx)
        count = member_valid.sum(-1)
        normalizer = count.clamp_min(1)
        member_idx = gather_members(token_idx.permute(0, 2, 1, 3), neighbor_idx)
        member_weight = gather_members(point_weight.permute(0, 2, 1, 3), neighbor_idx)
        pool_weight = uv.new_zeros(batch_size, num_patches, num_views * grid_h * grid_w)
        pool_weight.scatter_add_(
            2,
            member_idx.reshape(batch_size, num_patches, -1),
            member_weight.reshape(batch_size, num_patches, -1)
            / normalizer.unsqueeze(-1),
        )
        member_confidence = gather_members(confidence.amax(dim=1), neighbor_idx)
        return {
            "pool_weight": pool_weight,
            "semantic_valid_mask": count > 0,
            "semantic_coverage": count.float() / group_size,
            "semantic_confidence": member_confidence.sum(-1) / normalizer,
        }


class PointImageFusion(nn.Module):
    def __init__(
        self,
        image_channels: int,
        token_channels: int = 192,
        gate_init: float = 0.1,
    ):
        super().__init__()
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0,1)")
        self.image_channels = image_channels
        self.token_channels = token_channels

        self.image_norm = nn.LayerNorm(image_channels)
        self.proj = nn.Linear(image_channels, token_channels)
        self.geometry_norm = nn.LayerNorm(token_channels)
        self.semantic_norm = nn.LayerNorm(token_channels)
        self.gate = nn.Linear(token_channels * 2 + 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))

    def forward(
        self,
        local_token: torch.Tensor,
        neighbor_idx: torch.Tensor,
        image_tokens: torch.Tensor,
        point_uv: torch.Tensor,
        view_weight: torch.Tensor,
        image_hw: tuple[int, int],
        patch_grid_size: tuple[int, int],
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        if image_tokens.shape[2] != math.prod(patch_grid_size):
            raise ValueError(
                "image token count must equal Hf*Wf; remove prefix tokens first"
            )

        geometry = build_patch_image_weights(
            neighbor_idx, point_uv, view_weight, image_hw, patch_grid_size
        )
        # 池化至少以 FP32 累积，避免 autocast 降低精度。
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            pool_dtype = (
                torch.float64 if image_tokens.dtype == torch.float64 else torch.float32
            )
            pooled = torch.bmm(
                geometry["pool_weight"].to(pool_dtype),
                image_tokens.flatten(1, 2).to(pool_dtype),
            )
        semantic = self.proj(self.image_norm(pooled.to(self.proj.weight.dtype)))
        quality = torch.stack(
            (geometry["semantic_coverage"], geometry["semantic_confidence"]), dim=-1
        )
        gate_input = torch.cat(
            (
                self.geometry_norm(local_token),
                self.semantic_norm(semantic),
                quality.to(local_token.dtype),
            ),
            dim=-1,
        )
        gate = self.gate(gate_input).sigmoid()
        valid = geometry["semantic_valid_mask"].unsqueeze(-1)
        update = torch.where(valid, gate * semantic, 0).to(local_token.dtype)
        outputs = {"fused_token": local_token + update}
        if return_intermediate:
            outputs.update(geometry)
            outputs.update(
                semantic_token=pooled, semantic_gate=torch.where(valid, gate, 0)
            )
        return outputs
