import math

import torch
import torch.nn as nn


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
