"""Pool geometrically aligned RGB semantics into spatial tokens."""

import math

import torch
from torch import nn

from .projection import PointProjection
from .ops import index_points


@torch.no_grad()
def build_patch_correspondence(
    neighbor_idx: torch.Tensor,
    projection: PointProjection,
    patch_grid_size: tuple[int, int],
    *,
    member_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build FP32 image-grid pooling weights; coverage/confidence always use geometric weights."""
    grid_xy, confidence = projection
    batch = grid_xy.shape[0]
    grid_h, grid_w = patch_grid_size
    with torch.autocast(device_type=grid_xy.device.type, enabled=False):
        coordinates = grid_xy.float()
        valid = (
            torch.isfinite(coordinates).all(-1)
            & torch.isfinite(confidence)
            & (coordinates >= -1).all(-1)
            & (coordinates < 1).all(-1)
        )
        confidence = torch.where(valid, confidence.clamp(0, 1), 0.0)
        u, v = torch.nan_to_num(coordinates).unbind(-1)
        x = ((u + 1) * (grid_w / 2) - 0.5).clamp(0, grid_w - 1)
        y = ((v + 1) * (grid_h / 2) - 0.5).clamp(0, grid_h - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp_max(grid_w - 1), (y0 + 1).clamp_max(grid_h - 1)
        dx, dy = x - x0, y - y0
        token_idx = torch.stack(
            (y0 * grid_w + x0, y0 * grid_w + x1, y1 * grid_w + x0, y1 * grid_w + x1),
            dim=-1,
        )
        bilinear = torch.stack(
            ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy),
            dim=-1,
        )
        supported_confidence = confidence * bilinear.sum(-1)
        reliability = supported_confidence
        active = (supported_confidence > 0).to(bilinear.dtype)
        mass = active[..., None] * bilinear  # [BT,N,4]
        total_mass = mass.sum(dim=-1, keepdim=True)
        point_weight = mass / torch.where(total_mass > 0, total_mass, 1)
        point_valid = total_mass[..., 0] > 0
        member_valid = index_points(point_valid, neighbor_idx)
        if member_weight is None:
            supported = member_valid.float()
            count = supported.sum(-1)
            coverage = count / neighbor_idx.shape[-1]
        else:
            weights = member_weight.float()
            base_weight = torch.where(torch.isfinite(weights) & (weights > 0), weights, 0.0)
            maximum = base_weight.amax(-1, keepdim=True)
            base_weight = base_weight / torch.where(maximum > 0, maximum, 1)
            supported = base_weight * member_valid
            count = supported.sum(-1)
            total = base_weight.sum(-1)
            coverage = count / torch.where(total > 0, total, 1)

        member_prior = supported / torch.where(count > 0, count, 1)[..., None]
        member_confidence = index_points(reliability, neighbor_idx)
        quality_confidence = (member_confidence * member_prior).sum(-1)

        member_idx = index_points(token_idx, neighbor_idx)
        member_edges = index_points(point_weight, neighbor_idx)
        pool_weight = coordinates.new_zeros(batch, neighbor_idx.shape[1], grid_h * grid_w)
        # Scatter accumulates repeated corners/members without [BT,M,K,Di] features.
        pool_weight.scatter_add_(
            2,
            member_idx.flatten(2),
            (member_edges * member_prior[..., None]).flatten(2),
        )
        return {
            "pool_weight": pool_weight,
            "semantic_valid": count > 0,
            "coverage": coverage.clamp(0, 1),
            "confidence": quality_confidence.clamp(0, 1),
        }


class FeatureAlignment(nn.Module):
    """Geometric RGB pooling followed by a quality-gated spatial residual."""

    def __init__(
        self,
        image_channels: int,
        token_channels: int = 192,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if min(image_channels, token_channels) <= 0:
            raise ValueError("channel counts must be positive")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0,1)")

        self.image_norm = nn.LayerNorm(image_channels)
        self.proj = nn.Linear(image_channels, token_channels)
        self.geometry_norm = nn.LayerNorm(token_channels)
        self.semantic_norm = nn.LayerNorm(token_channels)
        self.gate = nn.Linear(token_channels * 2 + 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        image_grid: torch.Tensor,
        projection: PointProjection,
        neighbor_idx: torch.Tensor,
        *,
        member_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse a uniform image grid with tokens; projection/neighbors share Sparse3D point order."""
        geometry = build_patch_correspondence(
            neighbor_idx,
            projection,
            (image_grid.shape[1], image_grid.shape[2]),
            member_weight=member_weight,
        )
        weight = geometry["pool_weight"]
        image = image_grid.flatten(1, 2)
        # Unreferenced NaN/Inf must not contaminate bmm through zero weights.
        image = torch.where(weight.ne(0).any(dim=1)[..., None], image, 0.0)
        with torch.autocast(device_type=image.device.type, enabled=False):
            pooled = torch.bmm(weight, image.float())
        semantic = self.proj(self.image_norm(pooled.to(self.proj.weight.dtype)))
        quality = torch.stack((geometry["coverage"], geometry["confidence"]), dim=-1)
        gate_input = torch.cat(
            (
                self.geometry_norm(spatial_tokens),
                self.semantic_norm(semantic),
                quality.to(spatial_tokens.dtype),
            ),
            dim=-1,
        )
        gate = self.gate(gate_input).float().sigmoid()
        valid = geometry["semantic_valid"].unsqueeze(-1)
        update = (gate * semantic).to(spatial_tokens.dtype)
        return torch.where(valid, spatial_tokens + update, spatial_tokens)
