"""Align image semantics to spatial tokens using projected point-patch correspondence."""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


def gather_patch_members(value: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = value.shape[:2]
    flat = value.reshape(batch_size, num_points, -1)
    index = neighbor_idx.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, index).reshape(*neighbor_idx.shape, *value.shape[2:])


@torch.no_grad()
def build_patch_correspondence(
    neighbor_idx: torch.Tensor,
    point_uv: torch.Tensor,
    view_weight: torch.Tensor,
    image_hw: tuple[int, int],
    patch_grid_size: tuple[int, int],
    *,
    image_token_valid: torch.Tensor | None = None,
    member_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a geometric prior [B,M,V*Hf*Wf] and patch support statistics."""
    batch_size, num_views, _, _ = point_uv.shape
    index_batch, num_patches, num_members = neighbor_idx.shape
    height, width = image_hw
    grid_h, grid_w = patch_grid_size
    if min(height, width, grid_h, grid_w) <= 0:
        raise ValueError("image and patch grid dimensions must be positive")
    if view_weight.shape != point_uv.shape[:-1]:
        raise ValueError("view_weight must match point_uv [B,V,N]")
    if index_batch != batch_size or num_members == 0:
        raise ValueError("neighbor_idx must match the point batch and have at least one member")
    if image_token_valid is not None and (
        image_token_valid.dtype != torch.bool or image_token_valid.shape != (batch_size, num_views, grid_h * grid_w)
    ):
        raise ValueError("image_token_valid must be bool [B,V,Hf*Wf]")
    if member_weight is not None and member_weight.shape != neighbor_idx.shape:
        raise ValueError("member_weight must match neighbor_idx [B,M,K]")

    dtype = (
        torch.float64
        if any(t is not None and t.dtype == torch.float64 for t in (point_uv, view_weight, member_weight))
        else torch.float32
    )
    with torch.autocast(device_type=point_uv.device.type, enabled=False):
        finite = torch.isfinite(point_uv).all(-1) & torch.isfinite(view_weight)
        uv = torch.nan_to_num(point_uv.to(dtype))
        u, v = uv.unbind(-1)
        valid = finite & (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        confidence = torch.where(valid, view_weight.to(dtype).clamp(0, 1), 0.0)

        # Half-pixel convention: align_corners=False with border clamping.
        x = ((u + 0.5) * (grid_w / width) - 0.5).clamp(0, grid_w - 1)
        y = ((v + 0.5) * (grid_h / height) - 0.5).clamp(0, grid_h - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp_max(grid_w - 1), (y0 + 1).clamp_max(grid_h - 1)
        dx, dy = x - x0, y - y0
        token_idx = torch.stack((y0 * grid_w + x0, y0 * grid_w + x1, y1 * grid_w + x0, y1 * grid_w + x1), -1)
        bilinear = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), -1)
        if image_token_valid is not None:
            corner_valid = image_token_valid.gather(2, token_idx.flatten(2)).reshape_as(token_idx)
            bilinear = bilinear * corner_valid
            support = bilinear.sum(-1)
            confidence = confidence * support
            bilinear = bilinear / torch.where(support > 0, support, 1)[..., None]

        view_sum = confidence.sum(dim=1, keepdim=True)
        normalized = confidence / torch.where(view_sum > 0, view_sum, 1)
        point_weight = normalized.unsqueeze(-1) * bilinear
        point_valid = view_sum[:, 0] > 0
        view_offset = torch.arange(num_views, device=uv.device) * (grid_h * grid_w)
        token_idx = token_idx + view_offset[None, :, None, None]

        member_valid = gather_patch_members(point_valid, neighbor_idx)
        if member_weight is None:
            supported = member_valid.to(dtype)
            count = supported.sum(-1)
            coverage = count / num_members
        else:
            weights = member_weight.to(dtype)
            base_weight = torch.where(torch.isfinite(weights) & (weights > 0), weights, 0.0)
            # Rescaling preserves relative weights and avoids overflow in sums.
            maximum = base_weight.amax(-1, keepdim=True)
            base_weight = base_weight / torch.where(maximum > 0, maximum, 1)
            supported = base_weight * member_valid
            count = supported.sum(-1)
            total = base_weight.sum(-1)
            coverage = count / torch.where(total > 0, total, 1)

        normalizer = torch.where(count > 0, count, 1)
        member_prior = supported / normalizer[..., None]
        member_idx = gather_patch_members(token_idx.permute(0, 2, 1, 3), neighbor_idx)
        member_edges = gather_patch_members(point_weight.permute(0, 2, 1, 3), neighbor_idx)
        pool_weight = uv.new_zeros(batch_size, num_patches, num_views * grid_h * grid_w)
        # Accumulate repeated references without expanding [B,M,K,C_image] features.
        pool_weight.scatter_add_(
            2,
            member_idx.flatten(2),
            (member_edges * member_prior[..., None, None]).flatten(2),
        )
        member_confidence = gather_patch_members(confidence.amax(dim=1), neighbor_idx)
        return {
            "pool_weight": pool_weight,
            "semantic_valid_mask": count > 0,
            "semantic_coverage": coverage,
            "semantic_confidence": (member_confidence * member_prior).sum(-1),
        }


class FeatureAlignment(nn.Module):
    """Align image semantics to spatial tokens with geometric pooling or support-restricted attention."""

    def __init__(
        self,
        image_channels: int,
        token_channels: int = 192,
        gate_init: float = 0.1,
        *,
        alignment: str = "pool",
        attention_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(image_channels, token_channels) <= 0:
            raise ValueError("channel counts must be positive")
        if alignment == "content" and attention_dim <= 0:
            raise ValueError("attention_dim must be positive for content alignment")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0,1)")
        if alignment not in {"pool", "content"}:
            raise ValueError("alignment must be 'pool' or 'content'")

        self.image_channels = image_channels
        self.alignment = alignment
        self.attention_dim = attention_dim
        self.image_norm = nn.LayerNorm(image_channels)
        self.proj = nn.Linear(image_channels, token_channels)
        self.geometry_norm = nn.LayerNorm(token_channels)
        self.semantic_norm = nn.LayerNorm(token_channels)
        self.gate = nn.Linear(token_channels * 2 + 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))
        if alignment == "content":
            self.query = nn.Linear(token_channels, attention_dim, bias=False)
            self.key = nn.Linear(image_channels, attention_dim, bias=False)
            self.content_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        neighbor_idx: torch.Tensor,
        image_tokens: torch.Tensor,
        point_uv: torch.Tensor,
        view_weight: torch.Tensor,
        image_hw: tuple[int, int],
        patch_grid_size: tuple[int, int],
        return_intermediate: bool = False,
        *,
        image_token_valid: torch.Tensor | None = None,
        member_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if image_tokens.shape[:2] != point_uv.shape[:2]:
            raise ValueError("image_tokens must match point_uv batch and views")
        if image_tokens.shape[2] != math.prod(patch_grid_size):
            raise ValueError("image token count must equal Hf*Wf; remove prefix tokens first")
        geometry = build_patch_correspondence(
            neighbor_idx,
            point_uv,
            view_weight,
            image_hw,
            patch_grid_size,
            image_token_valid=image_token_valid,
            member_weight=member_weight,
        )
        return self.fuse_tokens(spatial_tokens, image_tokens, geometry, return_intermediate)

    def fuse_tokens(
        self,
        spatial_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        geometry: Mapping[str, torch.Tensor],
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Reuse correspondence while projections, grouping, grids and masks remain unchanged."""
        batch, patches, _ = spatial_tokens.shape
        image_batch, views, tokens, _ = image_tokens.shape
        prior = geometry["pool_weight"]
        if image_batch != batch or prior.shape != (batch, patches, views * tokens):
            raise ValueError("pool_weight does not match point/image token layout")
        if geometry["semantic_valid_mask"].shape != (batch, patches):
            raise ValueError("semantic_valid_mask must have shape [B,M]")

        # Clear unreferenced NaN/Inf before normalization and matrix products.
        used = prior.ne(0).any(dim=1)
        image = torch.where(used[..., None], image_tokens.flatten(1, 2), 0.0)
        pool_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        weight = prior.to(pool_dtype)
        geometry_features = None
        if self.alignment == "content":
            geometry_features = self.geometry_norm(spatial_tokens.to(self.query.weight.dtype))
            query = self.query(geometry_features)
            if geometry_features.dtype != spatial_tokens.dtype:
                geometry_features = None
            key = self.key(F.layer_norm(image.to(self.key.weight.dtype), (self.image_channels,)))
            with torch.autocast(device_type=image.device.type, enabled=False):
                scores = torch.bmm(query.to(pool_dtype), key.to(pool_dtype).transpose(1, 2))
                support = weight > 0
                # log(1) on forbidden entries avoids log(0) in backward.
                log_prior = torch.where(support, weight, 1).log()
                logits = log_prior + self.content_scale.to(pool_dtype) * scores / math.sqrt(self.attention_dim)
                logits = logits.masked_fill(~support, float("-inf"))
                # Empty rows must not execute softmax([-inf,...,-inf]).
                logits = torch.where(support.any(-1, keepdim=True), logits, 0.0)
                weight = logits.softmax(-1).masked_fill(~support, 0.0)

        with torch.autocast(device_type=image.device.type, enabled=False):
            pooled = torch.bmm(weight, image.to(pool_dtype))
        semantic = self.proj(self.image_norm(pooled.to(self.proj.weight.dtype)))
        quality = torch.stack((geometry["semantic_coverage"], geometry["semantic_confidence"]), -1)
        gate_input = torch.cat(
            (
                self.geometry_norm(spatial_tokens) if geometry_features is None else geometry_features,
                self.semantic_norm(semantic),
                quality.to(spatial_tokens.dtype),
            ),
            dim=-1,
        )
        gate_logits = self.gate(gate_input)
        # Keep small changes visible under BF16/FP16 autocast.
        if gate_logits.dtype in (torch.float16, torch.bfloat16):
            gate_logits = gate_logits.float()
        gate = gate_logits.sigmoid()
        valid = geometry["semantic_valid_mask"].unsqueeze(-1)
        update = torch.where(valid, gate * semantic, 0).to(spatial_tokens.dtype)
        outputs = {"tokens": spatial_tokens + update}
        if return_intermediate:
            outputs = {
                **outputs,
                **geometry,
                "semantic_tokens": pooled,
                "semantic_gate": torch.where(valid, gate, 0),
                "semantic_weight": weight,
            }
        return outputs
