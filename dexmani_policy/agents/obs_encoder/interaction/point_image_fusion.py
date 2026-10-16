"""Geometry-supported 2D/3D patch fusion; no point-wise semantic expansion.

The caller supplies projections for the SAME points indexed by neighbor_idx.
RGB transforms, visibility/depth checks and prefix-token removal stay upstream.
See docs/point_image_fusion.md for layout, masking and precision contracts.
"""

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_members(value: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = value.shape[:2]
    flat = value.reshape(batch_size, num_points, -1)
    index = neighbor_idx.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, index).reshape(*neighbor_idx.shape, *value.shape[2:])


def _check_hw(value: tuple[int, int], name: str) -> None:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(x) is not int or x <= 0 for x in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")


@torch.no_grad()
def build_patch_image_weights(
    neighbor_idx: torch.Tensor,
    point_uv: torch.Tensor,
    view_weight: torch.Tensor,
    image_hw: tuple[int, int],
    patch_grid_size: tuple[int, int],
    *,
    image_token_valid: torch.Tensor | None = None,
    member_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a reusable geometry prior [B,M,V*Hf*Wf] from scalar edges.

    Each view is a row-major regular grid with the SAME processed image/grid
    size. image_token_valid is bool [B,V,Hf*Wf]; False removes that token before
    normalization. member_weight is optional nonnegative [B,M,K] support;
    zero/nonfinite/negative entries are excluded. All indices must still be
    in [0,N), including zero-weight members (no -1 padding sentinel).

    Finite positive view weights are clipped to [0,1]. Masked interpolation
    mass reduces view confidence; surviving contributions normalize per point,
    then per 3D patch. Default member weights reproduce uniform valid pooling.
    Quality scalars describe observation support, NOT calibrated uncertainty.
    """
    _check_hw(image_hw, "image_hw")
    _check_hw(patch_grid_size, "patch_grid_size")
    if point_uv.ndim != 4 or point_uv.shape[-1] != 2 or not point_uv.is_floating_point():
        raise ValueError("point_uv must be floating [B,V,N,2]")
    batch_size, num_views, num_points = point_uv.shape[:3]
    if min(batch_size, num_views, num_points) < 1:
        raise ValueError("point_uv batch, views and points must be nonempty")
    if view_weight.shape != point_uv.shape[:-1] or not view_weight.is_floating_point():
        raise ValueError("view_weight must be floating [B,V,N]")
    if (
        neighbor_idx.ndim != 3
        or neighbor_idx.dtype != torch.long
        or neighbor_idx.shape[0] != batch_size
        or min(neighbor_idx.shape[1:]) < 1
    ):
        raise ValueError("neighbor_idx must be nonempty int64 [B,M,K]")
    height, width = image_hw
    grid_h, grid_w = patch_grid_size
    num_patches = neighbor_idx.shape[1]
    if image_token_valid is not None and (
        image_token_valid.dtype != torch.bool
        or image_token_valid.shape != (batch_size, num_views, grid_h * grid_w)
    ):
        raise ValueError("image_token_valid must be bool [B,V,Hf*Wf]")
    if member_weight is not None and (
        member_weight.shape != neighbor_idx.shape or not member_weight.is_floating_point()
    ):
        raise ValueError("member_weight must be floating [B,M,K]")
    for tensor in (neighbor_idx, view_weight, image_token_valid, member_weight):
        if tensor is not None and tensor.device != point_uv.device:
            raise ValueError("all correspondence inputs must be on the same device")

    # Invalid indices are rejected by gather without a GPU-to-CPU scalar check.
    dtype = (
        torch.float64
        if any(
            t is not None and t.dtype == torch.float64
            for t in (point_uv, view_weight, member_weight)
        )
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
        token_idx = torch.stack(
            (y0 * grid_w + x0, y0 * grid_w + x1, y1 * grid_w + x0, y1 * grid_w + x1), -1
        )
        bilinear = torch.stack(
            ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), -1
        )
        if image_token_valid is not None:
            corner_valid = image_token_valid.gather(
                2, token_idx.flatten(2)
            ).reshape_as(token_idx)
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

        member_valid = gather_members(point_valid, neighbor_idx)
        if member_weight is None:
            base_weight = torch.ones_like(member_valid, dtype=dtype)
        else:
            weights = member_weight.to(dtype)
            base_weight = torch.where(torch.isfinite(weights) & (weights > 0), weights, 0.0)
            # Relative member weights: rescale before summing to avoid overflow.
            maximum = base_weight.amax(-1, keepdim=True)
            base_weight = base_weight / torch.where(maximum > 0, maximum, 1)
        supported = base_weight * member_valid
        count = supported.sum(-1)
        normalizer = torch.where(count > 0, count, 1)
        member_prior = supported / normalizer[..., None]
        member_idx = gather_members(token_idx.permute(0, 2, 1, 3), neighbor_idx)
        member_edges = gather_members(point_weight.permute(0, 2, 1, 3), neighbor_idx)
        pool_weight = uv.new_zeros(batch_size, num_patches, num_views * grid_h * grid_w)
        # Repeated references coalesce exactly; never gather [B,M,K,C_image].
        pool_weight.scatter_add_(
            2, member_idx.flatten(2),
            (member_edges * member_prior[..., None, None]).flatten(2),
        )
        member_confidence = gather_members(confidence.amax(dim=1), neighbor_idx)
        total = base_weight.sum(-1)
        return {
            "pool_weight": pool_weight,
            "semantic_valid_mask": count > 0,
            "semantic_coverage": count / torch.where(total > 0, total, 1),
            "semantic_confidence": (member_confidence * member_prior).sum(-1),
        }


class PointImageFusion(nn.Module):
    """Geometry pooling (default) or low-rank content refinement on its support.

    alignment='pool' preserves the original parameter names and ordering.
    alignment='content' adds Q/K projections and a zero-initialized scalar:
        alpha[m,j] = softmax_j(log A[m,j] + scale * q[m] @ k[j] / sqrt(d)).
    A==0 remains forbidden, including empty rows. Values are original image
    tokens; LayerNorm/projection remain AFTER pooling. Dense BMM is deliberate
    for small ViT grids; this is support-masked, NOT a sparse-compute kernel.
    """

    def __init__(
        self,
        image_channels: int,
        token_channels: int = 192,
        gate_init: float = 0.1,
        *,
        alignment: str = "pool",
        attention_dim: int = 32,
    ):
        super().__init__()
        if any(
            type(x) is not int or x < 1
            for x in (image_channels, token_channels, attention_dim)
        ):
            raise ValueError("channel counts and attention_dim must be positive integers")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0,1)")
        if alignment not in {"pool", "content"}:
            raise ValueError("alignment must be 'pool' or 'content'")
        self.image_channels = image_channels
        self.token_channels = token_channels
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
        local_token: torch.Tensor,
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
        if image_tokens.ndim != 4 or image_tokens.shape[:2] != point_uv.shape[:2]:
            raise ValueError("image_tokens must be [B,V,Hf*Wf,C] matching point_uv")
        if image_tokens.shape[2] != math.prod(patch_grid_size):
            raise ValueError("image token count must equal Hf*Wf; remove prefix tokens first")
        geometry = build_patch_image_weights(
            neighbor_idx, point_uv, view_weight, image_hw, patch_grid_size,
            image_token_valid=image_token_valid, member_weight=member_weight,
        )
        return self.fuse_tokens(local_token, image_tokens, geometry, return_intermediate)

    def fuse_tokens(
        self,
        local_token: torch.Tensor,
        image_tokens: torch.Tensor,
        geometry: Mapping[str, torch.Tensor],
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Reuse a builder result for the SAME point/image layout and masks.

        No implicit persistent cache: changed projection, grouping, grid, mask
        or point/view order requires rebuilding. Referenced image tokens and
        geometry tokens must be finite. Unreferenced/padded image values may
        be NaN/Inf; they are cleared BEFORE normalization and matrix products.
        """
        if (
            local_token.ndim != 3
            or local_token.shape[-1] != self.token_channels
            or not local_token.is_floating_point()
        ):
            raise ValueError("local_token must be floating [B,M,token_channels]")
        if (
            image_tokens.ndim != 4
            or image_tokens.shape[-1] != self.image_channels
            or not image_tokens.is_floating_point()
        ):
            raise ValueError("image_tokens must be floating [B,V,Hf*Wf,image_channels]")
        batch, patches = local_token.shape[:2]
        if min(batch, patches, *image_tokens.shape[1:3]) < 1 or image_tokens.shape[0] != batch:
            raise ValueError("token batches must match and dimensions must be nonempty")
        prior = geometry["pool_weight"]
        if prior.shape != (batch, patches, image_tokens.shape[1] * image_tokens.shape[2]):
            raise ValueError("pool_weight does not match point/image token layout")
        for name in ("semantic_valid_mask", "semantic_coverage", "semantic_confidence"):
            if geometry[name].shape != (batch, patches):
                raise ValueError(f"{name} must have shape [B,M]")
        if geometry["semantic_valid_mask"].dtype != torch.bool:
            raise ValueError("semantic_valid_mask must be bool")
        for tensor in (image_tokens, *geometry.values()):
            if tensor.device != local_token.device:
                raise ValueError("features and correspondence must be on the same device")
        if not prior.is_floating_point():
            raise ValueError("pool_weight must be floating")

        used = prior.ne(0).any(dim=1)
        image = torch.where(used[..., None], image_tokens.flatten(1, 2), 0.0)
        pool_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        weight = prior.to(pool_dtype)
        if self.alignment == "content":
            query = self.query(self.geometry_norm(local_token.to(self.query.weight.dtype)))
            key = self.key(
                F.layer_norm(image.to(self.key.weight.dtype), (self.image_channels,))
            )
            with torch.autocast(device_type=image.device.type, enabled=False):
                scores = torch.bmm(query.to(pool_dtype), key.to(pool_dtype).transpose(1, 2))
                support = weight > 0
                # log(1) on forbidden entries avoids log(0) in backward.
                log_prior = torch.where(support, weight, 1).log()
                logits = (
                    log_prior
                    + self.content_scale.to(pool_dtype) * scores / math.sqrt(self.attention_dim)
                )
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
                self.geometry_norm(local_token),
                self.semantic_norm(semantic),
                quality.to(local_token.dtype),
            ),
            dim=-1,
        )
        gate_logits = self.gate(gate_input)
        # Keep small changes visible under BF16/FP16 autocast.
        if gate_logits.dtype in (torch.float16, torch.bfloat16):
            gate_logits = gate_logits.float()
        gate = gate_logits.sigmoid()
        valid = geometry["semantic_valid_mask"].unsqueeze(-1)
        update = torch.where(valid, gate * semantic, 0).to(local_token.dtype)
        outputs = {"fused_token": local_token + update}
        if return_intermediate:
            outputs.update(geometry)
            outputs.update(
                semantic_token=pooled, semantic_gate=torch.where(valid, gate, 0),
                semantic_weight=weight,
            )
        return outputs
