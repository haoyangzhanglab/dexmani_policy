"""Hand–scene cross-attention with metric context and near-field geometry."""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points


def nearest_patch_geometry(
    anchors: Tensor,
    members: Tensor,
    member_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return nearest visible member offsets, metric distances and validity; missing distances are inf."""
    valid = torch.isfinite(members).all(-1)
    if member_valid is not None:
        valid = valid & member_valid
    members = torch.where(valid[..., None], members.float(), 0.0)
    offsets = members[:, None] - anchors.float()[:, :, None, None]
    squared_distance = offsets.square().sum(-1).masked_fill(~valid[:, None], torch.inf)
    min_squared_distance, indices = squared_distance.min(dim=-1)
    observed = torch.isfinite(min_squared_distance)
    relative = offsets.gather(3, indices[..., None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    relative = torch.where(observed[..., None], relative, 0.0)
    return relative, min_squared_distance.sqrt(), observed


class SpatialCrossAttention(nn.Module):
    """Cross-attention with relative geometry, per-head distance priors and a learned null token."""

    def __init__(
        self,
        channels: int,
        sigmas: Sequence[float],
        metric_scale: float,
        proximity_scales: tuple[float, float],
        radius_multiple: float | None,
    ) -> None:
        super().__init__()
        self.heads = len(sigmas)
        self.head_dim = channels // self.heads
        self.metric_scale = metric_scale
        self.radius_multiple = radius_multiple
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.register_buffer("proximity_scales", torch.tensor(proximity_scales, dtype=torch.float32))
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.edge = nn.Sequential(nn.Linear(6, channels), nn.GELU())
        self.edge_bias = nn.Linear(channels, self.heads)
        self.edge_value = nn.Linear(channels, channels)
        self.null_key = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.out = nn.Linear(channels, channels)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        relative: Tensor,
        distance: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, queries, channels = query.shape
        patches = memory.shape[1]
        query = self.q(self.query_norm(query)).reshape(batch, queries, self.heads, self.head_dim)
        memory = self.memory_norm(memory)
        key = self.k(memory).reshape(batch, patches, self.heads, self.head_dim)
        value = self.v(memory).reshape(batch, patches, self.heads, self.head_dim)

        distance = torch.where(valid, distance.float(), 0.0)
        relative = torch.where(valid[..., None], relative.float(), 0.0)
        proximity = torch.exp(-0.5 * (distance[..., None] / self.proximity_scales).square())
        geometry = torch.cat((relative / self.metric_scale, distance[..., None] / self.metric_scale, proximity), -1)
        edge = self.edge(geometry.to(memory.dtype))
        edge_value = self.edge_value(edge).reshape(batch, queries, patches, self.heads, self.head_dim)
        edge_bias = self.edge_bias(edge).permute(0, 1, 3, 2)

        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bqhd,bmhd->bqhm", query.float(), key.float())
            logits = logits / math.sqrt(self.head_dim) + edge_bias.float()
            ratios = distance[:, :, None] / self.sigmas.float()[None, None, :, None]
            logits = logits - 0.5 * ratios.square()
            allowed = valid[:, :, None]
            if self.radius_multiple is not None:
                allowed = allowed & (ratios <= self.radius_multiple)
            logits = logits.masked_fill(~allowed, -torch.inf)
            null_logit = torch.einsum("bqhd,hd->bqh", query.float(), self.null_key.float()) / math.sqrt(self.head_dim)
            attention = torch.cat((logits, null_logit[..., None]), -1).softmax(-1)

        weights = attention[..., :-1].to(value.dtype)
        semantic = torch.einsum("bqhm,bmhd->bqhd", weights, value)
        relational = torch.einsum("bqhm,bqmhd->bqhd", weights, edge_value)
        null = attention[..., -1:].to(value.dtype) * self.null_value[None, None].to(value.dtype)
        output = self.out((semantic + relational + null).reshape(batch, queries, channels))
        support = allowed.any(-1).expand(-1, -1, self.heads)
        return output, attention, support


class HandSceneAttention(nn.Module):
    """Read scene context and near-field geometry at the wrist and five fingertips."""

    def __init__(
        self,
        token_channels: int = 192,
        context_sigmas: Sequence[float] = (0.08, 0.08, 0.20, float("inf")),
        near_sigmas: Sequence[float] = (0.015, 0.015, 0.04, 0.04),
        metric_scale: float = 0.10,
        radius_multiple: float = 3.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if token_channels <= 0 or num_heads <= 0 or token_channels % num_heads:
            raise ValueError("token_channels must be positive and divisible by num_heads")
        if len(context_sigmas) != num_heads or len(near_sigmas) != num_heads:
            raise ValueError("context_sigmas and near_sigmas must have length num_heads")
        if any(not math.isfinite(scale) or scale <= 0 for scale in (metric_scale, radius_multiple, *near_sigmas)):
            raise ValueError("metric_scale, radius_multiple and near_sigmas must be finite and positive")
        if any(math.isnan(scale) or scale <= 0 for scale in context_sigmas):
            raise ValueError("context_sigmas must be positive; infinity is allowed")

        self.token_channels = token_channels
        proximity_scales = min(near_sigmas), max(near_sigmas)
        self.context = SpatialCrossAttention(token_channels, context_sigmas, metric_scale, proximity_scales, None)
        self.near = SpatialCrossAttention(token_channels, near_sigmas, metric_scale, proximity_scales, radius_multiple)

    def forward(
        self,
        pointcloud: Tensor,
        patches: Mapping[str, Tensor],
        anchor_xyz: Tensor,
        hand_tokens: Tensor,
        *,
        member_valid: Tensor | None = None,
        query_context: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """Consume projected Sparse3D features and finite hand tokens in wrist/thumb/index/middle/ring/pinky order."""
        memory = patches["tokens"]
        centers, indices = patches["xyz"], patches["neighbor_indices"]
        token_valid = patches["mask"]
        batch, patch_count, _ = memory.shape
        if centers.shape != (batch, patch_count, 3):
            raise ValueError("patch centers must have shape [BT,M,3] matching patch tokens")
        if token_valid.dtype != torch.bool or token_valid.shape != (batch, patch_count):
            raise ValueError("patch mask must be boolean [BT,M]")
        if member_valid is not None and (member_valid.shape != indices.shape or member_valid.dtype != torch.bool):
            raise ValueError("member_valid must be boolean with shape [BT,M,K]")
        if anchor_xyz.shape != (batch, 6, 3) or not anchor_xyz.is_floating_point():
            raise ValueError("anchor_xyz must be floating point [BT,6,3]")
        if hand_tokens.shape != (batch, 6, self.token_channels):
            raise ValueError("hand_tokens must have shape [BT,6,D]")

        anchors = anchor_xyz.float()
        query = hand_tokens
        if query_context is not None:
            if query_context.shape == (batch, self.token_channels):
                query_context = query_context[:, None]
            if query_context.shape not in ((batch, 1, self.token_channels), (batch, 6, self.token_channels)):
                raise ValueError("query_context must be [BT,D], [BT,1,D] or [BT,6,D]")
            query = query + query_context.to(query.dtype)

        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            members = index_points(pointcloud[..., :3], indices)
            valid_members = token_valid[..., None]
            if member_valid is not None:
                valid_members = valid_members & member_valid
            relative, distance, observed = nearest_patch_geometry(anchors, members, valid_members)
            center_valid = torch.isfinite(centers).all(-1) & token_valid
            centers = torch.where(center_valid[..., None], centers.float(), 0.0)
            center_relative = centers[:, None] - anchors[:, :, None]
            center_distance = center_relative.norm(dim=-1)
            context_valid = center_valid[:, None] & observed

        context, context_attention, _ = self.context(query, memory, center_relative, center_distance, context_valid)
        near, near_attention, support = self.near(query, memory, relative, distance, observed)
        # Suppress learned null values and output bias when no head has near-field support.
        near_update = near * support.any(-1, keepdim=True).to(near.dtype)
        output = {
            "hand_query": hand_tokens,
            "relation_update": context + near_update,
            "interaction_center": anchors,
        }
        if return_intermediate:
            output = {
                **output,
                "context_update": context,
                "near_update": near_update,
                "min_observed_distance": distance.min(-1).values,
                "near_has_support": support,
                "near_null_mass": near_attention[..., -1],
                "context_null_mass": context_attention[..., -1],
                "near_attention": near_attention,
                "context_attention": context_attention,
                "context_relative": center_relative,
                "surface_relative": relative,
                "surface_distance": distance,
                "surface_valid": observed,
            }
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
