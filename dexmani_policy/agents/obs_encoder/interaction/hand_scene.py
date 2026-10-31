"""Hand-scene interaction through local geometry and spatial cross-attention."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..pointcloud.rope import RotaryPositionEmbedding3D


@torch.no_grad()
def query_local_surface(
    points: Tensor,
    anchors: Tensor,
    k: int,
    radius: float,
    metric_scale: float,
) -> dict[str, Tensor]:
    with torch.autocast(device_type=points.device.type, enabled=False):
        points, anchors = points.float(), anchors.float()

        distance2 = (points[:, None, :, 0] - anchors[:, :, None, 0]).square_()
        for axis in (1, 2):
            distance2.add_((points[:, None, :, axis] - anchors[:, :, None, axis]).square_())
        distance2, indices = distance2.topk(min(k, points.shape[1]), largest=False)
        batch_indices = torch.arange(points.shape[0], device=points.device)[:, None, None]
        relative = points[batch_indices, indices] - anchors[:, :, None]

        nearest_distance = distance2[..., 0].sqrt()
        has_reference = torch.ones_like(nearest_distance, dtype=torch.bool)
        local_valid = distance2 <= radius**2
        near_supported = local_valid[..., 0]
        nearest_relative = relative[..., 0, :]
        safe_distance = nearest_distance
        denominator = metric_scale + safe_distance[..., None]
        proximity = torch.exp(-0.5 * (safe_distance / radius).square())
        descriptor = torch.cat(
            (
                nearest_relative / denominator,
                safe_distance[..., None] / denominator,
                proximity[..., None],
                has_reference[..., None].float(),
                near_supported[..., None].float(),
            ),
            -1,
        )

        return {
            "relative": torch.where(local_valid[..., None], relative, 0.0),
            "valid": local_valid,
            "descriptor": descriptor,
        }


class SpatialCrossAttention(nn.Module):
    def __init__(
        self, channels: int, scene_channels: int, num_heads: int, metric_scale: float
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.head_dim = channels // num_heads
        self.metric_scale = metric_scale
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(scene_channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(scene_channels, 2 * channels)
        self.rope = RotaryPositionEmbedding3D(self.head_dim, scale=metric_scale)
        self.out = nn.Linear(channels, channels, bias=False)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        query_xyz: Tensor,
        memory_xyz: Tensor,
    ) -> Tensor:
        batch, queries, channels = query.shape
        patches = memory.shape[1]
        memory = memory.to(self.kv.weight.dtype)
        q = self.q(self.query_norm(query.to(self.q.weight.dtype)))
        q = q.reshape(batch, queries, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.memory_norm(memory))
        k, v = (
            kv.reshape(batch, patches, 2, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        # RoPE recenters positions, so queries and keys must share one origin.
        cos, sin = self.rope(torch.cat((query_xyz, memory_xyz), 1))
        q = self.rope.apply_rotary(q, cos[:, :, :queries], sin[:, :, :queries])
        k = self.rope.apply_rotary(k, cos[:, :, queries:], sin[:, :, queries:])
        with torch.autocast(device_type=q.device.type, enabled=False):
            distance2 = (memory_xyz[:, None].float() - query_xyz[:, :, None].float()).square().sum(-1)
            distance_bias = -torch.log1p(distance2 / self.metric_scale**2)
            # Wrist and the final head read globally.
            attention_bias = distance_bias.new_zeros(batch, self.heads, queries, patches)
            attention_bias[:, :-1, 1:, :patches] = distance_bias[:, None, 1:]
        attention_bias = attention_bias.to(q)
        message = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        message = message.transpose(1, 2).reshape(batch, queries, channels)
        output = self.out(message.to(self.out.weight.dtype))

        return output


class HandSceneEncoder(nn.Module):
    def __init__(
        self,
        token_channels: int = 192,
        metric_scale: float = 0.10,
        num_heads: int = 4,
        surface_k: int = 32,
        surface_radius: float = 0.04,
        *,
        scene_channels: int | None = None,
    ) -> None:
        super().__init__()
        scene_channels = token_channels if scene_channels is None else scene_channels
        if token_channels % num_heads:
            raise ValueError("token_channels must be divisible by num_heads")
        if surface_k < 1:
            raise ValueError("surface_k must be positive")
        if not math.isfinite(surface_radius) or surface_radius <= 0:
            raise ValueError("surface_radius must be finite and positive")

        self.token_channels = token_channels
        self.metric_scale = metric_scale
        self.surface_k = surface_k
        self.surface_radius = surface_radius
        self.surface_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
        self.geometry_projection = nn.Linear(32 + 7, token_channels, bias=False)
        self.context = SpatialCrossAttention(token_channels, scene_channels, num_heads, metric_scale)

    def forward(
        self,
        pointcloud: Tensor,
        patches: Mapping[str, Tensor],
        anchor_xyz: Tensor,
        hand_tokens: Tensor,
        *,
        rotation_world_from_eef: Tensor,
        query_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """XYZ use world meters; rotation maps EEF axes to world; return scene increments."""
        memory, centers = patches["tokens"], patches["xyz"]
        batch, channels = memory.shape[0], self.token_channels
        if anchor_xyz.shape != (batch, 6, 3) or hand_tokens.shape != (batch, 6, channels):
            raise ValueError("anchor_xyz/hand_tokens must be [N,6,3] / [N,6,D]")
        if rotation_world_from_eef.shape != (batch, 3, 3):
            raise ValueError("rotation_world_from_eef must be [N,3,3]")

        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            anchors = anchor_xyz.float()
            rotation = rotation_world_from_eef.float()
            query_xyz = (anchors - anchors[:, :1]) @ rotation
            memory_xyz = (centers.float() - anchors[:, :1]) @ rotation
            support = query_local_surface(
                pointcloud[..., :3], anchors, self.surface_k,
                self.surface_radius, self.metric_scale,
            )
            surface_relative = support["relative"] @ rotation[:, None]
            geometry = torch.cat(
                (support["descriptor"][..., :3] @ rotation, support["descriptor"][..., 3:]),
                -1,
            )

        local = self.surface_mlp(
            (surface_relative / self.surface_radius).to(self.surface_mlp[0].weight.dtype)
        )
        with torch.autocast(device_type=local.device.type, enabled=False):
            local = local.float().masked_fill(~support["valid"][..., None], 0.0)
            local = local.sum(-2) / support["valid"].sum(-1, keepdim=True).clamp_min(1)

        geometry_update = self.geometry_projection(
            torch.cat((local, geometry), -1).to(self.geometry_projection.weight.dtype)
        ).float()
        query = hand_tokens.float() + geometry_update
        if query_context is not None:
            if query_context.ndim == 2:
                query_context = query_context[:, None]
            query = query + query_context.float()

        context = self.context(query, memory, query_xyz, memory_xyz)
        update = geometry_update + context.float()
        return {
            "hand_query": hand_tokens,
            "relation_update": update,
            "geometry_descriptor": geometry,
        }

    @property
    def out_dim(self) -> int:
        return self.token_channels
