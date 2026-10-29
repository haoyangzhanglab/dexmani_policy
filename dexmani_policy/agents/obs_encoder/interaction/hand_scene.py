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
    point_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    with torch.autocast(device_type=points.device.type, enabled=False):
        valid = torch.isfinite(points).all(-1)
        if point_mask is not None:
            valid = valid & point_mask
        points = torch.where(valid[..., None], points.float(), 0.0)
        anchors = anchors.float()

        # A masked sentinel lets empty clouds use the same KNN path.
        points = torch.cat((points, points.new_zeros(points.shape[0], 1, 3)), 1)
        valid = torch.cat((valid, valid.new_zeros(valid.shape[0], 1)), 1)

        distance2 = (points[:, None, :, 0] - anchors[:, :, None, 0]).square_()
        for axis in (1, 2):
            distance2.add_((points[:, None, :, axis] - anchors[:, :, None, axis]).square_())
        distance2.masked_fill_(~valid[:, None], torch.inf)
        distance2, indices = distance2.topk(min(k, points.shape[1]), largest=False)
        batch_indices = torch.arange(points.shape[0], device=points.device)[:, None, None]
        relative = points[batch_indices, indices] - anchors[:, :, None]

        nearest_distance = distance2[..., 0].sqrt()
        has_reference = torch.isfinite(nearest_distance)
        local_valid = distance2 <= radius**2
        near_supported = local_valid[..., 0]
        nearest_relative = torch.where(has_reference[..., None], relative[..., 0, :], 0.0)
        safe_distance = torch.where(has_reference, nearest_distance, 0.0)
        denominator = metric_scale + safe_distance[..., None]
        proximity = torch.exp(-0.5 * (safe_distance / radius).square()) * has_reference
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
            "nearest_distance": nearest_distance,
            "has_reference": has_reference,
            "near_supported": near_supported,
        }


class SpatialCrossAttention(nn.Module):
    def __init__(
        self, channels: int, scene_channels: int, num_heads: int, metric_scale: float
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.head_dim = channels // num_heads
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
        valid: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch, queries, channels = query.shape
        patches = memory.shape[1]
        memory = torch.where(valid[..., None], memory, 0.0).to(self.kv.weight.dtype)
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
        k, v = F.pad(k, (0, 0, 0, 1)), F.pad(v, (0, 0, 0, 1))
        available = valid.any(-1, keepdim=True)
        # Only empty scenes can read the zero K/V slot.
        allowed = torch.cat((valid, ~available), -1)
        message = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None, None])
        message = message.transpose(1, 2).reshape(batch, queries, channels)
        output = self.out(message.to(self.out.weight.dtype))

        attention = None
        if return_attention:
            with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
                attention = logits.masked_fill(~allowed[:, None, None], -torch.inf).softmax(-1)
                attention = attention.transpose(1, 2)
        return output, attention


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
        point_mask: Tensor | None = None,
        query_context: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """XYZ use world meters; rotation maps EEF axes to world; return scene increments."""
        memory, centers, token_valid = patches["tokens"], patches["xyz"], patches["mask"]
        batch, channels = memory.shape[0], self.token_channels
        if anchor_xyz.shape != (batch, 6, 3) or hand_tokens.shape != (batch, 6, channels):
            raise ValueError("anchor_xyz/hand_tokens must be [N,6,3] / [N,6,D]")
        if rotation_world_from_eef.shape != (batch, 3, 3):
            raise ValueError("rotation_world_from_eef must be [N,3,3]")

        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            anchors = anchor_xyz.float()
            rotation = rotation_world_from_eef.float()
            valid = token_valid & torch.isfinite(centers).all(-1) & torch.isfinite(memory).all(-1)
            query_xyz = (anchors - anchors[:, :1]) @ rotation
            memory_xyz = torch.where(valid[..., None], centers.float() - anchors[:, :1], 0.0) @ rotation
            support = query_local_surface(
                pointcloud[..., :3], anchors, self.surface_k,
                self.surface_radius, self.metric_scale, point_mask,
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

        context, attention = self.context(
            query, memory, query_xyz, memory_xyz, valid, return_attention=return_intermediate,
        )
        update = geometry_update + context.float()
        output = {
            "hand_query": hand_tokens,
            "relation_update": update,
            "wrist_relation": update[:, :1],
            "finger_relation": update[:, 1:],
            "interaction_center": anchors,
            "geometry_descriptor": geometry,
            "min_observed_distance": support["nearest_distance"],
            "has_reference": support["has_reference"],
            "near_has_support": support["near_supported"],
        }
        if return_intermediate:
            output.update(
                geometry_update=geometry_update,
                context_update=context,
                context_attention=attention,
                context_null_mass=attention[..., -1],
                surface_features=local,
                finger_local=local[:, 1:],
                surface_relative=surface_relative,
                surface_valid=support["valid"],
            )
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
