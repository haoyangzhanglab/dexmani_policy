from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..pointcloud.rope import RotaryPositionEmbedding3D


def _local_surface(points: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    distance2 = (points[:, None, :, 0] - anchors[:, :, None, 0]).square_()
    for axis in (1, 2):
        distance2.add_((points[:, None, :, axis] - anchors[:, :, None, axis]).square_())
    distance2, indices = distance2.topk(min(32, points.shape[1]), largest=False)
    batch_indices = torch.arange(points.shape[0], device=points.device)[:, None, None]
    relative = points[batch_indices, indices] - anchors[:, :, None]

    nearest_distance = distance2[..., 0].sqrt()
    local_valid = distance2 <= 0.04**2
    near_supported = local_valid[..., 0]
    nearest_relative = relative[..., 0, :]
    denominator = 0.10 + nearest_distance[..., None]
    proximity = torch.exp(-0.5 * (nearest_distance / 0.04).square())
    descriptor = torch.cat(
        (
            nearest_relative / denominator,
            nearest_distance[..., None] / denominator,
            proximity[..., None],
            near_supported[..., None].float(),
        ),
        -1,
    )

    return relative.masked_fill(~local_valid[..., None], 0.0), local_valid, descriptor


class SpatialCrossAttention(nn.Module):
    def __init__(self, scene_channels: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(192)
        self.memory_norm = nn.LayerNorm(scene_channels)
        self.q = nn.Linear(192, 192)
        self.kv = nn.Linear(scene_channels, 384)
        self.rope = RotaryPositionEmbedding3D(48, scale=0.10)
        self.out = nn.Linear(192, 192, bias=False)
        self.null_logits = nn.Parameter(torch.zeros(3))

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
        q = q.reshape(batch, queries, 4, 48).transpose(1, 2)
        kv = self.kv(self.memory_norm(memory))
        k, v = (
            kv.reshape(batch, patches, 2, 4, 48)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        # RoPE recenters positions, so queries and keys must share one origin.
        cos, sin = self.rope(torch.cat((query_xyz, memory_xyz), 1))
        q = self.rope.apply_rotary(q, cos[:, :, :queries], sin[:, :, :queries])
        k = self.rope.apply_rotary(k, cos[:, :, queries:], sin[:, :, queries:])
        with torch.autocast(device_type=q.device.type, enabled=False):
            distance2 = (memory_xyz[:, None].float() - query_xyz[:, :, None].float()).square().sum(-1)
            distance_bias = -torch.log1p(distance2 / 0.10**2)
            # Wrist and the final head read globally.
            attention_bias = distance_bias.new_zeros(batch, 4, queries, patches)
            attention_bias[:, :-1, 1:] = distance_bias[:, None, 1:]
        k = torch.cat((k, torch.zeros_like(k[:, :, :1])), dim=2)
        v = torch.cat((v, torch.zeros_like(v[:, :, :1])), dim=2)
        null_bias = self.null_logits.new_full((4, queries), -torch.inf)
        null_bias[:3, 1:] = self.null_logits[:, None]
        attention_bias = torch.cat(
            (attention_bias, null_bias[None, ..., None].expand(batch, -1, -1, -1)), -1,
        ).to(q)
        message = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        message = message.transpose(1, 2).reshape(batch, queries, channels)
        return self.out(message.to(self.out.weight.dtype))


class HandSceneFusion(nn.Module):
    def __init__(self, scene_channels: int, tactile_encoder: nn.Module) -> None:
        super().__init__()
        self.surface_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
        self.geometry_projection = nn.Linear(38, 192)
        self.context = SpatialCrossAttention(scene_channels)
        self.tactile_encoder = tactile_encoder
        tactile_dim = self.tactile_encoder.out_dim
        self.query_norm = nn.LayerNorm(192)
        self.force_affine = nn.Linear(tactile_dim + 3, 384)
        nn.init.zeros_(self.force_affine.weight)
        nn.init.zeros_(self.force_affine.bias)

    def forward(
        self,
        pointcloud: Tensor,
        patches: Mapping[str, Tensor],
        anchor_xyz: Tensor,
        hand_tokens: Tensor,
        *,
        rotation_world_from_eef: Tensor,
        tactile: Tensor,
    ) -> Tensor:
        """XYZ use world meters; rotation maps EEF axes to world."""
        memory, centers = patches["tokens"], patches["xyz"]
        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            anchors = anchor_xyz.float()
            rotation = rotation_world_from_eef.float()
            query_xyz = (anchors - anchors[:, :1]) @ rotation
            memory_xyz = (centers.float() - anchors[:, :1]) @ rotation
            relative, valid, descriptor = _local_surface(pointcloud[..., :3].float(), anchors)
            surface_relative = relative @ rotation[:, None]
            geometry = torch.cat(
                (descriptor[..., :3] @ rotation, descriptor[..., 3:]),
                -1,
            )

        local = self.surface_mlp(
            (surface_relative / 0.04).to(self.surface_mlp[0].weight.dtype)
        )
        with torch.autocast(device_type=local.device.type, enabled=False):
            local = local.float().masked_fill(~valid[..., None], 0.0)
            local = local.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)

        geometry_update = self.geometry_projection(
            torch.cat((local, geometry), -1).to(self.geometry_projection.weight.dtype)
        ).float()
        query = hand_tokens.float() + geometry_update

        context = self.context(query, memory, query_xyz, memory_xyz)
        x = hand_tokens.float() + (geometry_update + context.float())
        features = self.tactile_encoder(tactile).float()
        conditioning = torch.cat((features, tactile.float()), dim=-1)
        gamma, beta = self.force_affine(conditioning.to(self.force_affine.weight.dtype)).chunk(2, dim=-1)
        finger_x = x[:, 1:]
        normalized = self.query_norm(finger_x.to(self.query_norm.weight.dtype))
        fingers = finger_x.float() + gamma.float() * normalized.float() + beta.float()
        return torch.cat((x[:, :1].float(), fingers), dim=1)


class HandHistoryEncoder(nn.Module):
    def __init__(self, n_obs_steps: int) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.spatial_attn_norm = nn.LayerNorm(192)
        self.spatial_attn = nn.MultiheadAttention(192, 4, dropout=0.0, batch_first=True)
        self.spatial_ffn_norm = nn.LayerNorm(192)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192)
        )
        self.time_embedding = nn.Parameter(torch.empty(1, n_obs_steps, 192))
        nn.init.normal_(self.time_embedding, std=0.02)
        self.temporal_attn_norm = nn.LayerNorm(192)
        self.temporal_attn = nn.MultiheadAttention(192, 4, dropout=0.0, batch_first=True)
        self.temporal_ffn_norm = nn.LayerNorm(192)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192)
        )
        # MultiheadAttention uses True to forbid reading strictly future frames.
        self.register_buffer(
            "causal_mask",
            torch.ones(n_obs_steps, n_obs_steps, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        spatial = tokens.reshape(batch * self.n_obs_steps, 6, 192).float()
        normalized = self.spatial_attn_norm(spatial.to(self.spatial_attn_norm.weight.dtype))
        spatial = spatial + self.spatial_attn(
            normalized, normalized, normalized, need_weights=False
        )[0].float()
        normalized = self.spatial_ffn_norm(spatial.to(self.spatial_ffn_norm.weight.dtype))
        spatial = spatial + self.spatial_ffn(normalized).float()

        # Each temporal sequence contains one body slot from one batch item.
        temporal = (
            spatial.reshape(batch, self.n_obs_steps, 6, 192)
            .permute(0, 2, 1, 3)
            .reshape(batch * 6, self.n_obs_steps, 192)
        )
        temporal = temporal + self.time_embedding.float()
        normalized = self.temporal_attn_norm(temporal.to(self.temporal_attn_norm.weight.dtype))
        temporal = temporal + self.temporal_attn(
            normalized, normalized, normalized, attn_mask=self.causal_mask, need_weights=False
        )[0].float()
        normalized = self.temporal_ffn_norm(temporal.to(self.temporal_ffn_norm.weight.dtype))
        temporal = temporal + self.temporal_ffn(normalized).float()
        return temporal.reshape(batch, 6, self.n_obs_steps, 192).permute(0, 2, 1, 3)
