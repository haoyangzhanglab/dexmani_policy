import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..pointcloud.rope import RotaryPositionEmbedding3D


@torch.no_grad()
def _sample_region_seeds(points: Tensor, count: int) -> Tensor:
    """Centroid-initialized FPS; unique indices break ties by input order."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.float()
        centroid = points.sum(1, keepdim=True) / points.shape[1]
        distance = (points - centroid).square().sum(-1)
        farthest = distance.argmax(-1)
        batch = torch.arange(points.shape[0], device=points.device)
        nearest = torch.full_like(distance, torch.inf)
        seeds = []

        for step in range(count):
            seeds.append(farthest)
            if step + 1 < count:
                distance = (points - points[batch, farthest, None]).square().sum(-1)
                nearest = torch.minimum(nearest, distance)
                nearest.scatter_(1, farthest[:, None], -torch.inf)
                farthest = nearest.argmax(-1)

        return torch.stack(seeds, 1)


class SceneTokenizer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.geometry_embed = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(), nn.Linear(64, channels)
        )
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, 128, bias=False)
        self.kv = nn.Linear(channels, 256, bias=False)
        self.out = nn.Linear(128, channels)
        self.refinement_gain = nn.Parameter(torch.full((channels,), 0.1))
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels), nn.GELU(), nn.Linear(2 * channels, channels)
        )
        self.ffn_gain = nn.Parameter(torch.full((channels,), 0.1))
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: Tensor, points: Tensor) -> Tensor:
        if tokens.shape[1] < 12:
            raise ValueError("SceneTokenizer requires at least twelve patches")
        points = points.float()
        batch, patches, _ = tokens.shape
        seed_indices = _sample_region_seeds(points, 12)
        anchors = points.gather(1, seed_indices[..., None].expand(-1, -1, 3))
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            distance2 = (anchors[:, :, None] - points[:, None]).square().sum(-1)
            geometry_bias = -torch.log1p(distance2 / 0.10**2)

        with torch.no_grad():
            nearest = distance2.topk(3, dim=1, largest=False).indices
            allowed = torch.zeros_like(distance2, dtype=torch.bool).scatter_(1, nearest, True)
            # Each unique seed index owns itself, even for coincident coordinates.
            ownership = torch.zeros_like(allowed).scatter_(2, seed_indices[..., None], True)
            allowed = torch.where(ownership.any(dim=1, keepdim=True), ownership, allowed)

        with torch.autocast(device_type=tokens.device.type, enabled=False):
            logits = geometry_bias.masked_fill(~allowed, -torch.inf)
            assignment = logits.softmax(dim=1)
            mass = assignment.sum(-1)
            weight = assignment / mass[..., None]
            pooled = weight @ tokens.float()
            centers = weight @ points
            # Centered moments avoid cancellation in E[xx^T] - E[x]E[x]^T.
            delta = points[:, None] - centers[:, :, None]
            covariance = (weight[..., None] * delta).transpose(-2, -1) @ delta
            normalized_covariance = covariance / 0.10**2
            upper = torch.stack(
                tuple(
                    normalized_covariance[..., i, j]
                    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
                ),
                -1,
            )
            descriptor = torch.cat((centers, upper), -1)

        geometry = self.geometry_embed(descriptor.to(self.geometry_embed[0].weight.dtype))
        scene = pooled + geometry.float()
        query = self.q(self.query_norm(scene.to(self.q.weight.dtype)))
        key, value = self.kv(self.memory_norm(tokens.to(self.kv.weight.dtype))).chunk(2, dim=-1)
        query = query.reshape(batch, 12, 4, 32).transpose(1, 2)
        key = key.reshape(batch, patches, 4, 32).transpose(1, 2)
        value = value.reshape(batch, patches, 4, 32).transpose(1, 2)
        bias = geometry_bias[:, None].expand(-1, 4, -1, -1).clone()
        bias[:, -1] = 0
        update = F.scaled_dot_product_attention(query, key, value, attn_mask=bias.to(query))
        update = self.out(update.transpose(1, 2).reshape(batch, 12, 128))

        # FP32 residuals preserve small learned updates under autocast.
        scene = scene + self.refinement_gain.float() * update.float()
        normalized = self.ffn_norm(scene.to(self.ffn_norm.weight.dtype))
        scene = scene + self.ffn_gain.float() * self.ffn(normalized).float()
        return self.norm(scene.to(self.norm.weight.dtype))


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
        attention_bias = attention_bias.to(q)
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
        self.tactile_norm = nn.LayerNorm(tactile_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(192 + tactile_dim, tactile_dim), nn.SiLU(),
            nn.Linear(tactile_dim, 192),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(192 + tactile_dim + 3, tactile_dim),
            nn.SiLU(),
            nn.Linear(tactile_dim, 1),
        )

        # Nonzero residual weights let tactile and gate paths learn from step one.
        nn.init.normal_(self.fusion_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.fusion_mlp[-1].bias)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, math.log(0.1 / 0.9))

    def forward(
        self,
        pointcloud: Tensor,
        patches: Mapping[str, Tensor],
        anchor_xyz: Tensor,
        hand_tokens: Tensor,
        *,
        rotation_world_from_eef: Tensor,
        query_context: Tensor,
        tactile: Tensor,
    ) -> Tensor:
        """XYZ use world meters; rotation maps EEF axes to world; scene context is [N,D]."""
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
        query = hand_tokens.float() + geometry_update + query_context[:, None].float()

        context = self.context(query, memory, query_xyz, memory_xyz)
        x = hand_tokens.float() + (geometry_update + context.float())
        local = self.tactile_encoder(tactile).float()
        summary = local.mean(dim=1)

        query = self.query_norm(x.to(self.query_norm.weight.dtype))
        memory = torch.cat((summary[:, None], local), dim=1)
        memory = self.tactile_norm(memory.to(self.tactile_norm.weight.dtype))
        features = torch.cat((query, memory), dim=-1)
        delta = self.fusion_mlp(features.to(self.fusion_mlp[0].weight.dtype)).float()
        gate_input = torch.cat(
            (features, geometry[..., 3:].float()), -1,
        )
        gate = self.gate_mlp(gate_input.to(self.gate_mlp[0].weight.dtype)).float().sigmoid()
        return x + gate * delta


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
