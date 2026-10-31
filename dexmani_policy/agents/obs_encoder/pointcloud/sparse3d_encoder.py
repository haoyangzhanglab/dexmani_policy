"""Sparse, spatially anchored 3D tokens from points in a common metric frame."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dexmani_policy.agents.modules import build_mlp

from .ops import index_points, sample_point_patches, load_pointcloud_ops
from .rope import RotaryPositionEmbedding3D


class GeometricPatchEmbedding(nn.Module):
    """Member features, Gaussian pooling on shared kNN support, and geometric moments."""

    def __init__(self, input_channels: int, channels: int, geometry_scale: float):
        super().__init__()
        self.geometry_scale = geometry_scale
        self.register_buffer("wavelengths", torch.logspace(math.log10(0.02), math.log10(0.32), 8))
        self.register_buffer("pool_sigmas", torch.tensor([0.04, 0.12]))
        self.member_mlp = build_mlp([input_channels + 1 + 48, 64, 128], activate_last=True)
        self.refine = build_mlp([256, 128, 128])
        # Max, two weighted means, covariance upper triangle, and patch radius.
        self.projection = build_mlp([3 * 128 + 7, channels, channels])

    def forward(self, relative_xyz: torch.Tensor, attributes: torch.Tensor):
        # Geometry, phases, moments and weighted sums accumulate in FP32.
        with torch.autocast(device_type=relative_xyz.device.type, enabled=False):
            relative_xyz = relative_xyz.float()
            scaled = relative_xyz / self.geometry_scale
            distance = torch.linalg.vector_norm(relative_xyz, dim=-1, keepdim=True)
            angles = relative_xyz.unsqueeze(-1) * (2 * math.pi / self.wavelengths.float())
            fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
            features = torch.cat(
                (scaled, distance / self.geometry_scale, fourier, attributes.float()),
                dim=-1,
            )

            weights = (-0.5 * (distance / self.pool_sigmas.float()).square()).softmax(dim=2)

            centered = scaled - scaled.mean(dim=2, keepdim=True)
            covariance = centered.transpose(-1, -2) @ centered / centered.size(2)
            covariance = torch.stack(
                tuple(
                    covariance[..., i, j]
                    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
                ),
                dim=-1,
            )
            radius = distance.amax(dim=2)
            moments = torch.cat((covariance, radius / self.geometry_scale), dim=-1)

        member = self.member_mlp(features.to(self.member_mlp[0].weight.dtype))
        context = member.amax(dim=2, keepdim=True)
        # Project the shared context once per patch, keeping the existing refine parameters.
        linear = self.refine[0]
        d = member.shape[-1]
        member = F.linear(member, linear.weight[:, :d], linear.bias) + F.linear(
            context, linear.weight[:, d:], None
        )
        for i in range(1, len(self.refine)):
            member = self.refine[i](member)

        with torch.autocast(device_type=member.device.type, enabled=False):
            weighted = torch.einsum("bmks,bmkd->bmsd", weights, member.float()).flatten(-2)
            # Concatenate in FP32 to avoid mixed-dtype addmm in Torch 2.4 Inductor.
            pooled = torch.cat((member.amax(dim=2).float(), weighted, moments), dim=-1)

        return self.projection(pooled.to(member.dtype))


class GlobalTransformerBlock(nn.Module):
    """Global Pre-LN attention with Q/K normalization before 3D RoPE."""

    def __init__(self, channels: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.ffn = build_mlp([channels, hidden, channels], normalize=False)
        self.attention_gain = nn.Parameter(torch.full((channels,), 0.1))
        self.ffn_gain = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, m, d = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(b, m, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = RotaryPositionEmbedding3D.apply_rotary(self.q_norm(q), cos, sin)
        k = RotaryPositionEmbedding3D.apply_rotary(self.k_norm(k), cos, sin)
        update = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        update = update.transpose(1, 2).reshape(b, m, d)

        tokens = tokens + self.attention_gain * self.projection(update)
        return tokens + self.ffn_gain * self.ffn(self.norm2(tokens))


class Sparse3DEncoder(nn.Module):
    """Encode finite, unpadded metric point clouds into sparse contextual tokens."""

    def __init__(
        self,
        input_channels: int = 6,
        channels: int = 384,
        num_patches: int = 96,
        group_size: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        geometry_scale: float = 0.10,
        rope_scale: float = 0.05,
        rope_base: float = 100.0,
        fps_random_config: dict | None = None,
    ):
        config = locals().copy()
        super().__init__()
        load_pointcloud_ops()
        self.effective_config = {
            key: value for key, value in config.items()
            if key not in {"self", "input_channels", "fps_random_config", "__class__"}
        }
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
        sizes = (channels, num_patches, group_size, depth, num_heads)
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError("sizes must be positive integers")
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if (channels // num_heads) % 6:
            raise ValueError("channels // num_heads must be divisible by 6")
        if int(channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        if not math.isfinite(geometry_scale) or geometry_scale <= 0:
            raise ValueError("geometry_scale must be finite and positive")

        self.input_channels = input_channels
        self.channels = channels
        self.num_patches = num_patches
        self.group_size = group_size
        self.fps_random_config = dict(fps_random_config or {})

        self.patch_embedding = GeometricPatchEmbedding(input_channels, channels, geometry_scale)
        self.position_embedding = build_mlp([3, 128, channels], normalize=False)
        self.rope = RotaryPositionEmbedding3D(channels // num_heads, rope_scale, rope_base)
        self.blocks = nn.ModuleList(
            [GlobalTransformerBlock(channels, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels)

    def encode_global(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Compile global attention; keep sampling and local feature extraction eager."""
        for block in self.blocks:
            tokens = block(tokens, cos, sin)
        return self.norm(tokens)

    def encode_patches(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample once and encode local geometry, absolute position and shared 3D RoPE."""
        xyz = pointcloud[..., :3].float().contiguous()
        with (
            torch.no_grad(),
            torch.autocast(device_type=pointcloud.device.type, enabled=False),
        ):
            centers, _, neighbors = sample_point_patches(
                xyz,
                self.num_patches,
                self.group_size,
                self.fps_random_config,
                self.training,
            )
            cos, sin = self.rope(centers)

        relative_xyz = index_points(xyz, neighbors) - centers.unsqueeze(2)
        attributes = index_points(pointcloud[..., 3 : self.input_channels], neighbors)
        tokens = self.patch_embedding(relative_xyz, attributes)
        # Preserve absolute metric position as content alongside relative geometry/RoPE.
        position = self.position_embedding(centers.to(self.position_embedding[0].weight.dtype))
        tokens = tokens + position.to(tokens.dtype)
        return {
            "tokens": tokens,
            "xyz": centers,
            "neighbor_indices": neighbors,
            "relative_xyz": relative_xyz,
            "rope_cos": cos,
            "rope_sin": sin,
        }

    def forward(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = self.encode_patches(pointcloud)
        return {
            "tokens": self.encode_global(patches["tokens"], patches["rope_cos"], patches["rope_sin"]),
            "xyz": patches["xyz"],
        }

    @property
    def out_dim(self) -> int:
        return self.channels
