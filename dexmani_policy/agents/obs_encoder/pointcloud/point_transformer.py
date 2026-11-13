"""Lightweight, spatially anchored 3D tokens from points in a common metric frame."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dexmani_policy.agents.modules import build_mlp

from .ops import index_points, sample_point_patches
from .rope import RotaryPositionEmbedding3D


class PointPatchEmbedding(nn.Module):
    def __init__(self, input_channels: int, channels: int, geometry_scale: float):
        super().__init__()
        self.geometry_scale = geometry_scale
        self.point_mlp = build_mlp([input_channels, 64, 64], activate_last=True)
        self.projection = nn.Linear(64, channels)

    def forward(self, relative_xyz: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        points = torch.cat(
            (relative_xyz.float() / self.geometry_scale, attributes.float()), dim=-1,
        )
        features = self.point_mlp(points.to(self.point_mlp[0].weight.dtype))
        return self.projection(features.amax(dim=2))


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

    def forward(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, m, d = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(b, m, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = RotaryPositionEmbedding3D.apply_rotary(self.q_norm(q), cos, sin)
        k = RotaryPositionEmbedding3D.apply_rotary(self.k_norm(k), cos, sin)
        update = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        update = update.transpose(1, 2).reshape(b, m, d)

        tokens = tokens + self.attention_gain.to(tokens.dtype) * self.projection(update)
        return tokens + self.ffn_gain.to(tokens.dtype) * self.ffn(self.norm2(tokens))


class PointPatchEncoder(nn.Module):
    """Encode local geometry and absolute position at FPS patch centers."""

    def __init__(
        self,
        input_channels: int = 6,
        channels: int = 192,
        num_patches: int = 128,
        group_size: int = 32,
        geometry_scale: float = 0.10,
        fps_random_config: dict | None = None,
    ):
        super().__init__()
        self.effective_config = dict(
            channels=channels, num_patches=num_patches, group_size=group_size,
            geometry_scale=geometry_scale,
        )
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
        sizes = (channels, num_patches, group_size)
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError("sizes must be positive integers")
        if not math.isfinite(geometry_scale) or geometry_scale <= 0:
            raise ValueError("geometry_scale must be finite and positive")

        self.input_channels = input_channels
        self.out_dim = channels
        self.num_patches = num_patches
        self.group_size = group_size
        self.fps_random_config = dict(fps_random_config or {})

        self.patch_embedding = PointPatchEmbedding(input_channels, channels, geometry_scale)
        self.position_embedding = build_mlp([3, 128, channels], normalize=False)

    def forward(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz = pointcloud[..., :3].float().contiguous()
        with torch.autocast(device_type=pointcloud.device.type, enabled=False):
            centers, neighbors = sample_point_patches(
                xyz,
                self.num_patches,
                self.group_size,
                self.fps_random_config,
                self.training,
            )

        relative_xyz = index_points(xyz, neighbors) - centers.unsqueeze(2)
        attributes = index_points(pointcloud[..., 3 : self.input_channels], neighbors)
        tokens = self.patch_embedding(relative_xyz, attributes)
        # Preserve absolute metric position alongside relative geometry.
        position = self.position_embedding(centers.to(self.position_embedding[0].weight.dtype))
        tokens = tokens + position.to(tokens.dtype)
        return {"tokens": tokens, "xyz": centers}


def _encode_global(
    tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    blocks: nn.ModuleList, norm: nn.LayerNorm,
) -> torch.Tensor:
    for block in blocks:
        tokens = block(tokens, cos, sin)
    return norm(tokens)


class GlobalPointTransformer(nn.Module):
    """Contextualize spatial tokens using metric 3D RoPE."""

    def __init__(
        self, channels: int = 384, depth: int = 2, num_heads: int = 8,
        mlp_ratio: float = 2.0, rope_scale: float = 0.05, rope_base: float = 100.0,
    ):
        super().__init__()
        self.effective_config = dict(
            channels=channels, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, rope_scale=rope_scale, rope_base=rope_base,
        )
        if any(type(value) is not int or value <= 0 for value in (channels, depth, num_heads)):
            raise ValueError("sizes must be positive integers")
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if int(channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        self.out_dim = channels
        self.rope = RotaryPositionEmbedding3D(channels // num_heads, rope_scale, rope_base)
        self.blocks = nn.ModuleList(
            [GlobalTransformerBlock(channels, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope(xyz)
        return _encode_global(tokens, cos, sin, self.blocks, self.norm)


class PointTransformerEncoder(PointPatchEncoder):
    """Encode finite, unpadded metric point clouds into sparse contextual tokens."""

    def __init__(
        self,
        input_channels: int = 6,
        channels: int = 192,
        num_patches: int = 128,
        group_size: int = 32,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        geometry_scale: float = 0.10,
        rope_scale: float = 0.05,
        rope_base: float = 100.0,
        fps_random_config: dict | None = None,
    ):
        super().__init__(
            input_channels=input_channels, channels=channels, num_patches=num_patches,
            group_size=group_size, geometry_scale=geometry_scale,
            fps_random_config=fps_random_config,
        )
        global_encoder = GlobalPointTransformer(
            channels, depth, num_heads, mlp_ratio, rope_scale, rope_base,
        )
        self.effective_config.update(global_encoder.effective_config)
        # Reuse global attention without adding another level to the state dict.
        self.rope = global_encoder.rope
        self.blocks = global_encoder.blocks
        self.norm = global_encoder.norm

    def forward(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = super().forward(pointcloud)
        cos, sin = self.rope(patches["xyz"])
        return {
            "tokens": _encode_global(patches["tokens"], cos, sin, self.blocks, self.norm),
            "xyz": patches["xyz"],
        }
