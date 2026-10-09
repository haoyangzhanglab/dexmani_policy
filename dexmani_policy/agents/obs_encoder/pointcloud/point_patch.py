"""Lightweight point-patch encoder with continuous 3D RoPE.

Patch tokenization follows the Point-MAE/R3D family; 3D positional attention
follows Sparse2Act. The module only encodes point clouds and keeps patch centers.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.obs_encoder.pointcloud.ops import (
    farthest_point_sample,
    index_points,
    knn_point,
    resolve_fps_random_config,
)
from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import PatchEncoder


class _RotaryPositionEncoding3D(nn.Module):
    def __init__(
        self,
        head_dim: int,
        scale: float = 1.0,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim <= 0 or head_dim % 6 != 0:
            raise ValueError("head_dim must be positive and divisible by 6")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(base) or base <= 1:
            raise ValueError("base must be finite and greater than 1")

        axis_dim = head_dim // 3
        self.scale = float(scale)
        self.register_buffer(
            "inv_freq",
            base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim),
            persistent=False,
        )

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = xyz.float() / self.scale
        angles = positions.unsqueeze(-1) * self.inv_freq
        angles = angles.repeat_interleave(2, dim=-1).flatten(start_dim=-2)
        return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    @staticmethod
    def apply(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        pairs = x.unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x * cos + rotated * sin).to(dtype)


class _PointPatchBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)

        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        head_dim = channels // self.num_heads

        qkv = self.qkv(self.norm1(x)).reshape(
            batch_size, num_tokens, 3, self.num_heads, head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = _RotaryPositionEncoding3D.apply(q, cos, sin)
        k = _RotaryPositionEncoding3D.apply(k, cos, sin)

        context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = x + self.proj(context)
        return x + self.mlp(self.norm2(x))


class PointPatchEncoder(nn.Module):
    """Encode a point cloud into spatial patch tokens.

    XYZ must occupy the first three channels. Additional channels are treated as
    point features. FPS/KNN operate directly in the supplied XYZ coordinate system.
    """

    supports_global_token = False
    supports_intermediate_outputs = True
    requires_fixed_num_points = False

    def __init__(
        self,
        input_channels: int = 6,
        token_channels: int = 192,
        num_patches: int = 128,
        group_size: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        rope_scale: float = 1.0,
        rope_base: float = 10000.0,
        fps_random_config: dict | None = None,
    ):
        super().__init__()
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
        if num_patches <= 0 or group_size <= 0:
            raise ValueError("num_patches and group_size must be positive")
        if depth <= 0 or num_heads <= 0:
            raise ValueError("depth and num_heads must be positive")
        if token_channels % num_heads != 0:
            raise ValueError("token_channels must be divisible by num_heads")
        if (token_channels // num_heads) % 6 != 0:
            raise ValueError("attention head dimension must be divisible by 6")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")

        self.input_channels = input_channels
        self.token_channels = token_channels
        self.num_patches = num_patches
        self.group_size = group_size
        self.fps_random_config = fps_random_config or {}

        self.patch_encoder = PatchEncoder(
            in_channels=input_channels,
            out_channels=token_channels,
            hidden_dims=[128, 256],
        )
        self.rope = _RotaryPositionEncoding3D(
            token_channels // num_heads,
            scale=rope_scale,
            base=rope_base,
        )
        self.blocks = nn.ModuleList(
            [_PointPatchBlock(token_channels, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_intermediate: bool = False,
    ):
        if pointcloud.ndim != 3:
            raise ValueError(
                f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}"
            )
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, "
                f"but input_channels={self.input_channels}"
            )
        if pointcloud.size(1) < max(self.num_patches, self.group_size):
            raise ValueError(
                "pointcloud must contain at least max(num_patches, group_size) points"
            )

        xyz = pointcloud[..., :3].float()
        point_feature = pointcloud[..., 3 : self.input_channels]

        with torch.no_grad():
            fps_config = resolve_fps_random_config(
                self.fps_random_config, self.training
            )
            patch_center, patch_center_idx = farthest_point_sample(
                xyz, self.num_patches, **fps_config
            )
            neighbor_idx = knn_point(self.group_size, xyz, patch_center)

        neighbor_xyz = index_points(xyz, neighbor_idx)
        relative_xyz = neighbor_xyz - patch_center.unsqueeze(2)
        if self.input_channels > 3:
            neighbor_feature = index_points(point_feature, neighbor_idx)
            patch_input = torch.cat((relative_xyz, neighbor_feature), dim=-1)
        else:
            patch_input = relative_xyz

        local_token = self.patch_encoder(patch_input)
        cos, sin = self.rope(patch_center)

        patch_token = local_token
        for block in self.blocks:
            patch_token = block(patch_token, cos, sin)
        patch_token = self.norm(patch_token)

        outputs = [patch_token, patch_center]
        if return_intermediate:
            intermediate_outputs: Dict[str, torch.Tensor] = {
                "patch_center_idx": patch_center_idx,
                "neighbor_idx": neighbor_idx,
                "local_token": local_token,
            }
            outputs.append(intermediate_outputs)
        return tuple(outputs)

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.token_channels)
