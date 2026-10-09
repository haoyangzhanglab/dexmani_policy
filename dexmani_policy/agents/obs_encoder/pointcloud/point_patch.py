"""Point-patch encoder with continuous 3D RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dexmani_policy.agents.obs_encoder.pointcloud.ops import (
    farthest_point_sample,
    index_points,
    knn_point,
    resolve_fps_random_config,
)
from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import PatchEncoder


class RotaryPositionEncoding3D(nn.Module):
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
    def apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        pairs = x.unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x * cos + rotated * sin).to(dtype)


class PointPatchBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
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
        qkv = self.qkv(self.norm1(x)).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = RotaryPositionEncoding3D.apply_rotary(q, cos, sin)
        k = RotaryPositionEncoding3D.apply_rotary(k, cos, sin)

        context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = x + self.proj(context)
        return x + self.mlp(self.norm2(x))


class PointPatchEncoder(nn.Module):
    """Encode XYZ[RGB] into patch_token [B, M, D] and patch_center [B, M, 3]."""

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
            raise ValueError(f"input_channels must be 3 (XYZ) or 6 (XYZRGB), but got {input_channels}")
        if min(token_channels, num_patches, group_size, depth, num_heads) <= 0:
            raise ValueError("channels, patch sizes, depth and num_heads must be positive")
        if token_channels % num_heads != 0:
            raise ValueError("token_channels must be divisible by num_heads")
        if not math.isfinite(mlp_ratio) or int(token_channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be finite and produce at least one hidden channel")

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
        self.rope = RotaryPositionEncoding3D(
            token_channels // num_heads,
            scale=rope_scale,
            base=rope_base,
        )
        self.blocks = nn.ModuleList(
            [PointPatchBlock(token_channels, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        if pointcloud.ndim != 3 or pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"Expected [B, N, C] with C >= {self.input_channels}, got {tuple(pointcloud.shape)}"
            )
        if pointcloud.size(1) < max(self.num_patches, self.group_size):
            raise ValueError("N must be at least max(num_patches, group_size)")

        xyz = pointcloud[..., :3].float()
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
        patch_input = relative_xyz
        if self.input_channels > 3:
            point_feature = pointcloud[..., 3 : self.input_channels]
            neighbor_feature = index_points(point_feature, neighbor_idx)
            patch_input = torch.cat((relative_xyz, neighbor_feature), dim=-1)

        local_token = self.patch_encoder(patch_input)
        cos, sin = self.rope(patch_center)

        patch_token = local_token
        for block in self.blocks:
            patch_token = block(patch_token, cos, sin)
        patch_token = self.norm(patch_token)

        outputs = {"patch_token": patch_token, "patch_center": patch_center}
        if return_intermediate:
            outputs.update(
                patch_center_idx=patch_center_idx,
                neighbor_idx=neighbor_idx,
                local_token=local_token,
            )
        return outputs

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.token_channels)


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    print("=== PointPatchEncoder Example ===")
    encoder = PointPatchEncoder(input_channels=6).eval()
    with torch.no_grad():
        out = encoder(pointcloud, return_intermediate=True)
    print("input:", tuple(pointcloud.shape))
    for name, value in out.items():
        print(f"{name}:", tuple(value.shape))
    print("out_dim:", encoder.out_dim)
    print("out_shape:", encoder.out_shape)


if __name__ == "__main__":
    example()
