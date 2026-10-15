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
from dexmani_policy.agents.obs_encoder.interaction.point_image_fusion import PointImageFusion
from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import PatchEncoder


class CompactPatchEncoder(nn.Module):
    """Uni3D-style within-patch context, with wide projection after pooling.

    Keep the local max/broadcast/refine operation, but run it at 64 channels.
    Max and mean are over the K members only, never over the M spatial patches.
    This is a scratch-training alternative, not a pretrained Uni3D replacement.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, out_channels), nn.LayerNorm(out_channels), nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, point_patches: torch.Tensor) -> torch.Tensor:
        x = self.point_mlp(point_patches)
        pooled = x.max(dim=2, keepdim=True).values
        x = self.refine(torch.cat((x, pooled.expand_as(x)), dim=-1))
        # Accumulate the mean in FP32 under autocast, then restore feature dtype.
        mean = x.float().mean(dim=2).to(x.dtype)
        return self.proj(torch.cat((x.max(dim=2).values, mean), dim=-1))


class LocalPatchMixer(nn.Module):
    """One low-width geometric interaction on the existing patch anchors.

    Projects tokens once before gathering neighbors. No dynamic feature graph,
    second FPS, pooling of spatial tokens, or action-conditioned geometry.
    The coordinate scale is a fixed divisor, NOT a radius cutoff.
    """

    def __init__(self, channels: int, neighbors: int, scale: float = 0.1):
        super().__init__()
        if type(neighbors) is not int or neighbors < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("positive local neighbors and finite positive scale required")
        self.neighbors = neighbors
        self.scale = float(scale)
        width = min(64, channels)
        self.norm = nn.LayerNorm(channels)
        self.input_proj = nn.Linear(channels, width)
        self.position_mlp = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(), nn.Linear(16, width),
        )
        self.output_proj = nn.Linear(2 * width, channels)
        self.gain = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if self.neighbors >= centers.size(1):
            raise ValueError("local neighbors must be smaller than the patch count")
        # Small fixed anchor graph. Exclude self by INDEX, not zero distance:
        # distinct sampled points may share the same coordinates.
        with torch.no_grad(), torch.autocast(device_type=centers.device.type, enabled=False):
            distance = torch.cdist(
                centers.float(), centers.float(), compute_mode="donot_use_mm_for_euclid_dist"
            )
            diagonal = torch.eye(centers.size(1), device=centers.device, dtype=torch.bool)
            distance = distance.masked_fill(diagonal.unsqueeze(0), float("inf"))
            neighbor_idx = distance.topk(self.neighbors, largest=False, sorted=False).indices
        projected = self.input_proj(self.norm(tokens))
        relative_xyz = (index_points(centers.float(), neighbor_idx) - centers.float().unsqueeze(2)) / self.scale
        relative_distance = torch.linalg.vector_norm(relative_xyz, dim=-1, keepdim=True)
        geometry = torch.cat((relative_xyz, relative_distance), dim=-1)
        position = self.position_mlp(geometry.to(self.position_mlp[0].weight.dtype))
        message = F.gelu(
            index_points(projected, neighbor_idx) - projected.unsqueeze(2) + position
        )
        mean = message.float().mean(dim=2).to(message.dtype)
        update = self.output_proj(torch.cat((message.max(dim=2).values, mean), dim=-1))
        return tokens + self.gain * update


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


class GeometryPatchEncoder(nn.Module):
    """Encode XYZ[RGB] into patch_token [B,M,D] and patch_center [B,M,3].

    semantic_channels 启用 local_token 后的可选图像语义残差；point_uv 必须
    对应同一输入点序列，图像特征/投影每次观测只计算一次。默认参数保留纯几何路径。

    patch_encoder_type="uni3d" and local_mixer_neighbors=0 preserve the original
    parameter names, shapes and computation. Opt into "compact" to reduce the
    per-member MLP width; optionally add one local mixer before global RoPE
    attention. local_token always describes only the original K raw members.
    XYZ and local_mixer_scale use the same metric coordinate system.
    """

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
        semantic_channels: int | None = None,
        semantic_gate_init: float = 0.1,
        *,
        patch_encoder_type: str = "uni3d",
        local_mixer_neighbors: int = 0,
        local_mixer_scale: float = 0.1,
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

        if patch_encoder_type not in {"uni3d", "compact"}:
            raise ValueError("patch_encoder_type must be uni3d or compact")
        if type(local_mixer_neighbors) is not int or not 0 <= local_mixer_neighbors < num_patches:
            raise ValueError("local_mixer_neighbors must be an integer in [0, num_patches)")
        if not math.isfinite(local_mixer_scale) or local_mixer_scale <= 0:
            raise ValueError("local_mixer_scale must be finite and positive")

        self.input_channels = input_channels
        self.token_channels = token_channels
        self.num_patches = num_patches
        self.group_size = group_size
        self.fps_random_config = fps_random_config or {}

        self.patch_encoder_type = patch_encoder_type
        if patch_encoder_type == "uni3d":
            self.patch_encoder = PatchEncoder(
                in_channels=input_channels,
                out_channels=token_channels,
                hidden_dims=[128, 256],
            )
        else:
            self.patch_encoder = CompactPatchEncoder(input_channels, token_channels)
        self.local_mixer = (
            LocalPatchMixer(token_channels, local_mixer_neighbors, local_mixer_scale)
            if local_mixer_neighbors else None
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
        self.semantic_fusion = None
        if semantic_channels is not None:
            self.semantic_fusion = PointImageFusion(
                image_channels=semantic_channels,
                token_channels=token_channels,
                gate_init=semantic_gate_init,
            )

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_intermediate: bool = False,
        *,
        image_tokens: torch.Tensor | None = None,
        point_uv: torch.Tensor | None = None,
        view_weight: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        patch_grid_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        semantic_inputs = (image_tokens, point_uv, view_weight, image_hw, patch_grid_size)
        use_semantics = any(value is not None for value in semantic_inputs)
        if use_semantics and (self.semantic_fusion is None or any(value is None for value in semantic_inputs)):
            raise ValueError("set semantic_channels and provide all five semantic inputs")
        if pointcloud.ndim != 3 or pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"Expected [B, N, C] with C >= {self.input_channels}, got {tuple(pointcloud.shape)}"
            )
        if use_semantics and (point_uv.ndim != 4 or point_uv.shape[2] != pointcloud.shape[1]):
            raise ValueError("point_uv must preserve the input point order and count: [B,V,N,2]")
        if pointcloud.size(1) < max(self.num_patches, self.group_size):
            raise ValueError("N must be at least max(num_patches, group_size)")

        xyz = pointcloud[..., :3].float()
        if not torch.isfinite(pointcloud[..., :self.input_channels]).all():
            raise ValueError("pointcloud must contain finite observed points before patch sampling")
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
        patch_token = local_token
        semantic_outputs = None
        if use_semantics:
            semantic_outputs = self.semantic_fusion(
                local_token, neighbor_idx, image_tokens, point_uv, view_weight,
                image_hw, patch_grid_size, return_intermediate=return_intermediate,
            )
            patch_token = semantic_outputs["fused_token"]
        # Keep local_token and fused_local_token tied to raw patch membership.
        # Only the contextual patch_token path mixes neighboring patches.
        if self.local_mixer is not None:
            patch_token = self.local_mixer(patch_token, patch_center)
        cos, sin = self.rope(patch_center)
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
            if semantic_outputs is not None:
                outputs["fused_local_token"] = semantic_outputs["fused_token"]
                outputs.update({key: value for key, value in semantic_outputs.items() if key != "fused_token"})
        return outputs

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.token_channels)
