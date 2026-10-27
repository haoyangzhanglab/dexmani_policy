"""Sparse, spatially anchored 3D tokens from points in a common metric frame."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .ops import index_points, sample_point_patches, load_pointcloud_ops
from .pointnet import build_mlp


def _build_local_graph(
    centers: torch.Tensor, geometry_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # A single MxM distance matrix for a small, fixed anchor graph.
    num_patches = centers.shape[1]
    distances = torch.cdist(centers, centers, compute_mode="donot_use_mm_for_euclid_dist")
    diagonal = torch.eye(num_patches, dtype=torch.bool, device=centers.device)
    local_indices = (
        distances.masked_fill(diagonal[None], float("inf"))
        .topk(min(12, num_patches - 1), largest=False, sorted=False)
        .indices
    )
    offsets = (index_points(centers, local_indices) - centers.unsqueeze(2)) / geometry_scale
    edges = torch.cat((offsets, torch.linalg.vector_norm(offsets, dim=-1, keepdim=True)), dim=-1)
    return local_indices, edges


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
        context = member.amax(dim=2, keepdim=True).expand_as(member)
        member = self.refine(torch.cat((member, context), dim=-1))

        with torch.autocast(device_type=member.device.type, enabled=False):
            weighted = torch.einsum("bmks,bmkd->bmsd", weights, member.float()).flatten(-2)
            # Concatenate in FP32 to avoid mixed-dtype addmm in Torch 2.4 Inductor.
            pooled = torch.cat((member.amax(dim=2).float(), weighted, moments), dim=-1)

        return self.projection(pooled.to(member.dtype)), radius


class RotaryPositionEmbedding3D(nn.Module):
    """Continuous metric XYZ RoPE; relative-position encoding, not SE(3) equivariance."""

    def __init__(self, head_dim: int, scale: float = 0.05, base: float = 100.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 6:
            raise ValueError("head_dim must be positive and divisible by 6")
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(base) or base <= 1:
            raise ValueError("RoPE requires finite scale > 0 and base > 1")
        axis_dim = head_dim // 3
        self.scale = scale
        self.register_buffer(
            "inv_freq",
            base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim),
            persistent=False,
        )

    def forward(self, xyz: torch.Tensor):
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            # A shared origin improves phase precision without changing pairwise offsets.
            positions = (xyz.float() - xyz.float().mean(dim=1, keepdim=True)) / self.scale
            angles = positions.unsqueeze(-1) * self.inv_freq.float()
            angles = angles.repeat_interleave(2, dim=-1).flatten(-2)
            return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        pairs = x.float().unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x.float() * cos + rotated * sin).to(x.dtype)


class LocalGeometricTransformerBlock(nn.Module):
    """Local geometric attention over distinct patch indices, with pre-gather QKV."""

    def __init__(self, channels: int):
        super().__init__()
        self.heads = 4
        self.head_dim = 32
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * 128)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.edge_mlp = build_mlp([4, 64, 128 + self.heads], normalize=False)
        self.projection = nn.Linear(128, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = build_mlp([channels, 2 * channels, channels], normalize=False)
        self.attention_gain = nn.Parameter(torch.full((channels,), 0.1))
        self.ffn_gain = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, tokens: torch.Tensor, indices: torch.Tensor, edges: torch.Tensor):
        b, m, _ = tokens.shape
        neighbors = indices.shape[-1]

        q, k, v = self.qkv(self.norm1(tokens)).chunk(3, dim=-1)
        q = self.q_norm(q.reshape(b, m, self.heads, self.head_dim))
        k = self.k_norm(k.reshape(b, m, self.heads, self.head_dim))
        k = index_points(k, indices).permute(0, 1, 3, 2, 4)
        v = (
            index_points(v, indices)
            .reshape(b, m, neighbors, self.heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )

        relative_value, bias = self.edge_mlp(edges.to(self.edge_mlp[0].weight.dtype)).split(
            (128, self.heads), dim=-1
        )
        relative_value = relative_value.reshape(b, m, neighbors, self.heads, self.head_dim).permute(
            0, 1, 3, 2, 4
        )

        logits = (q.float().unsqueeze(-2) * k.float()).sum(-1) / math.sqrt(self.head_dim)
        logits = logits + bias.float().permute(0, 1, 3, 2)
        weights = logits.softmax(dim=-1).to(v.dtype)
        update = (
            (weights.unsqueeze(-1) * (v + relative_value.to(v.dtype))).sum(-2).reshape(b, m, 128)
        )

        tokens = tokens + self.attention_gain * self.projection(update)
        return tokens + self.ffn_gain * self.ffn(self.norm2(tokens))


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

    @staticmethod
    def validate_normalization(spec):
        if spec.get("point_cloud") != "identity":
            raise ValueError("Sparse3DEncoder requires normalization.point_cloud: identity (metric XYZ)")

    def __init__(
        self,
        input_channels: int = 6,
        channels: int = 384,
        num_patches: int = 192,
        group_size: int = 32,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
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
            if key not in {"self", "input_channels", "__class__"}
        }
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
        sizes = (channels, num_patches, group_size, depth, num_heads)
        if any(type(value) is not int or value <= 0 for value in sizes) or num_patches < 2:
            raise ValueError("sizes must be positive integers and num_patches >= 2")
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if int(channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        if not math.isfinite(geometry_scale) or geometry_scale <= 0:
            raise ValueError("geometry_scale must be finite and positive")

        self.input_channels = input_channels
        self.channels = channels
        self.num_patches = num_patches
        self.group_size = group_size
        self.geometry_scale = float(geometry_scale)
        self.fps_random_config = dict(fps_random_config or {})

        self.patch_embedding = GeometricPatchEmbedding(input_channels, channels, geometry_scale)
        self.local_relation = LocalGeometricTransformerBlock(channels)
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

    def forward(
        self, pointcloud: torch.Tensor, *, return_geometry: bool = False
    ) -> dict[str, torch.Tensor]:
        if (
            pointcloud.ndim != 3
            or pointcloud.shape[0] == 0
            or pointcloud.shape[-1] < self.input_channels
        ):
            raise ValueError("pointcloud must be nonempty [B,N,C] with enough channels")
        if not pointcloud.is_floating_point():
            raise ValueError("pointcloud must be floating point")
        if not torch.isfinite(pointcloud[..., : self.input_channels]).all():
            raise ValueError("pointcloud must contain only finite observed points")

        xyz = pointcloud[..., :3].float().contiguous()
        with (
            torch.no_grad(),
            torch.autocast(device_type=pointcloud.device.type, enabled=False),
        ):
            centers, center_indices, neighbors = sample_point_patches(
                xyz,
                self.num_patches,
                self.group_size,
                self.fps_random_config,
                self.training,
            )
            local_indices, edges = _build_local_graph(centers, self.geometry_scale)
            cos, sin = self.rope(centers)

        relative_xyz = index_points(xyz, neighbors) - centers.unsqueeze(2)
        attributes = index_points(pointcloud[..., 3 : self.input_channels], neighbors)
        tokens, radius = self.patch_embedding(relative_xyz, attributes)
        tokens = self.local_relation(tokens, local_indices, edges)
        # Preserve absolute metric position as content alongside relative geometry/RoPE.
        position = self.position_embedding(centers.to(self.position_embedding[0].weight.dtype))
        tokens = tokens + position.to(tokens.dtype)
        outputs = {
            "tokens": self.encode_global(tokens, cos, sin),
            "xyz": centers,
            "mask": torch.ones(centers.shape[:2], device=xyz.device, dtype=torch.bool),
        }
        if return_geometry:
            outputs.update(
                center_indices=center_indices,
                neighbor_indices=neighbors,
                patch_radius=radius,
            )
        return outputs

    @property
    def out_dim(self) -> int:
        return self.channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return self.num_patches, self.channels
