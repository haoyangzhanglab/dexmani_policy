"""KPConv-based feature pyramid network — faithful port of OPFA's
``geotransformer/modules/kpconv/`` + ``experiments/realworld/backbone.py``.

Architecture (identical to OPFA):
  KPConv → GroupNorm → LeakyReLU  (ConvBlock)
  Bottleneck with maxpool stride  (ResidualBlock)
  4-stage encoder: 64→128→256→512→1024  (KPConvFPN)

The pre-computed kernel point geometry is loaded from ``k_015_center_3D.ply``
bundled alongside this file.
"""

from __future__ import annotations

import math
import os.path as osp
from os.path import exists, join

import numpy as np
import torch
import torch.nn as nn

# Use official OPFA index_select when available; fall back to pure-PyTorch.
try:
    from dexmani_policy.agents.opfa._geotransformer_bridge import _ensure_geotransformer
    _ensure_geotransformer()
    from geotransformer.modules.ops import index_select
except ImportError:
    from dexmani_policy.agents.opfa.point_ops import index_select


# =========================================================================
# Kernel point loading (port of kernel_points.py — matplotlib removed)
# =========================================================================

def load_kernels(
    radius: float,
    num_kpoints: int,
    dimension: int = 3,
    fixed: str = "center",
) -> np.ndarray:
    """Load pre-computed kernel point geometry from a ``.ply`` file.

    The ``.ply`` file is read from the ``dispositions/`` directory next to
    the bundled ``k_015_center_3D.ply``.  If the file does not exist (e.g.
    wrong *num_kpoints*), a **RuntimeError** is raised — KPConv kernel
    generation requires ``matplotlib`` + ``open3d`` which are intentionally
    NOT imported at module level.

    Args:
        radius: kernel sphere radius (applied as scale factor).
        num_kpoints: number of kernel points (must be 15 for bundled file).
        dimension: spatial dimension (must be 3).
        fixed: fixing mode — "center" anchors one point at the origin.

    Returns:
        ``(num_kpoints, dimension)`` float32 array.
    """
    kernel_dir = osp.join(osp.dirname(osp.abspath(__file__)), "dispositions")
    kernel_file = join(kernel_dir, f"k_{num_kpoints:03d}_{fixed}_{dimension}D.ply")

    if not exists(kernel_file):
        # Fallback: try to generate via open3d (requires optional deps)
        try:
            import open3d as o3d  # noqa: F811
        except ImportError:
            raise RuntimeError(
                f"Kernel point file not found: {kernel_file}.  "
                f"Install open3d to auto-generate it, or place a pre-computed .ply file."
            )

        # Lazy import: matplotlib is only needed for kernel generation
        from ._kernel_gen import spherical_Lloyd

        kernel_points = spherical_Lloyd(1.0, num_kpoints, dimension=dimension, fixed=fixed, verbose=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(kernel_points)
        o3d.io.write_point_cloud(kernel_file, pcd)
    else:
        try:
            import open3d as o3d  # noqa: F811
        except ImportError:
            # Pure-numpy .ply reader fallback for the bundled file
            kernel_points = _read_ply_ascii(kernel_file)
        else:
            pcd = o3d.io.read_point_cloud(kernel_file)
            kernel_points = np.asarray(pcd.points, dtype=np.float32)

    # Random rotation (OPFA code always applies a random z-rotation + noise)
    theta = np.random.rand() * 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape).astype(np.float32)
    kernel_points = radius * kernel_points
    kernel_points = np.matmul(kernel_points.astype(np.float32), R)

    return kernel_points.astype(np.float32)


def _read_ply_ascii(filepath: str) -> np.ndarray:
    """Minimal ASCII ``.ply`` reader — no open3d dependency."""
    with open(filepath) as f:
        lines = f.readlines()

    # Find header end
    i = 0
    n_vertices = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex "):
            n_vertices = int(line.split()[-1])
        if line.strip() == "end_header":
            i += 1
            break

    points = []
    for line in lines[i : i + n_vertices]:
        parts = line.strip().split()
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.array(points, dtype=np.float32)


# =========================================================================
# GroupNorm for (N, C) tensors (port of kpconv/modules.py:GroupNorm)
# =========================================================================

class GroupNorm(nn.Module):
    """GroupNorm that operates on ``(N, C)`` tensors (no batch dimension).

    Internally transposes to ``(1, C, N)``, applies ``nn.GroupNorm``, then
    transposes back.
    """

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) → (1, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (1, C, N) → (N, C)
        return x.squeeze()


# =========================================================================
# Max pooling (port of kpconv/functional.py:maxpool)
# =========================================================================

def maxpool(x: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
    """Max-pool from neighbour indices (used by strided ResidualBlock).

    Args:
        x: ``(n1, d)`` features.
        neighbor_indices: ``(n2, max_num)`` pooling indices.

    Returns:
        ``(n2, d)`` pooled features.
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    return neighbor_feats.max(1)[0]


# =========================================================================
# UnaryBlock (port of kpconv/modules.py)
# =========================================================================

class UnaryBlock(nn.Module):
    """Linear → GroupNorm/LayerNorm → LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_norm: int,
        has_relu: bool = True,
        bias: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1) if has_relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.norm(x)
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)
        return x


# =========================================================================
# KPConv (port of kpconv/kpconv.py — identical)
# =========================================================================

class KPConv(nn.Module):
    """Kernel Point Convolution — linear correlation kernel.

    ``h(y_i, x_k) = max(0, 1 - ||y_i - x_k|| / sigma)``

    Reference: Thomas et al., ICCV 2019.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        bias: bool = False,
        dimension: int = 3,
        inf: float = 1e6,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.inf = inf

        self.weights = nn.Parameter(torch.zeros(self.kernel_size, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed="center")
        self.register_buffer("kernel_points", torch.from_numpy(kernel_points).float())

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        s_feats: torch.Tensor,  # (N, C_in)
        q_points: torch.Tensor,  # (M, 3)
        s_points: torch.Tensor,  # (N, 3)
        neighbor_indices: torch.Tensor,  # (M, H) LongTensor
    ) -> torch.Tensor:
        """KPConv forward — identical to OPFA.

        Returns:
            ``(M, C_out)``
        """
        # Pad with a sentinel point at infinity
        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N+1, 3)
        neighbors = index_select(s_points, neighbor_indices, dim=0)  # (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3) — centre on queries

        # Kernel point influences: h(y, x_k) = max(0, 1 - d / sigma)
        neighbors = neighbors.unsqueeze(2)  # (M, H, 1, 3)
        differences = neighbors - self.kernel_points  # (M, H, K, 3)
        sq_distances = torch.sum(differences**2, dim=3)  # (M, H, K)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = neighbor_weights.transpose(1, 2)  # (M, K, H)

        # Apply neighbour weights to features
        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N+1, C_in)
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)  # (M, H, C_in)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, C_in)

        # Convolution
        weighted_feats = weighted_feats.permute(1, 0, 2)  # (K, M, C_in)
        kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C_out)
        output_feats = torch.sum(kernel_outputs, dim=0, keepdim=False)  # (M, C_out)

        # Per-point normalisation (divide by number of active neighbours)
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats


# =========================================================================
# ConvBlock (port of kpconv/modules.py:ConvBlock)
# =========================================================================

class ConvBlock(nn.Module):
    """KPConv → GroupNorm/LayerNorm → LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        group_norm: int,
        negative_slope: float = 0.1,
        bias: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(
        self,
        s_feats: torch.Tensor,
        q_points: torch.Tensor,
        s_points: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


# =========================================================================
# ResidualBlock (port of kpconv/modules.py:ResidualBlock)
# =========================================================================

class ResidualBlock(nn.Module):
    """Bottleneck residual block with optional strided max-pool shortcut.

    OPFA bottleneck: in → mid(out//4) → KPConv(mid,mid) → out.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        group_norm: int,
        strided: bool = False,
        bias: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias, layer_norm=layer_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm_conv = nn.LayerNorm(mid_channels)
        else:
            self.norm_conv = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(
            mid_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
        )

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(
                in_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(
        self,
        s_feats: torch.Tensor,
        q_points: torch.Tensor,
        s_points: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.unary1(s_feats)
        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.norm_conv(x)
        x = self.leaky_relu(x)
        x = self.unary2(x)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)
        return x


# =========================================================================
# KPConvFPN (port of experiments/realworld/backbone.py — identical)
# =========================================================================

class KPConvFPN(nn.Module):
    """4-stage encoder-only feature pyramid network.

    Channel progression: init_dim → 2×init_dim → 4×init_dim → 8×init_dim → 16×init_dim.
    With ``init_dim=64``: 64 → 128 → 256 → 512 → 1024.

    Each stage: 1 strided ResidualBlock (down-sample) + 1-2 regular ResidualBlocks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_dim: int,
        kernel_size: int,
        init_radius: float,
        init_sigma: float,
        group_norm: int,
    ):
        super().__init__()
        # Note: output_dim is accepted but not used internally —
        # the output is always init_dim * 16 (faithful to OPFA).

        # Stage 1 (no stride)
        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        # Stage 2 (stride → ×2 radius)
        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        # Stage 3 (stride → ×4 radius)
        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        # Stage 4 (stride → ×8 radius)
        self.encoder4_1 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

    def forward(self, feats: torch.Tensor, data_dict: dict) -> torch.Tensor:
        """KPConvFPN forward — identical to OPFA.

        Args:
            feats: ``(N, input_dim)`` per-point initial features (PE embeddings).
            data_dict: dict with keys:
                - ``points``: list of ``(N_i, 3)`` point clouds per stage.
                - ``neighbors``: list of ``(N_i, K)`` neighbour index tensors.
                - ``subsampling``: list of ``(N_{i+1}, K')`` subsampling index tensors.

        Returns:
            ``(N_4, init_dim * 16)`` superpoint features at the coarsest scale.
        """
        points_list = data_dict["points"]
        neighbors_list = data_dict["neighbors"]
        subsampling_list = data_dict["subsampling"]

        # Stage 1
        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        # Stage 2 (first block is strided → uses subsampling[0])
        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        # Stage 3
        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        # Stage 4
        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        return feats_s4  # (N_4, init_dim * 16)
