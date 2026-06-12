import torch
import torch.nn as nn
from typing import Dict

from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    farthest_point_sample,
    index_points,
    query_ball_point,
    resolve_stage_values,
)

_MIN_CHANNELS = 32
"""Minimum channels in ResidualBlock hidden layer."""
_MIN_HIDDEN = 16
"""Minimum hidden dimension in LocalGeometryAggregation."""


class PointGroupNorm(nn.Module):
    def __init__(self, num_channels: int, max_groups: int = 8):
        super().__init__()
        for groups in (max_groups, 4, 2, 1):
            if num_channels % groups == 0 and num_channels // groups >= 8:
                break
        self.norm = nn.GroupNorm(groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class RawPointEmbedding(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, max_groups: int = 8):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        self.norm = PointGroupNorm(output_channels, max_groups=max_groups)
        self.act = nn.GELU()

    def forward(self, point_feature: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(point_feature)))


class DropPath(nn.Module):
    """Stochastic depth (DINOv2 style)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor_()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5,
                 max_groups: int = 8, drop_path: float = 0.0):
        super().__init__()
        hidden = max(int(channels * expansion), _MIN_CHANNELS)

        groups1 = self._best_groups(hidden, max_groups)
        self.net1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm1 = PointGroupNorm(hidden, max_groups=groups1)

        groups2 = self._best_groups(channels, max_groups)
        self.net2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm2 = PointGroupNorm(channels, max_groups=groups2)

        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)

        # Zero-init the last conv so the residual branch starts as identity.
        # Equivalent to LayerScale with init=0 (DINOv2 pattern), but uses no
        # extra nn.Parameter — compatible with existing optimizer grouping.
        nn.init.zeros_(self.net2.weight)

    @staticmethod
    def _best_groups(ch: int, max_groups: int = 8) -> int:
        for g in (max_groups, 4, 2, 1):
            if ch % g == 0 and ch // g >= 8:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.net1(x)))
        x = self.norm2(self.net2(x))
        x = self.drop_path(x)
        return self.act(x + residual)


class LearnedPosE(nn.Module):
    """Learned relative-position encoding (replaces sin/cos PE).

    Memory:  O(B * G * K * 4)  instead of  O(B * 3 * G * K * F).
    """
    def __init__(self, out_dim: int, hidden_ratio: float = 0.25):
        super().__init__()
        hidden = max(int(out_dim * hidden_ratio), _MIN_HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),           # x, y, z, distance
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, knn_xyz: torch.Tensor) -> torch.Tensor:
        """knn_xyz: (B, 3, G, K)  – relative coordinates (already normalised)."""
        B, _, G, K = knn_xyz.shape
        dist = torch.linalg.norm(knn_xyz, dim=1, keepdim=True)   # (B, 1, G, K)
        inp = torch.cat([knn_xyz, dist], dim=1)                   # (B, 4, G, K)
        inp = inp.permute(0, 2, 3, 1).reshape(B * G * K, 4)      # (B*G*K, 4)
        out = self.mlp(inp)                                       # (B*G*K, out_dim)
        return out.reshape(B, G, K, -1).permute(0, 3, 1, 2)      # (B, out_dim, G, K)


class Linear1Layer(nn.Module):
    """Conv1d that maps  concat(neighbor, center)  →  out_dim  channels."""
    def __init__(self, in_channels: int, out_channels: int, max_groups: int = 8):
        super().__init__()
        groups = ResidualBlock._best_groups(out_channels, max_groups)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            PointGroupNorm(out_channels, max_groups=groups),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FPSBallQuery(nn.Module):
    def __init__(self, num_centers: int, radius: float, max_neighbors: int,
                 fps_random_config: dict | None = None):
        super().__init__()
        self.num_centers = num_centers
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.fps_random_config = fps_random_config or {}

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        _, fps_idx = farthest_point_sample(xyz, self.num_centers,
                                           **self.fps_random_config)
        center_xyz = index_points(xyz, fps_idx)
        center_feature = index_points(point_feature, fps_idx)

        neighbor_idx = query_ball_point(self.radius, self.max_neighbors, xyz, center_xyz)
        neighbor_xyz = index_points(xyz, neighbor_idx)
        neighbor_feature = index_points(point_feature, neighbor_idx)
        return center_xyz, center_feature, neighbor_xyz, neighbor_feature


class LocalGeometryAggregation(nn.Module):
    def __init__(self, output_channels: int, radius: float, block_num: int,
                 point_cloud_type: str, residual_expansion: float = 0.5,
                 max_groups: int = 8, drop_path: float = 0.0,
                 input_channels: int = None):
        super().__init__()
        self.point_cloud_type = point_cloud_type
        self.pos_enc = LearnedPosE(output_channels)
        in_ch = input_channels if input_channels is not None else output_channels
        self.linear1 = Linear1Layer(in_ch * 2, output_channels, max_groups=max_groups)
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(output_channels, expansion=residual_expansion,
                          max_groups=max_groups, drop_path=drop_path)
            for _ in range(block_num)
        ])

    def normalize_relative_xyz(self, center_xyz: torch.Tensor,
                               neighbor_xyz: torch.Tensor) -> torch.Tensor:
        relative_xyz = neighbor_xyz - center_xyz.unsqueeze(2)

        if self.point_cloud_type == "scan":
            scale = relative_xyz.abs().amax(dim=2, keepdim=True).clamp_min(1e-5)
            return relative_xyz / scale

        if self.point_cloud_type == "mn40":
            scale = torch.std(relative_xyz).clamp_min(1e-5)
            return relative_xyz / scale

        return relative_xyz

    def forward(self, center_xyz: torch.Tensor, center_feature: torch.Tensor,
                neighbor_xyz: torch.Tensor, neighbor_feature: torch.Tensor) -> torch.Tensor:
        # ---- normalise relative coordinates ----
        relative_xyz = self.normalize_relative_xyz(center_xyz, neighbor_xyz)

        # ---- feature expansion ----
        # index_points returns channel-last:  centre (B,G,C)  neighbour (B,G,K,C)
        B, G, C = center_feature.shape
        K = neighbor_feature.shape[2]
        center_expanded = center_feature.unsqueeze(2).expand(-1, -1, K, -1)
        group_input = torch.cat([neighbor_feature, center_expanded], dim=-1)
        group_input = group_input.permute(0, 3, 1, 2).contiguous()  # (B, 2*C, G, K)

        # ---- Linear1Layer: channel projection ----
        group_feat = self.linear1(group_input.reshape(B, -1, G * K)).reshape(B, -1, G, K)

        # ---- Learned position encoding ----
        relative_xyz_perm = relative_xyz.permute(0, 3, 1, 2).contiguous()  # (B, 3, G, K)
        pe = self.pos_enc(relative_xyz_perm)  # (B, out_dim, G, K)

        # ---- Weigh gating (Point-PN: add + multiply, fused to avoid in-place) ----
        group_feat = (group_feat + pe) * pe   # (B, out_dim, G, K)
        del pe

        # ---- residual blocks ----
        return self.residual_blocks(group_feat)


class Pooling(nn.Module):
    def __init__(self, out_dim: int, max_groups: int = 8):
        super().__init__()
        groups = ResidualBlock._best_groups(out_dim, max_groups)
        self.norm = PointGroupNorm(out_dim, max_groups=groups)
        self.act = nn.GELU()

    def forward(self, knn_x_w: torch.Tensor) -> torch.Tensor:
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return self.act(self.norm(lc_x))


class ParametricEncoder(nn.Module):
    def __init__(self, input_channels: int, input_points: int, num_stages: int,
                 embed_channels: int, stage_radii,
                 stage_num_neighbors, stage_lga_blocks,
                 stage_channel_expansion, point_cloud_type: str,
                 residual_expansion: float = 0.5, max_groups: int = 8,
                 drop_path: float = 0.0,
                 fps_random_config: dict | None = None):
        super().__init__()
        self.fps_random_config = fps_random_config or {}
        self.raw_point_embedding = RawPointEmbedding(input_channels, embed_channels,
                                                     max_groups=max_groups)
        self.fps_ballquery_stages = nn.ModuleList()
        self.lga_stages = nn.ModuleList()
        self.pooling_stages = nn.ModuleList()

        stage_radii = resolve_stage_values(stage_radii, num_stages, "stage_radii")
        stage_num_neighbors = resolve_stage_values(stage_num_neighbors, num_stages,
                                                    "stage_num_neighbors")
        stage_lga_blocks = resolve_stage_values(stage_lga_blocks, num_stages,
                                                 "stage_lga_blocks")
        stage_channel_expansion = resolve_stage_values(stage_channel_expansion,
                                                       num_stages,
                                                       "stage_channel_expansion")

        current_channels = embed_channels
        num_centers = input_points
        prev_channels = embed_channels  # input channels for stage 0
        for stage_index in range(num_stages):
            current_channels = current_channels * stage_channel_expansion[stage_index]
            num_centers = max(num_centers // 2, 1)

            self.fps_ballquery_stages.append(
                FPSBallQuery(num_centers=num_centers,
                             radius=stage_radii[stage_index],
                             max_neighbors=stage_num_neighbors[stage_index],
                             fps_random_config=self.fps_random_config))
            self.lga_stages.append(
                LocalGeometryAggregation(
                    output_channels=current_channels,
                    radius=stage_radii[stage_index],
                    block_num=stage_lga_blocks[stage_index],
                    point_cloud_type=point_cloud_type,
                    residual_expansion=residual_expansion,
                    max_groups=max_groups,
                    drop_path=drop_path,
                    input_channels=prev_channels))
            self.pooling_stages.append(
                Pooling(current_channels, max_groups=max_groups))
            prev_channels = current_channels

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        point_feature = self.raw_point_embedding(point_feature)
        intermediate_outputs: Dict[str, torch.Tensor] = {"raw_point_feature": point_feature}

        for stage_index, (fps_bq, lga, pool) in enumerate(
                zip(self.fps_ballquery_stages, self.lga_stages, self.pooling_stages)):
            center_xyz, center_feature, neighbor_xyz, neighbor_feature = fps_bq(
                xyz, point_feature.transpose(1, 2).contiguous())
            group_feature = lga(center_xyz, center_feature, neighbor_xyz, neighbor_feature)
            xyz = center_xyz
            point_feature = pool(group_feature)

            intermediate_outputs[f"stage_{stage_index}_xyz"] = xyz
            intermediate_outputs[f"stage_{stage_index}_point_feature"] = point_feature

        intermediate_outputs["global_point_feature"] = point_feature.max(dim=-1).values
        return xyz, point_feature, intermediate_outputs


class PointPNTokenizer(nn.Module):
    supports_global_token = True
    supports_intermediate_outputs = True
    requires_fixed_num_points = False

    def __init__(self, input_channels: int = 6, input_points: int = 1024,
                 num_stages: int = 3, embed_channels: int = 64,
                 stage_radii=(0.04, 0.08, 0.16),
                 stage_num_neighbors=(24, 24, 16),
                 stage_lga_blocks=(1, 1, 1),
                 stage_channel_expansion=(2, 2, 2),
                 point_cloud_type: str = "scan",
                 residual_expansion: float = 0.5,
                 max_groups: int = 8,
                 drop_path: float = 0.0,
                 fps_random_config: dict | None = None):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        stage_channel_expansion = resolve_stage_values(stage_channel_expansion,
                                                       num_stages,
                                                       "stage_channel_expansion")
        if any(expansion != 2 for expansion in stage_channel_expansion):
            raise ValueError(
                "PointPNTokenizer currently requires stage_channel_expansion == 2 "
                "at every stage because LocalGeometryAggregation concatenates neighbour "
                "features with centre features, doubling channels."
            )

        self.input_channels = input_channels
        self.input_points = input_points
        self.num_stages = num_stages
        self.fps_random_config = fps_random_config or {}
        self.num_patches = self._compute_num_patches(input_points, num_stages)
        self.output_channels = embed_channels
        for expansion in stage_channel_expansion:
            self.output_channels *= expansion

        self.encoder = ParametricEncoder(
            input_channels=input_channels,
            input_points=input_points,
            num_stages=num_stages,
            embed_channels=embed_channels,
            stage_radii=stage_radii,
            stage_num_neighbors=stage_num_neighbors,
            stage_lga_blocks=stage_lga_blocks,
            stage_channel_expansion=stage_channel_expansion,
            point_cloud_type=point_cloud_type,
            residual_expansion=residual_expansion,
            max_groups=max_groups,
            drop_path=drop_path,
            fps_random_config=self.fps_random_config)

    def forward(self, pointcloud: torch.Tensor,
                return_global_token: bool = False,
                return_intermediate: bool = False):
        num_pts = pointcloud.size(1)
        if num_pts > self.input_points:
            pointcloud, _ = farthest_point_sample(pointcloud, self.input_points,
                                                  **self.fps_random_config)
        elif num_pts < self.input_points:
            pad_n = self.input_points - num_pts
            pointcloud = torch.cat([
                pointcloud,
                pointcloud[:, -1:].expand(-1, pad_n, -1)], dim=1)

        feat_dtype = pointcloud.dtype
        xyz = pointcloud[..., :3].float()                                   # always fp32 for pytorch3d
        point_feature = pointcloud[..., :self.input_channels].transpose(1, 2).contiguous()
        final_xyz, final_point_feature, intermediate_outputs = self.encoder(xyz, point_feature)
        if final_point_feature.dtype != feat_dtype:
            final_point_feature = final_point_feature.to(feat_dtype)

        patch_token = final_point_feature.permute(0, 2, 1).contiguous()
        patch_center = final_xyz
        outputs = [patch_token, patch_center]

        if return_global_token:
            outputs.append(self.get_global_token(patch_token))
        if return_intermediate:
            outputs.append(intermediate_outputs)
        return tuple(outputs)

    def get_global_token(self, patch_token: torch.Tensor,
                         patch_center: torch.Tensor = None) -> torch.Tensor:
        return patch_token.max(dim=1, keepdim=True).values

    @property
    def out_dim(self) -> int:
        return self.output_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.output_channels)

    @staticmethod
    def _compute_num_patches(input_points: int, num_stages: int) -> int:
        num_patches = input_points
        for _ in range(num_stages):
            num_patches = max(num_patches // 2, 1)
        return num_patches


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    pointpn = PointPNTokenizer(
        input_channels=6,
        input_points=num_points,
        num_stages=3,
        embed_channels=64,
        stage_radii=(0.04, 0.08, 0.16),
        stage_num_neighbors=(24, 24, 16),
        stage_lga_blocks=(1, 1, 1),
        stage_channel_expansion=(2, 2, 2),
        point_cloud_type="scan",
    )

    patch_token, patch_center, global_token, intermediate_outputs = pointpn(
        pointcloud, return_global_token=True, return_intermediate=True)

    print("=== PointPNTokenizer Example (optimized) ===")
    print("input_pointcloud:", tuple(pointcloud.shape))
    for name, value in intermediate_outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("patch_token:", tuple(patch_token.shape))
    print("patch_center:", tuple(patch_center.shape))
    print("global_token:", tuple(global_token.shape))
    print("out_dim:", pointpn.out_dim)
    print("out_shape:", pointpn.out_shape)
    print("=== PASSED ===")


if __name__ == "__main__":
    example()
