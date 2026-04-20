import torch
import torch.nn as nn
from typing import Dict

from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    farthest_point_sample,
    index_points,
    knn_point,
    resolve_stage_values,
)


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


class ResidualPointMLP2d(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5, max_groups: int = 8):
        super().__init__()
        hidden_channels = max(int(channels * expansion), 32)
        self.net1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = PointGroupNorm(hidden_channels, max_groups=max_groups)
        self.net2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.norm2 = PointGroupNorm(channels, max_groups=max_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_feature = x
        x = self.act(self.norm1(self.net1(x)))
        x = self.norm2(self.net2(x))
        return self.act(x + residual_feature)


class PositionalEncodingGeometry(nn.Module):
    def __init__(self, output_channels: int, alpha: float, beta: float):
        super().__init__()
        self.output_channels = output_channels
        self.beta = beta
        self.input_channels = 3

        feat_channels = max((self.output_channels + self.input_channels * 2 - 1) // (self.input_channels * 2), 1)
        feat_range = torch.arange(feat_channels, dtype=torch.float32)
        dim_embed = torch.pow(torch.tensor(alpha, dtype=torch.float32), feat_range / feat_channels)
        self.register_buffer("dim_embed", dim_embed, persistent=False)

    def forward(self, relative_xyz: torch.Tensor, point_feature: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_centers, num_neighbors = relative_xyz.shape
        dim_embed = self.dim_embed.to(device=relative_xyz.device, dtype=relative_xyz.dtype)
        div_embed = self.beta * relative_xyz.unsqueeze(-1) / dim_embed
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        relative_pos_feature = torch.cat([sin_embed, cos_embed], dim=-1)
        relative_pos_feature = relative_pos_feature.permute(0, 1, 4, 2, 3).contiguous()
        relative_pos_feature = relative_pos_feature.view(batch_size, -1, num_centers, num_neighbors)
        relative_pos_feature = relative_pos_feature[:, : self.output_channels]
        return point_feature + relative_pos_feature


class FPSkNN(nn.Module):
    def __init__(self, num_centers: int, num_neighbors: int):
        super().__init__()
        self.num_centers = num_centers
        self.num_neighbors = num_neighbors

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        _, fps_idx = farthest_point_sample(xyz, self.num_centers)
        center_xyz = index_points(xyz, fps_idx)
        center_feature = index_points(point_feature, fps_idx)

        neighbor_idx = knn_point(self.num_neighbors, xyz, center_xyz)
        neighbor_xyz = index_points(xyz, neighbor_idx)
        neighbor_feature = index_points(point_feature, neighbor_idx)
        return center_xyz, center_feature, neighbor_xyz, neighbor_feature


class LocalGeometryAggregation(nn.Module):
    def __init__(
        self,
        output_channels: int,
        alpha: float,
        beta: float,
        block_num: int,
        point_cloud_type: str,
        residual_expansion: float = 0.5,
        max_groups: int = 8,
    ):
        super().__init__()
        self.point_cloud_type = point_cloud_type
        self.positional_encoding = PositionalEncodingGeometry(output_channels, alpha, beta)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualPointMLP2d(
                    output_channels,
                    expansion=residual_expansion,
                    max_groups=max_groups,
                )
                for _ in range(block_num)
            ]
        )

    def normalize_relative_xyz(self, center_xyz: torch.Tensor, neighbor_xyz: torch.Tensor) -> torch.Tensor:
        relative_xyz = neighbor_xyz - center_xyz.unsqueeze(2)

        if self.point_cloud_type == "scan":
            scale = relative_xyz.abs().amax(dim=2, keepdim=True).clamp_min(1e-5)
            return relative_xyz / scale

        if self.point_cloud_type == "mn40":
            scale = torch.std(relative_xyz).clamp_min(1e-5)
            return relative_xyz / scale

        return relative_xyz

    def forward(
        self,
        center_xyz: torch.Tensor,
        center_feature: torch.Tensor,
        neighbor_xyz: torch.Tensor,
        neighbor_feature: torch.Tensor,
    ) -> torch.Tensor:
        relative_xyz = self.normalize_relative_xyz(center_xyz, neighbor_xyz)
        center_feature_expanded = center_feature.unsqueeze(2).expand(-1, -1, neighbor_feature.size(2), -1)
        group_input = torch.cat([neighbor_feature, center_feature_expanded], dim=-1)

        relative_xyz = relative_xyz.permute(0, 3, 1, 2).contiguous()
        group_input = group_input.permute(0, 3, 1, 2).contiguous()

        group_input = self.positional_encoding(relative_xyz, group_input)
        return self.residual_blocks(group_input)


class ParametricEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_points: int,
        num_stages: int,
        embed_channels: int,
        stage_num_neighbors,
        alpha: float,
        beta: float,
        stage_lga_blocks,
        stage_channel_expansion,
        point_cloud_type: str,
        residual_expansion: float = 0.5,
        max_groups: int = 8,
    ):
        super().__init__()
        self.raw_point_embedding = RawPointEmbedding(input_channels, embed_channels, max_groups=max_groups)
        self.fps_knn_stages = nn.ModuleList()
        self.local_geometry_aggregation_stages = nn.ModuleList()

        stage_num_neighbors = resolve_stage_values(stage_num_neighbors, num_stages, "stage_num_neighbors")
        stage_lga_blocks = resolve_stage_values(stage_lga_blocks, num_stages, "stage_lga_blocks")
        stage_channel_expansion = resolve_stage_values(
            stage_channel_expansion,
            num_stages,
            "stage_channel_expansion",
        )

        current_channels = embed_channels
        num_centers = input_points
        for stage_index in range(num_stages):
            current_channels = current_channels * stage_channel_expansion[stage_index]
            num_centers = max(num_centers // 2, 1)

            self.fps_knn_stages.append(
                FPSkNN(
                    num_centers=num_centers,
                    num_neighbors=stage_num_neighbors[stage_index],
                )
            )
            self.local_geometry_aggregation_stages.append(
                LocalGeometryAggregation(
                    output_channels=current_channels,
                    alpha=alpha,
                    beta=beta,
                    block_num=stage_lga_blocks[stage_index],
                    point_cloud_type=point_cloud_type,
                    residual_expansion=residual_expansion,
                    max_groups=max_groups,
                )
            )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        point_feature = self.raw_point_embedding(point_feature)
        intermediate_outputs: Dict[str, torch.Tensor] = {"raw_point_feature": point_feature}

        for stage_index, (fps_knn, local_geometry_aggregation) in enumerate(
            zip(self.fps_knn_stages, self.local_geometry_aggregation_stages)
        ):
            center_xyz, center_feature, neighbor_xyz, neighbor_feature = fps_knn(
                xyz,
                point_feature.transpose(1, 2).contiguous(),
            )
            group_feature = local_geometry_aggregation(center_xyz, center_feature, neighbor_xyz, neighbor_feature)
            xyz = center_xyz
            point_feature = group_feature.max(dim=-1).values

            intermediate_outputs[f"stage_{stage_index}_xyz"] = xyz
            intermediate_outputs[f"stage_{stage_index}_point_feature"] = point_feature

        intermediate_outputs["global_point_feature"] = point_feature.max(dim=-1).values
        return xyz, point_feature, intermediate_outputs


class PointPNTokenizer(nn.Module):
    supports_global_token = True
    supports_intermediate_outputs = True
    requires_fixed_num_points = True

    def __init__(
        self,
        input_channels: int = 6,
        input_points: int = 1024,
        num_stages: int = 3,
        embed_channels: int = 64,
        beta: float = 16.0,
        alpha: float = 100.0,
        stage_lga_blocks=(2, 2, 1),
        stage_channel_expansion=(2, 2, 2),
        point_cloud_type: str = "scan",
        stage_num_neighbors=(24, 24, 16),
        residual_expansion: float = 0.5,
        max_groups: int = 8,
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        stage_channel_expansion = resolve_stage_values(
            stage_channel_expansion,
            num_stages,
            "stage_channel_expansion",
        )
        if any(expansion != 2 for expansion in stage_channel_expansion):
            raise ValueError(
                "PointPNTokenizer currently requires stage_channel_expansion == 2 at every stage because "
                "LocalGeometryAggregation concatenates neighbor features with center features, doubling channels. "
                f"Got stage_channel_expansion={stage_channel_expansion}."
            )

        self.input_channels = input_channels
        self.input_points = input_points
        self.num_stages = num_stages
        self.num_patches = self.compute_num_patches(input_points, num_stages)
        self.output_channels = embed_channels
        for expansion in stage_channel_expansion:
            self.output_channels *= expansion

        self.encoder = ParametricEncoder(
            input_channels=input_channels,
            input_points=input_points,
            num_stages=num_stages,
            embed_channels=embed_channels,
            stage_num_neighbors=stage_num_neighbors,
            alpha=alpha,
            beta=beta,
            stage_lga_blocks=stage_lga_blocks,
            stage_channel_expansion=stage_channel_expansion,
            point_cloud_type=point_cloud_type,
            residual_expansion=residual_expansion,
            max_groups=max_groups,
        )

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_global_token: bool = False,
        return_intermediate: bool = False,
    ):
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(1) != self.input_points:
            raise ValueError(
                f"pointcloud has {pointcloud.size(1)} points, but input_points={self.input_points}"
            )
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        xyz = pointcloud[..., :3]
        point_feature = pointcloud[..., : self.input_channels].transpose(1, 2).contiguous()
        final_xyz, final_point_feature, intermediate_outputs = self.encoder(xyz, point_feature)

        patch_token = final_point_feature.permute(0, 2, 1).contiguous()
        patch_center = final_xyz
        outputs = [patch_token, patch_center]

        if return_global_token:
            outputs.append(self.get_global_token(patch_token))
        if return_intermediate:
            outputs.append(intermediate_outputs)
        return tuple(outputs)

    def get_global_token(self, patch_token: torch.Tensor) -> torch.Tensor:
        return patch_token.max(dim=1, keepdim=True).values

    @property
    def out_dim(self) -> int:
        return self.output_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.output_channels)

    @staticmethod
    def compute_num_patches(input_points: int, num_stages: int) -> int:
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
        beta=16.0,
        alpha=100.0,
        stage_lga_blocks=(2, 2, 1),
        stage_channel_expansion=(2, 2, 2),
        point_cloud_type="scan",
        stage_num_neighbors=(24, 24, 16),
    )

    patch_token, patch_center, global_token, intermediate_outputs = pointpn(
        pointcloud,
        return_global_token=True,
        return_intermediate=True,
    )

    print("=== PointPNTokenizer Example ===")
    print("input_pointcloud:", tuple(pointcloud.shape))
    for name, value in intermediate_outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("patch_token:", tuple(patch_token.shape))
    print("patch_center:", tuple(patch_center.shape))
    print("global_token:", tuple(global_token.shape))
    print("out_dim:", pointpn.out_dim)
    print("out_shape:", pointpn.out_shape)


if __name__ == "__main__":
    example()