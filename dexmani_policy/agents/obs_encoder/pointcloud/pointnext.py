import torch
import torch.nn as nn
from typing import Dict

from dexmani_policy.agents.obs_encoder.pointcloud.common.position_encodings import (
    RelativePositionalEncoding3D,
    SinusoidalPosEmb3D,
)
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    group,
    resolve_stage_values,
    sample_and_group,
    sample_and_group_all,
)


def normalize_relative_xyz(relative_xyz: torch.Tensor, radius: float, eps: float = 1e-6) -> torch.Tensor:
    return relative_xyz / max(radius, eps)


class PointMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)


class SetAbstraction(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        radius: float,
        num_neighbors: int,
        layers: int = 2,
        use_residual: bool = True,
        is_head: bool = False,
        global_aggr: bool = False,
        position_encoding_channels: int = 16,
    ):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.is_head = is_head
        self.global_aggr = global_aggr
        self.use_residual = use_residual and (not is_head) and (not global_aggr)
        self.relative_position_encoding = None if is_head else RelativePositionalEncoding3D(position_encoding_channels)

        if is_head:
            mlp_channels = [input_channels] + [output_channels] * layers
        else:
            group_input_channels = input_channels + 3 + position_encoding_channels
            hidden_channels = output_channels if stride == 1 else max(output_channels // 2, 1)
            mlp_channels = [group_input_channels] + [hidden_channels] * (layers - 1) + [output_channels]

        blocks = []
        for block_index in range(len(mlp_channels) - 1):
            use_activation_in_block = block_index < len(mlp_channels) - 2 or not self.use_residual
            blocks.append(
                PointMLP(
                    mlp_channels[block_index],
                    mlp_channels[block_index + 1],
                    use_activation=use_activation_in_block,
                )
            )
        self.point_mlp = nn.Sequential(*blocks)

        if self.use_residual:
            self.residual_projection = (
                nn.Identity()
                if input_channels == output_channels
                else PointMLP(input_channels, output_channels, use_activation=False)
            )
        else:
            self.residual_projection = None
        self.output_activation = nn.GELU()

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        if self.is_head:
            return xyz, self.point_mlp(point_feature)

        if self.global_aggr:
            center_xyz, neighbor_feature = sample_and_group_all(xyz, point_feature)
            center_feature = None
        else:
            sample_ratio = min(1.0 / self.stride, 1.0)
            center_xyz, neighbor_feature, center_feature = sample_and_group(
                sample_ratio,
                self.radius,
                self.num_neighbors,
                xyz,
                point_feature,
            )

        relative_xyz = neighbor_feature[..., :3]
        normalized_relative_xyz = normalize_relative_xyz(relative_xyz, self.radius)
        relative_pos_feature = self.relative_position_encoding(normalized_relative_xyz)
        group_input = torch.cat(
            (normalized_relative_xyz, neighbor_feature[..., 3:], relative_pos_feature),
            dim=-1,
        )
        center_feature_out = self.point_mlp(group_input).max(dim=2).values

        if self.use_residual and center_feature is not None:
            center_feature_out = self.output_activation(
                center_feature_out + self.residual_projection(center_feature)
            )

        return center_xyz, center_feature_out


class LocalAggregation(nn.Module):
    def __init__(
        self,
        input_channels: int,
        radius: float,
        num_neighbors: int,
        expansion: int = 2,
        position_encoding_channels: int = 16,
    ):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.relative_position_encoding = RelativePositionalEncoding3D(position_encoding_channels)
        hidden_channels = input_channels * expansion
        self.point_mlp = nn.Sequential(
            PointMLP(input_channels + 3 + position_encoding_channels, hidden_channels),
            PointMLP(hidden_channels, input_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor) -> torch.Tensor:
        neighbor_feature = group(self.radius, self.num_neighbors, xyz, point_feature)
        relative_xyz = neighbor_feature[..., :3]
        normalized_relative_xyz = normalize_relative_xyz(relative_xyz, self.radius)
        relative_pos_feature = self.relative_position_encoding(normalized_relative_xyz)
        group_input = torch.cat(
            (normalized_relative_xyz, neighbor_feature[..., 3:], relative_pos_feature),
            dim=-1,
        )
        return self.point_mlp(group_input).max(dim=2).values


class InvertedResidualPointBlock(nn.Module):
    def __init__(self, channels: int, radius: float, num_neighbors: int, expansion: int = 2):
        super().__init__()
        self.local_aggregation = LocalAggregation(channels, radius, num_neighbors, expansion=expansion)
        self.channel_mlp = nn.Sequential(
            PointMLP(channels, channels * expansion),
            PointMLP(channels * expansion, channels, use_activation=False),
        )
        self.activation = nn.GELU()

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        residual_feature = point_feature
        point_feature = self.local_aggregation(xyz, point_feature)
        point_feature = self.channel_mlp(point_feature)
        return xyz, self.activation(point_feature + residual_feature)


class PointNextEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        output_channels: int = 256,
        stage_depths: tuple[int, ...] = (1, 2, 2),
        stage_strides: tuple[int, ...] = (1, 2, 2),
        stage_channels: tuple[int, ...] = (64, 128, 256),
        radii: tuple[float, ...] = (0.04, 0.08, 0.16),
        num_neighbors: tuple[int, ...] = (24, 24, 32),
        sa_layers: int = 2,
        expansion: int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        num_stages = len(stage_depths)
        stage_strides = resolve_stage_values(stage_strides, num_stages, "stage_strides")
        stage_channels = resolve_stage_values(stage_channels, num_stages, "stage_channels")
        radii = resolve_stage_values(radii, num_stages, "radii")
        num_neighbors = resolve_stage_values(num_neighbors, num_stages, "num_neighbors")

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.stages = nn.ModuleList()
        current_channels = input_channels
        for stage_index, (depth, stride, channels, radius, neighbors) in enumerate(
            zip(stage_depths, stage_strides, stage_channels, radii, num_neighbors)
        ):
            blocks = [
                SetAbstraction(
                    input_channels=current_channels,
                    output_channels=channels,
                    stride=stride,
                    radius=radius,
                    num_neighbors=neighbors,
                    layers=1 if stage_index == 0 and stride == 1 else sa_layers,
                    use_residual=use_residual,
                    is_head=stage_index == 0 and stride == 1,
                )
            ]
            for _ in range(1, depth):
                blocks.append(InvertedResidualPointBlock(channels, radius, neighbors, expansion=expansion))
            self.stages.append(nn.ModuleList(blocks))
            current_channels = channels

        self.global_position_embedding = SinusoidalPosEmb3D(96)
        self.global_position_projection = nn.Sequential(
            nn.Linear(96, current_channels),
            nn.LayerNorm(current_channels),
            nn.GELU(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(current_channels, output_channels),
            nn.LayerNorm(output_channels),
        )

    def forward(self, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        xyz = pointcloud[..., :3]
        point_feature = pointcloud[..., : self.input_channels]

        for stage in self.stages:
            for block in stage:
                xyz, point_feature = block(xyz, point_feature)

        global_token = self._get_global_token(point_feature, xyz)
        return {"global_token": global_token}

    def _get_global_token(self, final_point_feature: torch.Tensor, final_xyz: torch.Tensor) -> torch.Tensor:
        pooled_point_feature = final_point_feature.max(dim=1).values
        pooled_xyz = final_xyz.mean(dim=1)
        global_token_feature = pooled_point_feature + self.global_position_projection(
            self.global_position_embedding(pooled_xyz)
        )
        return self.output_projection(global_token_feature)

    @property
    def out_dim(self) -> int:
        return self.output_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (1, self.output_channels)


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    pointnext = PointNextEncoder(
        input_channels=6,
        output_channels=256,
        stage_depths=(1, 2, 2),
        stage_strides=(1, 2, 2),
        stage_channels=(64, 128, 256),
        radii=(0.04, 0.08, 0.16),
        num_neighbors=(24, 24, 32),
    )

    with torch.no_grad():
        out = pointnext(pointcloud)

    print("=== PointNextEncoder Example ===")
    print("input:", tuple(pointcloud.shape))
    print("global_token:", tuple(out["global_token"].shape))
    print("out_dim:", pointnext.out_dim)
    print("out_shape:", pointnext.out_shape)


if __name__ == "__main__":
    example()