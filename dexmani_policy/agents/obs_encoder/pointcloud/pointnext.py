import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    group,
    sample_and_group,
    sample_and_group_all,
)
from dexmani_policy.agents.obs_encoder.pointcloud.common.position_encodings import (
    SinusoidalPosEmb3D,
    RelativePositionalEncoding3D,
)


def resolve_stage_values(value, num_stages: int, name: str):
    if len(value) != num_stages:
        raise ValueError(f"{name} must have length {num_stages}, but got {len(value)}")
    return tuple(value)


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
        in_channels: int,
        out_channels: int,
        stride: int,
        radius: float,
        num_neighbors: int,
        layers: int = 2,
        use_residual: bool = True,
        is_head: bool = False,
        global_aggr: bool = False,
        position_encoding_dim: int = 16,
    ):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.is_head = is_head
        self.global_aggr = global_aggr
        self.use_residual = use_residual and (not self.is_head) and (not self.global_aggr)
        self.relative_position_encoding = None if is_head else RelativePositionalEncoding3D(position_encoding_dim)

        if is_head:
            channels = [in_channels] + [out_channels] * layers
        else:
            grouped_in_channels = in_channels + 3 + position_encoding_dim
            hidden_channels = out_channels if stride == 1 else max(out_channels // 2, 1)
            channels = [grouped_in_channels] + [hidden_channels] * (layers - 1) + [out_channels]

        blocks = []
        for i in range(len(channels) - 1):
            use_activation_i = i < len(channels) - 2 or not self.use_residual
            blocks.append(PointMLP(channels[i], channels[i + 1], use_activation=use_activation_i))
        self.point_mlp = nn.Sequential(*blocks)

        if self.use_residual:
            self.residual_projection = (
                nn.Identity()
                if in_channels == out_channels
                else PointMLP(in_channels, out_channels, use_activation=False)
            )
        else:
            self.residual_projection = None
        self.output_activation = nn.GELU()

    def forward(self, position: torch.Tensor, feature: torch.Tensor):
        if self.is_head:
            feature = self.point_mlp(feature)
            return position, feature

        if self.global_aggr:
            new_position, grouped_feature = sample_and_group_all(position, feature)
            identity = None
        else:
            sample_ratio = min(1.0 / self.stride, 1.0)
            new_position, grouped_feature, identity = sample_and_group(
                sample_ratio,
                self.radius,
                self.num_neighbors,
                position,
                feature,
            )

        relative_xyz = grouped_feature[..., :3]
        relative_position = self.relative_position_encoding(relative_xyz)
        grouped_feature = torch.cat((grouped_feature, relative_position), dim=-1)
        new_feature = self.point_mlp(grouped_feature).max(dim=2).values

        if self.use_residual and identity is not None:
            new_feature = self.output_activation(new_feature + self.residual_projection(identity))

        return new_position, new_feature


class LocalAggregation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        radius: float,
        num_neighbors: int,
        expansion: int = 2,
        position_encoding_dim: int = 16,
    ):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.relative_position_encoding = RelativePositionalEncoding3D(position_encoding_dim)
        hidden_channels = in_channels * expansion
        self.point_mlp = nn.Sequential(
            PointMLP(in_channels + 3 + position_encoding_dim, hidden_channels),
            PointMLP(hidden_channels, in_channels, use_activation=False),
        )

    def forward(self, position: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        grouped_feature = group(self.radius, self.num_neighbors, position, feature)
        relative_xyz = grouped_feature[..., :3]
        relative_position = self.relative_position_encoding(relative_xyz)
        grouped_feature = torch.cat((grouped_feature, relative_position), dim=-1)
        return self.point_mlp(grouped_feature).max(dim=2).values


class InvertedResidualPointBlock(nn.Module):
    def __init__(self, channels: int, radius: float, num_neighbors: int, expansion: int = 2):
        super().__init__()
        self.local_aggregation = LocalAggregation(channels, radius, num_neighbors, expansion=expansion)
        self.channel_mlp = nn.Sequential(
            PointMLP(channels, channels * expansion),
            PointMLP(channels * expansion, channels, use_activation=False),
        )
        self.activation = nn.GELU()

    def forward(self, position: torch.Tensor, feature: torch.Tensor):
        identity = feature
        feature = self.local_aggregation(position, feature)
        feature = self.channel_mlp(feature)
        feature = self.activation(feature + identity)
        return position, feature


class PointNextEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        output_channels: int = 256,
        point_wise: bool = False,
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
        self.point_wise = point_wise

        self.stages = nn.ModuleList()
        current_channels = input_channels
        for stage_index, (depth, stride, channels, radius, neighbors) in enumerate(
            zip(stage_depths, stage_strides, stage_channels, radii, num_neighbors)
        ):
            blocks = [
                SetAbstraction(
                    in_channels=current_channels,
                    out_channels=channels,
                    stride=stride,
                    radius=radius,
                    num_neighbors=neighbors,
                    layers=1 if stage_index == 0 and stride == 1 else sa_layers,
                    use_residual=use_residual,
                    is_head=stage_index == 0 and stride == 1,
                    global_aggr=False,
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

    def forward(self, pointcloud: torch.Tensor, return_intermediate: bool = False):
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        position = pointcloud[..., :3]
        feature = pointcloud[..., : self.input_channels]

        traces = {
            "input_xyz": position,
            "input_feature": feature,
            "stage_positions": [],
            "stage_features": [],
        }

        for stage in self.stages:
            for block in stage:
                position, feature = block(position, feature)
            traces["stage_positions"].append(position)
            traces["stage_features"].append(feature)

        traces["final_position"] = position

        if self.point_wise:
            point_feature = self.output_projection(feature)
            traces["point_feature"] = point_feature
            if return_intermediate:
                return point_feature, traces
            return point_feature

        pooled_feature = feature.max(dim=1).values
        pooled_position = position.mean(dim=1)
        global_feature = pooled_feature + self.global_position_projection(
            self.global_position_embedding(pooled_position)
        )
        global_feature = self.output_projection(global_feature)
        traces["global_feature"] = global_feature

        if return_intermediate:
            return global_feature, traces
        return global_feature


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    global_model = PointNextEncoder(
        input_channels=6,
        output_channels=256,
        point_wise=False,
        stage_depths=(1, 2, 2),
        stage_strides=(1, 2, 2),
        stage_channels=(64, 128, 256),
        radii=(0.04, 0.08, 0.16),
        num_neighbors=(24, 24, 32),
    )
    point_model = PointNextEncoder(
        input_channels=6,
        output_channels=256,
        point_wise=True,
        stage_depths=(1, 2, 2),
        stage_strides=(1, 2, 2),
        stage_channels=(64, 128, 256),
        radii=(0.04, 0.08, 0.16),
        num_neighbors=(24, 24, 32),
    )

    with torch.no_grad():
        global_feature, global_traces = global_model(pointcloud, return_intermediate=True)
        point_feature, point_traces = point_model(pointcloud, return_intermediate=True)

    print("=== PointNextEncoder Example ===")
    print("input:", tuple(pointcloud.shape))
    for stage_index, (position, feature) in enumerate(zip(global_traces["stage_positions"], global_traces["stage_features"])):
        print(f"stage_{stage_index}_position:", tuple(position.shape))
        print(f"stage_{stage_index}_feature:", tuple(feature.shape))
    print("global_feature:", tuple(global_feature.shape))
    print("point_feature:", tuple(point_feature.shape))
    print("point_feature_position:", tuple(point_traces["final_position"].shape))


if __name__ == "__main__":
    example()