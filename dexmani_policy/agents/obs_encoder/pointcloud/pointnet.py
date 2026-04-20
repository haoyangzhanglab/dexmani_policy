import torch
import torch.nn as nn
from typing import List, Dict


def create_mlp(
    in_channels: int,
    layer_channels: List[int],
    activation: type[nn.Module] = nn.ReLU,
):
    layers = []
    prev = in_channels
    for h in layer_channels:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        prev = h
    return nn.Sequential(*layers)


class PointNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 256,
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        self.input_channels = input_channels
        self.output_channels = output_channels

        hidden_channels = [64, 128, 256, 512] if input_channels > 3 else [64, 128, 256]
        self.mlp = create_mlp(input_channels, hidden_channels)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], output_channels),
            nn.LayerNorm(output_channels),
        )

    def forward(self, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        point_feature = self.mlp(pointcloud[..., : self.input_channels])
        point_feature = self.output_projection(point_feature)
        global_token = point_feature.amax(dim=1)
        return {"global_token": global_token}

    @property
    def out_dim(self) -> int:
        return self.output_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (1, self.output_channels)


class MultiStagePointNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, but got {num_layers}")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=False)
        self.output_projection = nn.Conv1d(hidden_channels * num_layers, output_channels, kernel_size=1)

        self.point_layers = nn.ModuleList(
            [nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1) for _ in range(num_layers)]
        )
        self.global_fusion_layers = nn.ModuleList(
            [nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1) for _ in range(num_layers)]
        )

    def forward(self, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        point_feature = pointcloud[..., : self.input_channels].transpose(1, 2)
        point_feature = self.activation(self.input_projection(point_feature))

        stage_feature_list = []
        for point_layer, global_fusion_layer in zip(self.point_layers, self.global_fusion_layers):
            point_feature = self.activation(point_layer(point_feature))
            global_point_feature = point_feature.amax(dim=-1, keepdim=True)
            point_feature = torch.cat([point_feature, global_point_feature.expand_as(point_feature)], dim=1)
            point_feature = self.activation(global_fusion_layer(point_feature))
            stage_feature_list.append(point_feature)

        point_feature = torch.cat(stage_feature_list, dim=1)
        point_feature = self.output_projection(point_feature)
        global_token = point_feature.amax(dim=-1)
        return {"global_token": global_token}

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

    print("=== PointNet Example ===")
    pointnet = PointNet(input_channels=6, output_channels=256)
    with torch.no_grad():
        out = pointnet(pointcloud)
    print("input:", tuple(pointcloud.shape))
    print("global_token:", tuple(out["global_token"].shape))
    print("out_dim:", pointnet.out_dim)
    print("out_shape:", pointnet.out_shape)

    print("=== MultiStagePointNet Example ===")
    multistage_pointnet = MultiStagePointNet(input_channels=6, output_channels=128)
    with torch.no_grad():
        out = multistage_pointnet(pointcloud)
    print("input:", tuple(pointcloud.shape))
    print("global_token:", tuple(out["global_token"].shape))
    print("out_dim:", multistage_pointnet.out_dim)
    print("out_shape:", multistage_pointnet.out_shape)


if __name__ == "__main__":
    example()