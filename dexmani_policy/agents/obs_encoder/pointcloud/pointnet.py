import torch
import torch.nn as nn


def build_mlp(
    channels: list[int],
    *,
    activation: type[nn.Module] = nn.GELU,
    normalize: bool = True,
    activate_last: bool = False,
) -> nn.Sequential:
    """Linear layers with optional LayerNorm and activation after hidden layers."""
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(channels, channels[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(channels) - 2 or activate_last:
            if normalize:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation())
    return nn.Sequential(*layers)


def build_point_mlp(input_channels: int) -> nn.Sequential:
    if input_channels not in (3, 6):
        raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
    channels = [input_channels, 64, 128, 256]
    if input_channels == 6:
        channels.append(512)
    return build_mlp(channels, activation=nn.ReLU, activate_last=input_channels == 3)


class PointNet(nn.Module):
    """Encode a point cloud into one global token, pooling before projection."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 256,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.mlp = build_point_mlp(input_channels)
        self.output_projection = nn.Sequential(
            nn.Linear(512 if input_channels == 6 else 256, output_channels),
            nn.LayerNorm(output_channels),
        )

    def forward(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        point_feature = self.mlp(pointcloud[..., : self.input_channels])
        global_token = point_feature.amax(dim=1)
        global_token = self.output_projection(global_token)
        return {"global_token": global_token}

    @property
    def out_dim(self) -> int:
        return self.output_channels


class PointNetDense(nn.Module):
    """Encode each point independently into a dense token sequence [B, N, D]."""

    def __init__(
        self,
        input_channels: int = 3,
        out_channels: int = 128,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels

        self.mlp = build_point_mlp(input_channels)
        self.final_proj = nn.Sequential(
            nn.Linear(512 if input_channels == 6 else 256, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        point_feature = self.mlp(pointcloud[..., : self.input_channels])
        return self.final_proj(point_feature)

    @property
    def out_dim(self) -> int:
        return self.out_channels


class MultiStagePointNet(nn.Module):
    """Fuse per-point and global features across successive PointNet stages."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, but got {num_layers}")

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=False)
        self.output_projection = nn.Conv1d(
            hidden_channels * num_layers, output_channels, kernel_size=1
        )

        self.point_layers = nn.ModuleList(
            [nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1) for _ in range(num_layers)]
        )
        self.global_fusion_layers = nn.ModuleList(
            [
                nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1)
                for _ in range(num_layers)
            ]
        )

    def forward(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        point_feature = pointcloud[..., : self.input_channels].transpose(1, 2)
        point_feature = self.activation(self.input_projection(point_feature))

        stage_feature_list = []
        for point_layer, global_fusion_layer in zip(self.point_layers, self.global_fusion_layers):
            point_feature = self.activation(point_layer(point_feature))
            global_point_feature = point_feature.amax(dim=-1, keepdim=True)
            point_feature = torch.cat(
                [point_feature, global_point_feature.expand_as(point_feature)], dim=1
            )
            point_feature = self.activation(global_fusion_layer(point_feature))
            stage_feature_list.append(point_feature)

        point_feature = torch.cat(stage_feature_list, dim=1)
        point_feature = self.output_projection(point_feature)
        global_token = point_feature.amax(dim=-1)
        return {"global_token": global_token}

    @property
    def out_dim(self) -> int:
        return self.output_channels
