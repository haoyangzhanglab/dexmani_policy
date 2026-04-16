import torch
import torch.nn as nn
from typing import Dict

from dexmani_policy.agents.common.mlp import create_mlp
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay


class PointNet(nn.Module):
    """PointNet 风格全局向量提取器。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels > 3:
            block_channels = [64, 128, 256, 512]
        else:
            block_channels = [64, 128, 256]

        self.mlp = create_mlp(in_channels, block_channels)
        self.final_projection = nn.Sequential(
            nn.Linear(block_channels[-1], out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.mlp(pointcloud)
        x = self.final_projection(x)
        global_token = x.amax(dim=1)
        return {"global_token": global_token}

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(self, weight_decay)

    @property
    def out_shape(self) -> int:
        return self.out_channels


class MultiStagePointNet(nn.Module):
    """改进版 Multi-Stage PointNet 全局向量提取器。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        h_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.conv_in = nn.Conv1d(in_channels, h_dim, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))

    def forward(self, pointcloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = pointcloud.transpose(1, 2)
        y = self.act(self.conv_in(x))

        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.amax(dim=-1, keepdim=True)
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)

        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)
        global_token = x.amax(dim=-1)
        return {"global_token": global_token}

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(self, weight_decay)

    @property
    def out_shape(self) -> int:
        return self.out_channels


def example():
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    print("=== PointNet Example ===")
    model = PointNet(in_channels=6, out_channels=256)
    with torch.no_grad():
        out = model(pointcloud)
    print("input:", tuple(pointcloud.shape))
    print("global_token:", tuple(out["global_token"].shape))
    print("out_shape:", model.out_shape)

    print("\n=== MultiStagePointNet Example ===")
    model = MultiStagePointNet(in_channels=6, out_channels=128)
    with torch.no_grad():
        out = model(pointcloud)
    print("input:", tuple(pointcloud.shape))
    print("global_token:", tuple(out["global_token"].shape))
    print("out_shape:", model.out_shape)


if __name__ == "__main__":
    example()
