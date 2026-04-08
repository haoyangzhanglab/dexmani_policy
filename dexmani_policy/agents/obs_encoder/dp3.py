import torch
import torch.nn as nn
from typing import List, Dict
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext import PointNextEncoder


def create_mlp(
    in_channels: int,
    layer_channels: List[int],
    activation: type[nn.Module] = nn.ReLU
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
        in_channels: int = 3,
        out_channels: int = 256,
        point_wise: bool = False,
    ):
        super().__init__()

        self.point_wise = point_wise
        self.out_channels = out_channels
        if in_channels > 3:
            block_channels = [64, 128, 256, 512]
        else:
            block_channels = [64, 128, 256]

        self.mlp = create_mlp(in_channels, block_channels)
        self.final_projection = nn.Sequential(
            nn.Linear(block_channels[-1], out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.final_projection(x)
        if not self.point_wise:
            return x.amax(dim=1)
        else:
            return x


class MultiStagePointNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=128,
        point_wise=True,
        h_dim=128,
        num_layers=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.point_wise = point_wise

        self.h_dim = h_dim
        self.num_layers = num_layers

        self.conv_in = nn.Conv1d(in_channels, h_dim, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))

    def forward(self, x):
        x = x.transpose(1, 2)
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

        if not self.point_wise:
            feat = x.amax(dim=-1)
        else:
            feat = x.transpose(1, 2)
        return feat


class DP3Encoder(nn.Module):

    def __init__(
        self,
        type: str = "dp3",
        pc_dim: int = 3,
        pc_out_dim: int = 256,
        point_wise: bool = False,
        state_dim: int = 19,
    ):
        super().__init__()

        self.modality_keys = ["point_cloud", "joint_state"]
        self.pc_out_dim = pc_out_dim
        self.state_out_dim = 64

        if type == "dp3":
            self.pointnet = PointNet(
                in_channels=pc_dim,
                out_channels=pc_out_dim,
                point_wise=point_wise
            )
        elif type == "idp3":
            self.pointnet = MultiStagePointNet(
                in_channels=pc_dim,
                out_channels=pc_out_dim,
                point_wise=point_wise
            )
        elif type == "pointnext":
            self.pointnet = PointNextEncoder(
                input_channels=pc_dim,
                output_channels=pc_out_dim,
                point_wise=point_wise,
            )
        else:
            raise ValueError(f"Unsupported pointnet type: {type}")

        self.state_mlp = create_mlp(state_dim, [64, self.state_out_dim])

    def forward(self, observations: Dict) -> torch.Tensor:
        for key in self.modality_keys:
            if key not in observations:
                raise KeyError(f"Required modality key '{key}' missed")
        point_cloud, state = observations["point_cloud"], observations["joint_state"]
        if point_cloud.shape[:-2] != state.shape[:-1]:
            raise ValueError(
                f"point_cloud leading dims {tuple(point_cloud.shape[:-2])} != "
                f"joint_state leading dims {tuple(state.shape[:-1])}"
            )

        pc_feat = self.pointnet(point_cloud)
        state_feat = self.state_mlp(state)

        if self.pointnet.point_wise:
            state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)

        feat = torch.cat([pc_feat, state_feat], dim=-1)
        return feat

    @property
    def out_shape(self):
        return self.pc_out_dim + self.state_out_dim

    def get_optim_groups(self, weight_decay):
        return get_optim_group_with_no_decay(self, weight_decay)


def example():
    batch_size, num_points = 2, 1024
    state_dim = 19

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    point_cloud = torch.cat([xyz, rgb], dim=-1)

    observations = {
        "point_cloud": point_cloud,
        "joint_state": torch.randn(batch_size, state_dim),
    }

    global_encoder = DP3Encoder(
        type="dp3",
        pc_dim=6,
        pc_out_dim=256,
        point_wise=False,
        state_dim=state_dim,
    )
    point_wise_encoder = DP3Encoder(
        type="dp3",
        pc_dim=6,
        pc_out_dim=256,
        point_wise=True,
        state_dim=state_dim,
    )

    with torch.no_grad():
        global_feat = global_encoder(observations)
        point_wise_feat = point_wise_encoder(observations)

    print("=== DP3Encoder Example ===")
    print("point_cloud:", tuple(observations["point_cloud"].shape))
    print("joint_state:", tuple(observations["joint_state"].shape))
    print("global_feat:", tuple(global_feat.shape))
    print("point_wise_feat:", tuple(point_wise_feat.shape))


if __name__ == "__main__":
    example()
