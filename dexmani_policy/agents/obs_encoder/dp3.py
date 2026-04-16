import torch
import torch.nn as nn
from typing import Dict
from dexmani_policy.agents.common.mlp import create_mlp
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext import PointNextEncoder


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
    """点云编码器，统一返回 (patch_token, patch_center, global_token)。

    point_wise=True 时输出逐点特征 (B, N, C)：
        patch_token = pointnet(pc), patch_center = pc 坐标, global_token = max pool
    point_wise=False 时输出全局特征 (B, C)：
        patch_token = global_token, patch_center = 零占位, global_token = feat.unsqueeze(1)
    """

    def __init__(
        self,
        type: str = "dp3",
        pc_dim: int = 3,
        pc_out_dim: int = 256,
        point_wise: bool = False,
    ):
        super().__init__()

        self.pc_out_dim = pc_out_dim
        self.point_wise = point_wise

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

    def forward(self, pointcloud: torch.Tensor) -> tuple:
        """
        Args:
            pointcloud: (B, N, C) 点云 tensor
        Returns:
            (patch_token, patch_center, global_token)
                patch_token:  (B, seq_len, C)  或 (B, 1, C)
                patch_center: (B, seq_len, 3)  或 (B, 1, 3) 零占位
                global_token: (B, 1, C)
        """
        feat = self.pointnet(pointcloud)

        if self.point_wise:
            patch_token = feat  # (B, N, C)
            patch_center = pointcloud[..., :3]  # (B, N, 3)
            global_token = patch_token.max(dim=1, keepdim=True).values  # (B, 1, C)
        else:
            global_token = feat.unsqueeze(1)  # (B, 1, C)
            patch_token = global_token
            patch_center = torch.zeros(feat.size(0), 1, 3, device=feat.device, dtype=feat.dtype)

        return patch_token, patch_center, global_token

    def get_optim_groups(self, weight_decay):
        return get_optim_group_with_no_decay(self, weight_decay)

    @property
    def out_shape(self):
        return self.pc_out_dim


def example():
    batch_size, num_points = 2, 1024
    state_dim = 19

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    point_cloud = torch.cat([xyz, rgb], dim=-1)

    state = torch.randn(batch_size, state_dim)

    print("=== DP3Encoder Example ===")
    print("point_cloud:", tuple(point_cloud.shape))
    print("joint_state:", tuple(state.shape))

    # point_wise=True 时返回三 tuple
    encoder = DP3Encoder(
        type="dp3",
        pc_dim=6,
        pc_out_dim=256,
        point_wise=True,
    )
    with torch.no_grad():
        patch_token, patch_center, global_token = encoder(point_cloud)
    print("point_wise=True:")
    print("  patch_token: ", tuple(patch_token.shape))
    print("  patch_center:", tuple(patch_center.shape))
    print("  global_token:", tuple(global_token.shape))

    # point_wise=False 时全局特征
    global_encoder = DP3Encoder(
        type="dp3",
        pc_dim=6,
        pc_out_dim=256,
        point_wise=False,
    )
    with torch.no_grad():
        patch_token, patch_center, global_token = global_encoder(point_cloud)
    print("point_wise=False:")
    print("  patch_token: ", tuple(patch_token.shape))
    print("  patch_center:", tuple(patch_center.shape))
    print("  global_token:", tuple(global_token.shape))

    # idp3 type
    idp3_encoder = DP3Encoder(
        type="idp3",
        pc_dim=6,
        pc_out_dim=128,
        point_wise=True,
    )
    with torch.no_grad():
        patch_token, patch_center, global_token = idp3_encoder(point_cloud)
    print("idp3 point_wise=True:")
    print("  patch_token: ", tuple(patch_token.shape))
    print("  patch_center:", tuple(patch_center.shape))
    print("  global_token:", tuple(global_token.shape))


if __name__ == "__main__":
    example()
