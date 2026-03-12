import torch
import torch.nn as nn
from typing import List, Dict
from dexmani_policy.agents.common.optim_util import get_default_optim_group


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
        in_channels:int=3, 
        out_channels:int=256, 
        point_wise:bool=False,
    ):
        super().__init__()

        self.point_wise = point_wise
        if in_channels > 3:
            block_channels = [64, 128, 256, 512]
        else:
            block_channels = [64, 128, 256]
        
        self.mlp = create_mlp(in_channels, block_channels)
        self.final_projection =  nn.Sequential(
            nn.Linear(block_channels[-1], out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.final_projection(x) 
        if not self.point_wise:
            return x.amax(dim=1)    # [B, C]
        else:
            return x   # [B, N, C]


class MultiStagePointNet(nn.Module):

    def __init__(
        self,
        in_channels = 3,
        out_channels = 128,
        point_wise = True,            
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
            feat = x.amax(dim=-1)  # [B, C]
        else:
            feat = x.transpose(1, 2) # [B, N, C]
        return feat



class DP3Encoder(nn.Module):

    def __init__(
        self,
        type: str = "dp3",
        pc_dim: int = 3,
        pc_out_dim: int = 256,
        point_wise: bool = False,
        state_dim: int = 19,
        modality_keys: List[str] = ["point_cloud", "joint_state"], 
    ):
        super().__init__()

        self.modality_keys = modality_keys

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
        else:
            raise ValueError(f"Unsupported pointnet type: {type}")
        
        self.state_mlp = create_mlp(state_dim, [64, 64])


    def forward(self, observations: Dict):
        for key in self.modality_keys:
            assert key in observations, f"Required modality key '{key}' missed"
        point_cloud, state = observations["point_cloud"], observations["joint_state"]

        pc_feat = self.pointnet(point_cloud)
        state_feat = self.state_mlp(state)

        if self.pointnet.point_wise:
            state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)
        
        feat = torch.cat([pc_feat, state_feat], dim=-1)  
        return feat


    @property
    def out_shape(self):
        return self.pointnet.out_channels + self.state_mlp[-3].out_features
    
    def get_optim_groups(self, weight_decay):
        optim_groups = get_default_optim_group(self, weight_decay)
        return optim_groups



        


        

        
