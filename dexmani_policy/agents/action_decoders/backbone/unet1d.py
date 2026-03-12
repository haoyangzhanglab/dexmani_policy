import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from dexmani_policy.agents.common.optim_util import get_default_optim_group

#################################################################################
#                               去噪/加噪步数编码                                    
#################################################################################
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#################################################################################
#                               特征调制                                    
#################################################################################
class CrossAttention(nn.Module):

    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):

        query = self.query_proj(x)
        key = self.key_proj(cond)
        value = self.value_proj(cond)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output


#################################################################################
#                               一维卷积组件                                    
#################################################################################
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_dim: int,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 condition_type: str = 'film'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        self.out_channels = out_channels
        self.condition_type = condition_type
        self.use_cross_attention = (condition_type == 'cross_attention_film')
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

        cond_channels = out_channels * 2
        if condition_type == 'film':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'), 
            )
        elif condition_type == 'cross_attention_film':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        else:
            raise NotImplementedError(f"Unsupported condition type: {condition_type}")

    def apply_film(self, out, embed):
        b, _, t = embed.shape
        embed = embed.reshape(b, 2, self.out_channels, t)
        scale = embed[:, 0]
        bias  = embed[:, 1]
        return scale * out + bias

    def forward(self, x, cond=None):
        out = self.blocks[0](x)
        if cond is not None:
            if self.use_cross_attention:
                embed = self.cond_encoder(x.permute(0, 2, 1), cond).permute(0, 2, 1)
            else:
                embed = self.cond_encoder(cond)
            out = self.apply_film(out, embed)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


#################################################################################
#                               UNet1D骨干网络                                    
#################################################################################
class ConditionalUnet1D(nn.Module):

    def __init__(
            self, 
            input_dim,
            global_cond_dim,
            diffusion_step_embed_dim=256,
            down_dims=[256,512,1024],
            kernel_size=5,
            n_groups=8,
            condition_type='film',
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            ):
        super().__init__()

        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        
        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        self.up_modules = up_modules
        
        start_dim = down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    
    def get_optim_groups(self, weight_decay):
        return get_default_optim_group(self, weight_decay)


    def forward(self, x, timestep, global_cond):

        x = einops.rearrange(x, 'b h t -> b t h')

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        t_embed = self.diffusion_step_encoder(timestep)
        
        if 'cross_attention' in self.condition_type:
            # 判断 global_cond是否满足shape: (batch, n_obs, cond_dim)
            assert len(global_cond.shape) == 3, f"Expected global_cond shape (batch, n_obs, cond_dim) when use cross-attn, but got {global_cond.shape}"
            t_embed = t_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
        global_cond = torch.cat([t_embed, global_cond], axis=-1)
        
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_cond)
                x = resnet2(x, global_cond)
            else:
                x = resnet(x)
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_cond)
            else:
                x = mid_module(x)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_cond)
                x = resnet2(x, global_cond)
            else:
                x = resnet(x)
                x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')

        return x