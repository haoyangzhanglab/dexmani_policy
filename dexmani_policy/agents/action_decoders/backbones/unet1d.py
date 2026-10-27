"""Conditional temporal UNet with FiLM action denoising."""

import torch
import torch.nn as nn

from .modules import TimestepMLP, batch_timesteps


def validate_unet_horizon(horizon: int, down_dims) -> None:
    if not down_dims:
        raise ValueError("UNet down_dims must be nonempty")
    divisor = 2 ** (len(down_dims) - 1)
    if horizon % divisor:
        raise ValueError(f"UNet horizon must be divisible by {divisor}, got {horizon}")


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvBlock1D(in_channels, out_channels, kernel_size, n_groups=n_groups),
                ConvBlock1D(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        self.cond_predict_scale = cond_predict_scale
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            # Diffusion Policy FiLM uses scale directly, without an identity offset.
            scale, bias = embed.chunk(2, dim=1)
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        context_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    ):
        super().__init__()

        if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("UNet kernel_size must be a positive odd integer")

        self.diffusion_step_encoder = TimestepMLP(
            pos_emb_dim=diffusion_step_embed_dim, output_dim=diffusion_step_embed_dim
        )
        cond_dim = diffusion_step_embed_dim + context_dim
        block_kwargs = dict(
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            ConditionalResidualBlock1D(mid_dim, mid_dim, **block_kwargs)
            for _ in range(2)
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, **block_kwargs),
                        ConditionalResidualBlock1D(dim_out, dim_out, **block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, **block_kwargs),
                        ConditionalResidualBlock1D(dim_in, dim_in, **block_kwargs),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1),
                    ]
                )
            )

        start_dim = down_dims[0]
        self.final_conv = nn.Sequential(
            ConvBlock1D(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, x, timestep, context):
        x = x.transpose(1, 2)

        timestep = batch_timesteps(timestep, x)
        t_embed = self.diffusion_step_encoder(timestep)

        cond = torch.cat([t_embed, context], dim=-1)

        skips = []
        for index, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, cond)
            x = resnet2(x, cond)
            if index > 0:
                skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)
