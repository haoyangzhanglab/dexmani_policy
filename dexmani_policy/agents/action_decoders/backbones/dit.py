"""DiT with action tokens and global observation conditioning."""

import torch
from torch import nn
from timm.models.vision_transformer import Mlp

from .modules import (
    AdaLNOutput,
    DiTTimestepEmbedding,
    SelfAttention,
    WEIGHT_INIT_STD,
    approximate_gelu,
    batch_timesteps,
    initialize_linear,
    modulate,
)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approximate_gelu,
            drop=0,
        )

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        cond_dim: int,
        n_emb: int = 512,
        num_heads: int = 8,
        n_layers: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.cond_dim = cond_dim
        self.action_dim = action_dim

        self.x_embedder = nn.Linear(action_dim, n_emb)
        self.time_embedder = DiTTimestepEmbedding(n_emb)
        self.cond_embedder = nn.Linear(cond_dim, n_emb)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, n_emb))

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=n_emb,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer = AdaLNOutput(
            n_emb, action_dim, nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(initialize_linear)

        nn.init.normal_(self.pos_embed, mean=0.0, std=WEIGHT_INIT_STD)

        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)

        nn.init.normal_(self.cond_embedder.weight, mean=0.0, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.cond_embedder.bias)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=WEIGHT_INIT_STD)

        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x, timestep, context):
        t = batch_timesteps(timestep, x)

        x = self.x_embedder(x)
        x = x + self.pos_embed.to(dtype=x.dtype)
        t_emb = self.time_embedder(t)

        context = context.reshape(context.shape[0], -1)
        cond_emb = self.cond_embedder(context)

        c = t_emb + cond_emb

        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)
