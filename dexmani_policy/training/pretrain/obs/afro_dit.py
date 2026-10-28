"""AFRO feature DiT, adapted from KolaKivy/AFRO.

Commit: 283024a49e2c14bc030debf15383a4e7ee73836d
Source: afro_workspace/model/transformer/DiT1D.py
Changes: active 4-layer/4-head defaults; remove unused parameterization helpers
and the unused pred_norm layer.
License: MIT, Copyright (c) 2025 Qiwei Liang; see licenses/AFRO.txt.
"""

import math

import torch
from torch import nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # (B, hidden_size)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTFeaturePredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        timestep_embed_dim=256,
    ):
        super().__init__()
        self.noise_proj = nn.Linear(input_dim, hidden_size)

        self.vis_proj = nn.Linear(input_dim, hidden_size)
        self.act_proj = nn.Linear(16, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=timestep_embed_dim)
        self.ca_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for blk in self.blocks:
            last = blk.adaLN_modulation[-1]
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, z, t, vis, act):
        n0 = self.noise_proj(z)                 # (B, H)
        x = n0.unsqueeze(1)                     # (B, 1, H)
        x = x + self.pos_embed

        t_emb = self.t_embedder(t)              # (B, H)
        v = self.vis_proj(vis)                  # (B, H)
        a = self.act_proj(act)                  # (B, H)
        c = self.ca_proj(torch.cat([t_emb, v, a], dim=-1))  # (B, H)

        for blk in self.blocks:
            x = blk(x, c)

        token_out = x[:, 0, :]                  # (B, H)
        pred = self.output_layer(token_out)     # (B, D)
        return pred
