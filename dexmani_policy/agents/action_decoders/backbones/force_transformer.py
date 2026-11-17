"""Causal future-force prediction conditioned on visual and tactile observations."""

import torch
from torch import nn

from .modules import SinusoidalTimestepEmbedding, batch_timesteps


class ForceTransformer(nn.Module):
    def __init__(
        self,
        horizon: int,
        cond_dim: int,
        force_dim: int,
        depth: int = 8,
        num_heads: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_emb = nn.Linear(force_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, embed_dim))
        self.time_emb = SinusoidalTimestepEmbedding(embed_dim)
        self.cond_obs_emb = nn.Linear(cond_dim, embed_dim)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.condition_mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.Mish(), nn.Linear(4 * embed_dim, embed_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.register_buffer(
            "causal_mask", torch.full((horizon, horizon), float("-inf")).triu(1),
            persistent=False,
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, force_dim)

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor) -> torch.Tensor:
        timesteps = batch_timesteps(timestep, x)
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        input_emb = self.input_emb(x)
        cond_embeddings = torch.cat((time_emb, self.cond_obs_emb(context)), dim=1)
        memory = self.condition_mlp(cond_embeddings + self.cond_pos_emb)

        x = input_emb + self.pos_emb
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.causal_mask, tgt_is_causal=True)
        return self.head(self.ln_f(x))
