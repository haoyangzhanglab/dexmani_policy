"""Conditioned causal Transformer for single-observation force prediction."""

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.modules import (
    SinusoidalTimestepEmbedding,
    batch_timesteps,
)


class ForcePredictionTransformer(nn.Module):
    def __init__(
        self,
        horizon: int,
        cond_dim: int,
        force_dim: int,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_emb = nn.Linear(force_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalTimestepEmbedding(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 2, n_emb))
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.register_buffer(
            "mask", torch.full((horizon, horizon), float("-inf")).triu(1),
        )
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, force_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        elif isinstance(module, ForcePredictionTransformer):
            nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor) -> torch.Tensor:
        timesteps = batch_timesteps(timestep, x)
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        input_emb = self.input_emb(x)
        cond_embeddings = torch.cat((time_emb, self.cond_obs_emb(context)), dim=1)
        memory = self.encoder(self.drop(cond_embeddings + self.cond_pos_emb))
        x = self.drop(input_emb + self.pos_emb)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask)
        return self.head(self.ln_f(x))
