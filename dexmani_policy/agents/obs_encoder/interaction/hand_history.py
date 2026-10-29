import torch
from torch import nn


class HandHistoryEncoder(nn.Module):
    def __init__(self, n_obs_steps: int) -> None:
        super().__init__()
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be greater than 0")
        self.n_obs_steps = n_obs_steps
        self.spatial_attn_norm = nn.LayerNorm(192)
        self.spatial_attn = nn.MultiheadAttention(192, 4, dropout=0.0, batch_first=True)
        self.spatial_ffn_norm = nn.LayerNorm(192)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192)
        )
        self.time_embedding = nn.Parameter(torch.empty(1, n_obs_steps, 192))
        nn.init.normal_(self.time_embedding, std=0.02)
        self.temporal_attn_norm = nn.LayerNorm(192)
        self.temporal_attn = nn.MultiheadAttention(192, 4, dropout=0.0, batch_first=True)
        self.temporal_ffn_norm = nn.LayerNorm(192)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192)
        )
        # MultiheadAttention uses True to forbid reading strictly future frames.
        self.register_buffer(
            "causal_mask",
            torch.ones(n_obs_steps, n_obs_steps, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[1:] != (self.n_obs_steps, 6, 192):
            raise ValueError("tokens must have shape [B,n_obs_steps,6,192]")
        batch = tokens.shape[0]
        spatial = tokens.reshape(batch * self.n_obs_steps, 6, 192).float()
        normalized = self.spatial_attn_norm(spatial.to(self.spatial_attn_norm.weight.dtype))
        spatial = spatial + self.spatial_attn(
            normalized, normalized, normalized, need_weights=False
        )[0].float()
        normalized = self.spatial_ffn_norm(spatial.to(self.spatial_ffn_norm.weight.dtype))
        spatial = spatial + self.spatial_ffn(normalized).float()

        # Each temporal sequence contains one body slot from one batch item.
        temporal = (
            spatial.reshape(batch, self.n_obs_steps, 6, 192)
            .permute(0, 2, 1, 3)
            .reshape(batch * 6, self.n_obs_steps, 192)
        )
        temporal = temporal + self.time_embedding.float()
        normalized = self.temporal_attn_norm(temporal.to(self.temporal_attn_norm.weight.dtype))
        temporal = temporal + self.temporal_attn(
            normalized, normalized, normalized, attn_mask=self.causal_mask, need_weights=False
        )[0].float()
        normalized = self.temporal_ffn_norm(temporal.to(self.temporal_ffn_norm.weight.dtype))
        temporal = temporal + self.temporal_ffn(normalized).float()
        return temporal.reshape(batch, 6, self.n_obs_steps, 192).permute(0, 2, 1, 3)
