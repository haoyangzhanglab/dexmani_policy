"""Zero-referenced tactile cross-attention with channel and global gating."""

import torch
from torch import nn


class TactileGatedFusion(nn.Module):
    """Fuse wrist-first geometry [B,6,192] with five-finger force [B,5,3].

    Forces retain the source's frame and scale; zero force leaves geometry unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.force_encoder = nn.Sequential(
            nn.Linear(3, 64, bias=False), nn.SiLU(), nn.Linear(64, 64, bias=False),
        )
        self.tactile_channel_gate = nn.Linear(64, 64, bias=False)
        self.finger_embedding = nn.Parameter(torch.empty(1, 5, 64))
        self.geometry_norm = nn.LayerNorm(192)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=192, num_heads=4, kdim=64, vdim=64,
            batch_first=True, dropout=0.0, bias=False,
        )
        self.global_gate = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.tactile_channel_gate.weight)
        nn.init.trunc_normal_(self.finger_embedding, std=0.02)
        nn.init.constant_(self.global_gate[-2].bias, -1.5)

    def forward(self, geometry: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        tactile = self.force_encoder(force)
        tactile = tactile * torch.sigmoid(self.tactile_channel_gate(tactile))
        global_tactile = tactile.mean(dim=1)
        # Add identity only to keys to preserve zero-force values after training.
        key = tactile + self.finger_embedding.to(dtype=tactile.dtype)
        query = self.geometry_norm(geometry)
        tactile_context, _ = self.cross_attn(query, key, tactile, need_weights=False)
        gate = self.global_gate(global_tactile)
        return geometry + gate.unsqueeze(1) * tactile_context
