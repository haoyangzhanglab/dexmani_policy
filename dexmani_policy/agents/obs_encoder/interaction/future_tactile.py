"""Visual–tactile interaction and future-force-guided fusion."""

import torch
from torch import nn


class ForceGuidedFusion(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        tactile_dim: int,
        guide_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)
        self.force_proj = nn.Linear(guide_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, visual_feat: torch.Tensor, tactile_feat: torch.Tensor, force_guide: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.visual_proj(visual_feat)
        tactile = self.tactile_proj(tactile_feat)
        query = self.force_proj(force_guide).unsqueeze(1)
        memory = torch.stack([visual, tactile], dim=1)
        update = self.cross_attn(query, memory, memory, need_weights=False)[0]
        query = self.attn_norm(query + self.dropout(update))
        return self.ffn_norm(query + self.dropout(self.ffn(query))).squeeze(1)


class VisuoTactileAttention(nn.Module):
    def __init__(self, visual_dim: int, tactile_dim: int, num_fingers: int) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.tactile_norm = nn.LayerNorm(tactile_dim)
        self.tactile_attn = nn.MultiheadAttention(
            embed_dim=tactile_dim, num_heads=8, kdim=visual_dim, vdim=visual_dim,
            dropout=0.1, batch_first=True,
        )
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=visual_dim, num_heads=8, kdim=tactile_dim, vdim=tactile_dim,
            dropout=0.1, batch_first=True,
        )
        self.tactile_dropout = nn.Dropout(0.1)
        self.visual_dropout = nn.Dropout(0.1)
        self.finger_embedding = nn.Parameter(torch.empty(1, num_fingers, tactile_dim))
        nn.init.normal_(self.finger_embedding, mean=0.0, std=0.02)

    def forward(
        self, visual: torch.Tensor, tactile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.visual_norm(visual)
        t = self.tactile_norm(tactile)
        e = self.finger_embedding.to(dtype=t.dtype)
        # Both directions read the same pre-update tokens; identity only enters Q/K.
        delta_t = self.tactile_attn(query=t + e, key=v, value=v, need_weights=False)[0]
        delta_v = self.visual_attn(query=v, key=t + e, value=t, need_weights=False)[0]
        return (
            visual + self.visual_dropout(delta_v),
            tactile + self.tactile_dropout(delta_t),
        )
