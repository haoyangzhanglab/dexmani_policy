"""Force-guided attention fusion (FGAF) for visual and tactile features."""

import torch
from torch import nn


class ForceGuidedAttentionFusion(nn.Module):
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
        self.cross_attn = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, visual_feat: torch.Tensor, tactile_feat: torch.Tensor, force_guide: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.visual_proj(visual_feat)
        tactile = self.tactile_proj(tactile_feat)
        query = self.force_proj(force_guide).unsqueeze(1)
        memory = torch.stack([visual, tactile], dim=1)
        attended_feat = self.cross_attn(tgt=query, memory=memory)
        return attended_feat.squeeze(1)
