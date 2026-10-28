import torch
from torch import nn


class VisuoTactileInteraction(nn.Module):
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
