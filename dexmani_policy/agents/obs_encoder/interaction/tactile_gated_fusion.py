"""InterAct-only zero-referenced local and global tactile fusion."""

import torch
from torch import nn


class TactileGatedFusion(nn.Module):
    """Consume identity force (XHand Real: SDK / 255); all Value paths preserve physical zero."""

    def __init__(self):
        super().__init__()
        self.force_encoder = nn.Sequential(nn.Linear(3, 64, bias=False), nn.SiLU(), nn.Linear(64, 64, bias=False))
        self.geometry_norm = nn.LayerNorm(192)
        self.pool_geometry = nn.Linear(192, 32)
        self.pool_tactile = nn.Linear(64, 32, bias=False)
        self.pool_score = nn.Linear(32, 1, bias=False)
        self.gate = nn.Sequential(nn.Linear(320, 64), nn.SiLU(), nn.Linear(64, 1))
        self.residual = nn.Linear(64, 192, bias=False)
        nn.init.constant_(self.gate[-1].bias, -1.5)
        nn.init.xavier_uniform_(self.residual.weight, gain=0.1)

    def forward(self, geometry: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        tactile = self.force_encoder(force)
        normalized = self.geometry_norm(geometry)
        scores = self.pool_score(torch.tanh(self.pool_geometry(normalized[:, 1:]) + self.pool_tactile(tactile)))
        global_tactile = (scores.softmax(dim=1) * tactile).sum(1, keepdim=True)
        local = torch.cat((torch.zeros_like(global_tactile), tactile), 1)
        values = torch.cat((global_tactile, tactile), 1)
        gates = self.gate(torch.cat((normalized, local, global_tactile.expand(-1, 6, -1)), -1)).sigmoid()
        return geometry + (gates * self.residual(values)).float()
