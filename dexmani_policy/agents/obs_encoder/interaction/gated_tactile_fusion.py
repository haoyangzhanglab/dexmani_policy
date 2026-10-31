"""Same-finger tactile fusion with a five-finger summary for the wrist."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..xhand_tactile import XHandForceEncoder


class GatedTactileFusion(nn.Module):
    def __init__(
        self,
        token_channels: int = 192,
        *,
        tactile_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.tactile_encoder = XHandForceEncoder() if tactile_encoder is None else tactile_encoder
        tactile_dim = self.tactile_encoder.out_dim
        self.query_norm = nn.LayerNorm(token_channels)
        self.tactile_norm = nn.LayerNorm(tactile_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(token_channels + tactile_dim, tactile_dim), nn.SiLU(),
            nn.Linear(tactile_dim, token_channels),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(token_channels + tactile_dim + 4, tactile_dim),
            nn.SiLU(),
            nn.Linear(tactile_dim, 1),
        )

        # Nonzero residual weights let tactile and gate paths learn from step one.
        nn.init.normal_(self.fusion_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.fusion_mlp[-1].bias)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, math.log(0.1 / 0.9))

    def forward(
        self,
        hand_scene: Mapping[str, Tensor],
        *,
        tactile: Tensor,
    ) -> Tensor:
        """Return complete [N,6,D] tokens; callers must not add hand or scene residuals again."""
        # FP32 residuals preserve small tactile updates under BF16 autocast.
        x = hand_scene["hand_query"].float() + hand_scene["relation_update"].float()
        local = self.tactile_encoder(tactile).float()
        summary = local.mean(dim=1)

        query = self.query_norm(x.to(self.query_norm.weight.dtype))
        memory = torch.cat((summary[:, None], local), dim=1)
        memory = self.tactile_norm(memory.to(self.tactile_norm.weight.dtype))
        features = torch.cat((query, memory), dim=-1)
        delta = self.fusion_mlp(features.to(self.fusion_mlp[0].weight.dtype)).float()
        gate_input = torch.cat(
            (features, hand_scene["geometry_descriptor"][..., 3:].float()), -1,
        )
        gate = self.gate_mlp(gate_input.to(self.gate_mlp[0].weight.dtype)).float().sigmoid()
        tactile_update = gate * delta

        return x + tactile_update
