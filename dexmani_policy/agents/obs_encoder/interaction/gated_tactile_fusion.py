"""Same-finger tactile fusion with a valid-finger summary for the wrist."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..proprio import HandStateEncoder
from ..xhand_tactile import XHandForceEncoder


class GatedTactileFusion(nn.Module):
    KEYPOINTS = HandStateEncoder.KEYPOINTS

    def __init__(
        self,
        token_channels: int = 192,
        *,
        tactile_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.tactile_encoder = XHandForceEncoder() if tactile_encoder is None else tactile_encoder
        self.input_shape = tuple(self.tactile_encoder.input_shape)
        self.token_channels = token_channels

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
        tactile: Tensor | None = None,
        tactile_valid: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """Return complete [N,6,D] tokens; callers must not add hand or scene residuals again."""
        # FP32 residuals preserve small tactile updates under BF16 autocast.
        x = hand_scene["hand_query"].float() + hand_scene["relation_update"].float()
        batch = x.shape[0]
        if tactile is None:
            if tactile_valid is None:
                tactile_valid = torch.zeros(batch, 5, dtype=torch.bool, device=x.device)
            elif tactile_valid.any():
                raise ValueError("tactile=None cannot have valid fingers")
            tactile = x.new_zeros(batch, *self.input_shape)

        tokens = self.tactile_encoder(tactile, tactile_valid=tactile_valid)
        if tactile_valid is None:
            tactile_valid = torch.ones(batch, 5, dtype=torch.bool, device=x.device)
        local = torch.where(tactile_valid[..., None], tokens.float(), 0.0)
        summary = local.sum(dim=1) / tactile_valid.sum(dim=1, keepdim=True).clamp_min(1)
        available = tactile_valid.any(-1, keepdim=True)
        body_valid = torch.cat((available, tactile_valid), dim=1)

        query = self.query_norm(x.to(self.query_norm.weight.dtype))
        memory = torch.cat((summary[:, None], local), dim=1)
        memory = self.tactile_norm(memory.to(self.tactile_norm.weight.dtype))
        features = torch.cat((query, memory), dim=-1)
        delta = self.fusion_mlp(features.to(self.fusion_mlp[0].weight.dtype)).float()
        delta = torch.where(body_valid[..., None], delta, 0.0)
        gate_input = torch.cat(
            (features, hand_scene["geometry_descriptor"][..., 3:].float()), -1,
        )
        gate = self.gate_mlp(gate_input.to(self.gate_mlp[0].weight.dtype)).float().sigmoid()
        gate = gate * body_valid[..., None]
        tactile_update = gate * delta

        output = {"interaction_tokens": x + tactile_update}
        if return_intermediate:
            output.update(
                tactile_tokens=local,
                tactile_summary=summary,
                tactile_valid=tactile_valid,
                gate=gate.squeeze(-1),
                tactile_update=tactile_update,
            )
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
