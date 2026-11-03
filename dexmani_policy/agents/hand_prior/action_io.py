"""SAT joint-trajectory tokens and action-coordinate velocity readout."""

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTXOutput
from dexmani_policy.agents.action_decoders.backbones.modules import initialize_linear


class SATActionIO(nn.Module):
    def __init__(self, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        self.x_embedder = nn.Sequential(
            nn.Linear(horizon, 64), nn.Mish(), nn.Linear(64, hidden_dim // 2),
        )
        self.robot_embed = nn.Embedding(2, hidden_dim // 4)
        self.joint_embed = nn.Embedding(12, hidden_dim // 4)
        self.register_buffer("robot_ids", torch.tensor([0] * 7 + [1] * 12), persistent=False)
        self.register_buffer("joint_ids", torch.tensor([*range(7), *range(12)]), persistent=False)
        self.final_layer = DiTXOutput(hidden_dim, horizon)
        self.apply(initialize_linear)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        trajectory = self.x_embedder(x.transpose(1, 2))
        descriptor = torch.cat((
            self.robot_embed(self.robot_ids), self.joint_embed(self.joint_ids),
        ), dim=-1)
        descriptor = descriptor.to(trajectory.dtype).expand(x.shape[0], -1, -1)
        return torch.cat((trajectory, descriptor), dim=-1)

    def decode_velocity(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.final_layer(tokens).transpose(1, 2)
