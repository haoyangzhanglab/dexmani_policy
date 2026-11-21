import torch
from torch import nn


class FingerTactileEncoder(nn.Module):
    """Encode aggregated forces in the sensor's frame and scale."""

    def __init__(self) -> None:
        super().__init__()
        self.force_encoder = nn.Sequential(
            nn.Linear(3, 64, bias=False), nn.SiLU(), nn.Linear(64, 64, bias=False),
        )

    def forward(self, finger_tactile_force: torch.Tensor) -> dict[str, torch.Tensor]:
        local_tokens = self.force_encoder(finger_tactile_force.to(dtype=self.force_encoder[0].weight.dtype))
        return {"local_tokens": local_tokens, "global_token": local_tokens.mean(dim=1)}


class TactileFusion(nn.Module):
    """Inject finger-aligned and shared tactile residuals through a shared contact gate."""

    def __init__(self) -> None:
        super().__init__()
        self.local_projection = nn.Linear(64, 192, bias=False)
        self.global_projection = nn.Linear(64, 192, bias=False)
        self.local_gate = nn.Linear(256, 1)
        self.global_gate = nn.Linear(256, 1)
        nn.init.constant_(self.local_gate.bias, -1.5)
        nn.init.constant_(self.global_gate.bias, -1.5)

    def forward(
        self, hand_tokens: torch.Tensor, tactile_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        local_tokens = tactile_features["local_tokens"]
        global_token = tactile_features["global_token"]
        finger_tokens = hand_tokens[:, 1:]
        local_gate_input = torch.cat((local_tokens, finger_tokens), dim=-1)
        global_hand_state = hand_tokens.mean(dim=1)
        global_gate_input = torch.cat((global_token, global_hand_state), dim=-1)
        contact_gate = self.global_gate(global_gate_input).sigmoid()
        local_gate = self.local_gate(local_gate_input).sigmoid()
        local_update = contact_gate.unsqueeze(1) * local_gate * self.local_projection(local_tokens)
        global_update = contact_gate * self.global_projection(global_token)
        hand_tokens = torch.cat((hand_tokens[:, :1], finger_tokens + local_update), dim=1)
        return hand_tokens + global_update.unsqueeze(1)
