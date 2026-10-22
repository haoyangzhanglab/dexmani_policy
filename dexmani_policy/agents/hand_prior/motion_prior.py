"""LAMP's history-only hand prior and differentiable pointwise decoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from dexmani_policy.utils.config import positive_int


class HandMotionPrior(nn.Module):
    """History-conditioned hand prior with a differentiable pointwise decoder."""

    def __init__(
        self, hand_dim: int = 12, history_steps: int = 8, latent_dim: int = 2, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.model_config = {
            "hand_dim": positive_int(hand_dim, "hand_dim"),
            "history_steps": positive_int(history_steps, "history_steps"),
            "latent_dim": positive_int(latent_dim, "latent_dim"),
            "hidden_dim": positive_int(hidden_dim, "hidden_dim"),
        }
        self.hand_dim, self.history_steps = hand_dim, history_steps
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(history_steps * hand_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hand_dim),
        )
        self.register_buffer("hand_scale", torch.ones(hand_dim))
        self.register_buffer("hand_offset", torch.zeros(hand_dim))
        self.register_buffer("ready", torch.tensor(False))

    def normalize_hand(self, raw: Tensor) -> Tensor:
        raw = torch.as_tensor(raw, device=self.hand_scale.device, dtype=self.hand_scale.dtype)
        if raw.shape[-1:] != (self.hand_dim,):
            raise ValueError(f"Expected hand coordinates with final dimension {self.hand_dim}")
        return raw * self.hand_scale + self.hand_offset

    def unnormalize_hand(self, normalized: Tensor) -> Tensor:
        return (normalized - self.hand_offset) / self.hand_scale

    def encode_history(self, raw_history: Tensor) -> tuple[Tensor, Tensor]:
        history = self.normalize_hand(raw_history)
        if history.ndim < 3 or history.shape[-2:] != (self.history_steps, self.hand_dim):
            raise ValueError(f"Expected [..., {self.history_steps}, {self.hand_dim}] history")
        hidden = self.encoder(history.flatten(start_dim=-2))
        return self.fc_mu(hidden), self.fc_log_var(hidden)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def freeze(self) -> HandMotionPrior:
        self.requires_grad_(False)
        self.eval()
        return self

    def validate_state(self) -> None:
        if not self.ready:
            raise ValueError("Prior checkpoint is not ready")
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Prior checkpoint contains nonfinite tensors")
        if (self.hand_scale <= 0).any():
            raise ValueError("Prior normalization scales must be positive")


def load_motion_prior(path: str | Path) -> tuple[HandMotionPrior, dict]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    prior = HandMotionPrior(**payload["model_config"])
    prior.load_state_dict(payload["model_state_dict"], strict=True)
    prior.validate_state()
    return prior.freeze(), payload
