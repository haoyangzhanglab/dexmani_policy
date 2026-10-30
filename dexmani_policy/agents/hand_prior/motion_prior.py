"""LAMP's history-only hand prior and differentiable pointwise decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from dexmani_policy.common.config import positive_int

from .losses import diagonal_gaussian_kl


class HandMotionPrior(nn.Module):
    """History-conditioned hand prior with a differentiable pointwise decoder."""

    coordinate_space = "raw_joint_targets_radians"
    hand_joint_order = ["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"]

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
        self.register_buffer("ready", torch.tensor(False))

    def encode_history(self, raw_history: Tensor) -> tuple[Tensor, Tensor]:
        if raw_history.ndim < 3 or raw_history.shape[-2:] != (self.history_steps, self.hand_dim):
            raise ValueError(f"Expected [..., {self.history_steps}, {self.hand_dim}] history")
        hidden = self.encoder(raw_history.flatten(start_dim=-2))
        return self.fc_mu(hidden), self.fc_log_var(hidden)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def compute_loss(self, history: Tensor, target: Tensor, beta: float):
        with torch.autocast(device_type=history.device.type, enabled=False):
            mu, log_var = self.encode_history(history.float())
            epsilon = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * log_var) * epsilon
            reconstruction = (self.decode(z) - target.float()).square().mean()
            kl = diagonal_gaussian_kl(mu, log_var)
        return reconstruction + beta * kl, reconstruction, kl

    def freeze(self) -> HandMotionPrior:
        self.requires_grad_(False)
        self.eval()
        return self

    def validate_state(self) -> None:
        if not self.ready:
            raise ValueError("Prior checkpoint is not ready")
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Prior checkpoint contains nonfinite tensors")


def load_motion_prior(path: str | Path, prior: HandMotionPrior) -> dict:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload["coordinate_space"] != HandMotionPrior.coordinate_space:
        raise ValueError("LAMP requires raw-radians prior coordinates")
    if payload["metadata"]["hand_joint_order"] != HandMotionPrior.hand_joint_order:
        raise ValueError("LAMP prior hand joint order differs from the local robot contract")
    if payload["model_config"] != prior.model_config:
        raise ValueError("LAMP prior architecture differs from the policy configuration")
    prior.load_state_dict(payload["model_state_dict"], strict=True)
    prior.validate_state()
    prior.freeze()
    return payload


def initial_hand_history(initial_hand, history_steps):
    return np.repeat(np.asarray(initial_hand, dtype=np.float32)[None], history_steps, axis=0)


def append_hand_history(history, command):
    # Allocate a new history so an in-flight inference keeps its original snapshot.
    command = np.asarray(command, dtype=history.dtype)
    return np.concatenate((history[1:], command[None]), axis=0)
