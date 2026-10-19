"""LAMP's history-only hand prior and differentiable pointwise decoder."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch
from torch import Tensor, nn

from dexmani_policy.utils.validation import positive_int


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
        self.register_buffer("latent_scale", torch.ones(latent_dim))
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
        if (self.hand_scale <= 0).any() or (self.latent_scale <= 0).any():
            raise ValueError("Prior normalization scales must be positive")

    @torch.enable_grad()
    def invert(
        self,
        target_raw: Tensor,
        history_raw: Tensor,
        steps: int = 100,
        lr: float = 0.05,
        center_penalty: float = 0.0,
    ) -> Tensor:
        """Retain each sample's best latent, including its initial history mean."""
        if steps < 0:
            raise ValueError("Inversion steps must be nonnegative")
        if not 0 < lr < float("inf"):
            raise ValueError("Inversion learning rate must be positive and finite")
        if not 0 <= center_penalty < float("inf"):
            raise ValueError("Inversion center_penalty must be nonnegative and finite")

        with torch.no_grad():
            target = self.normalize_hand(target_raw)
            mu, _ = self.encode_history(history_raw)
            if target.shape[:-1] != mu.shape[:-1] or target.numel() == 0:
                raise ValueError("Inversion requires nonempty matching target/history batches")
        best = mu.clone()
        best_objective = mu.new_full(mu.shape[:-1], torch.inf)
        # Accumulate finite checks on-device and synchronize once after inversion.
        finite = mu.new_ones((), dtype=torch.bool)
        z = mu.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        for step in range(steps + 1):
            with torch.set_grad_enabled(step < steps):
                objective = (self.decode(z) - target).square().mean(dim=-1)
                if center_penalty:
                    objective = objective + center_penalty * (z - mu).square().mean(dim=-1)

            with torch.no_grad():
                valid = torch.isfinite(objective)
                finite &= valid.all()
                improved = valid & (objective < best_objective)
                best = torch.where(improved.unsqueeze(-1), z, best)
                best_objective = torch.where(improved, objective, best_objective)

            if step == steps:
                break
            # Summed objectives keep each latent update independent of batch size.
            (gradient,) = torch.autograd.grad(objective.sum(), z)
            finite &= torch.isfinite(gradient).all()
            z.grad = gradient
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not finite:
            raise ValueError("Decoder inversion produced nonfinite objectives or gradients")
        return best.detach()


def load_motion_prior(path: str | Path) -> tuple[HandMotionPrior, dict]:
    """Restore a complete artifact, retaining the hash of exactly these bytes."""
    raw = Path(path).expanduser().read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("_format") != "lamp_prior.v1":
        raise ValueError("Expected a lamp_prior.v1 checkpoint")
    prior = HandMotionPrior(**payload["model_config"])
    prior.load_state_dict(payload["model_state_dict"], strict=True)
    prior.validate_state()
    payload["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return prior.freeze(), payload
