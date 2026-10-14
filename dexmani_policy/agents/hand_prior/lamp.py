"""LAMP's history-only hand prior and differentiable pointwise decoder."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from dexmani_policy.utils.validation import positive_int


class HandMotionPrior(nn.Module):
    """Predict a normalized next hand target from raw command history.

    The encoder never receives the target being supervised. ``decode`` returns
    normalized hand coordinates and remains differentiable after ``freeze``.
    Normalization and latent scaling are tensor buffers, so a policy checkpoint
    contains everything needed for inference without reopening prior files.
    """

    def __init__(self, hand_dim=12, history_steps=8, latent_dim=2, hidden_dim=256):
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
            nn.Linear(history_steps * hand_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hand_dim),
        )
        self.register_buffer("hand_scale", torch.ones(hand_dim))
        self.register_buffer("hand_offset", torch.zeros(hand_dim))
        self.register_buffer("latent_scale", torch.ones(latent_dim))
        self.register_buffer("ready", torch.tensor(False))

    def normalize_hand(self, raw):
        raw = torch.as_tensor(raw, device=self.hand_scale.device, dtype=self.hand_scale.dtype)
        if raw.ndim < 1 or raw.shape[-1] != self.hand_dim:
            raise ValueError(f"Expected hand feature width {self.hand_dim}, got {tuple(raw.shape)}")
        return raw * self.hand_scale + self.hand_offset

    def unnormalize_hand(self, normalized):
        if normalized.ndim < 1 or normalized.shape[-1] != self.hand_dim:
            raise ValueError("Normalized hand feature width does not match prior")
        return (normalized - self.hand_offset) / self.hand_scale

    def encode_history(self, raw_history):
        history = self.normalize_hand(raw_history)
        if history.ndim < 3 or history.shape[-2] != self.history_steps:
            raise ValueError(f"Expected [..., {self.history_steps}, {self.hand_dim}] history")
        hidden = self.encoder(history.flatten(start_dim=-2))
        return self.fc_mu(hidden), self.fc_log_var(hidden)

    def decode(self, z):
        if z.ndim < 1 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent feature width {self.latent_dim}")
        return self.decoder(z)

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self

    @torch.enable_grad()
    def invert(self, target_raw, history_raw, steps=100, lr=0.05, center_penalty=0.0):
        """Find absolute latent labels, initialized from each target's history.

        Default zero center_penalty preserves reconstruction-only inversion.
        A positive value adds mean-squared distance to the source history mean
        in the prior's raw latent coordinates. Select by the same objective,
        including the initial mean: its zero penalty guarantees reconstruction
        cannot get worse than initialization. Summed per-sample objectives keep
        optimization independent of batch size. Prior gradients never accumulate.
        """
        if type(steps) is not int or steps < 0:
            raise ValueError("Inversion steps must be a nonnegative integer")
        if not 0 < lr < float("inf"):
            raise ValueError("Inversion learning rate must be positive and finite")
        if not 0 <= center_penalty < float("inf"):
            raise ValueError("Inversion center_penalty must be nonnegative and finite")
        with torch.no_grad():
            target = self.normalize_hand(target_raw)
            mu, _ = self.encode_history(history_raw)
            if target.shape[:-1] != mu.shape[:-1] or target.numel() == 0:
                raise ValueError("Inversion requires nonempty matching target/history batches")
            if not torch.isfinite(target).all() or not torch.isfinite(mu).all():
                raise ValueError("Inversion targets and encoded histories must be finite")
            best = mu.detach().clone()
            best_objective = (self.decode(best) - target).square().mean(dim=-1)
            if not torch.isfinite(best_objective).all():
                raise ValueError("Prior decoder produced nonfinite initialization")
        z = best.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            objective = (self.decode(z) - target).square().mean(dim=-1)
            if center_penalty:
                objective = objective + center_penalty * (z - mu).square().mean(dim=-1)
            if not torch.isfinite(objective).all():
                raise ValueError("Decoder inversion became nonfinite")
            (gradient,) = torch.autograd.grad(objective.sum(), z)
            if not torch.isfinite(gradient).all():
                raise ValueError("Decoder inversion gradient became nonfinite")
            z.grad = gradient
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                objective = (self.decode(z) - target).square().mean(dim=-1)
                if center_penalty:
                    objective = objective + center_penalty * (z - mu).square().mean(dim=-1)
                improved = torch.isfinite(objective) & (objective < best_objective)
                best = torch.where(improved.unsqueeze(-1), z.detach(), best)
                best_objective = torch.where(improved, objective, best_objective)
        return best.detach()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prior(path):
    """Restore a complete artifact, retaining the hash of exactly these bytes."""
    raw = Path(path).expanduser().read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("_format") != "lamp_prior.v1":
        raise ValueError("Expected a lamp_prior.v1 checkpoint")
    prior = HandMotionPrior(**payload["model_config"])
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("Prior model_state_dict must contain only tensors")
    prior.load_state_dict(state, strict=True)
    if not bool(prior.ready):
        raise ValueError("Prior checkpoint is not ready")
    if not all(torch.isfinite(value).all() for value in prior.state_dict().values()):
        raise ValueError("Prior checkpoint contains nonfinite tensors")
    if not (prior.hand_scale > 0).all() or not (prior.latent_scale > 0).all():
        raise ValueError("Prior normalization scales must be positive")
    payload["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return prior.freeze(), payload
