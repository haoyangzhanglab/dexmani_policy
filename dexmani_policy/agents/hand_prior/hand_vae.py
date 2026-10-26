"""Full-chunk temporal hand VAE adapted from arXiv:2609.19666."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS


class HandVAE(nn.Module):
    """Bidirectional VAE; callers supply FP32 tensors and disable autocast."""

    def __init__(
        self,
        hand_dim: int = 12,
        horizon: int = 32,
        latent_dim: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        mlp_ratio: float = 2,
    ) -> None:
        super().__init__()
        dimensions = dict(
            hand_dim=hand_dim,
            horizon=horizon,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.model_config = {key: positive_int(value, key) for key, value in dimensions.items()}
        if horizon < 3:
            raise ValueError("HandVAE horizon must be at least 3 for second differences")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive and finite")
        feedforward_dim = positive_int(int(hidden_dim * mlp_ratio), "feedforward_dim")
        self.model_config["mlp_ratio"] = mlp_ratio

        self.hand_dim, self.horizon = hand_dim, horizon
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.encoder_input = nn.Linear(hand_dim, hidden_dim)
        self.encoder_q0 = nn.Linear(hand_dim, hidden_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_q0 = nn.Linear(hand_dim, hidden_dim)
        self.encoder_position = nn.Parameter(torch.empty(1, horizon + 1, hidden_dim))
        self.decoder_position = nn.Parameter(torch.empty(1, horizon + 1, hidden_dim))
        nn.init.normal_(self.encoder_position, std=0.02)
        nn.init.normal_(self.decoder_position, std=0.02)
        # Independent layers avoid TransformerEncoder's cloned initialization.
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    n_heads,
                    feedforward_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    n_heads,
                    feedforward_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_output = nn.Linear(hidden_dim, hand_dim)
        self.register_buffer("action_mean", torch.zeros(hand_dim))
        self.register_buffer("action_std", torch.ones(hand_dim))
        self.register_buffer("joint_weights", torch.ones(hand_dim))
        self.register_buffer("ready", torch.tensor(False))

    def normalize_actions(self, actions: Tensor, q0: Tensor) -> Tensor:
        """Physical commands -> standardized displacement from measured q0."""
        return (actions - q0[:, None] - self.action_mean) / self.action_std

    def denormalize_actions(self, relative: Tensor, q0: Tensor) -> Tensor:
        return relative * self.action_std + self.action_mean + q0[:, None]

    def encode(self, actions: Tensor, q0: Tensor) -> tuple[Tensor, Tensor]:
        relative = self.normalize_actions(actions, q0)
        tokens = torch.cat((self.encoder_q0(q0)[:, None], self.encoder_input(relative)), dim=1)
        tokens = tokens + self.encoder_position
        for layer in self.encoder:
            tokens = layer(tokens)
        encoded = self.encoder_norm(tokens[:, 1:])
        return self.fc_mu(encoded), self.fc_log_var(encoded)

    def decode(self, z: Tensor, q0: Tensor) -> Tensor:
        """Latent chunk -> standardized relative commands, with latent gradients."""
        tokens = torch.cat((self.decoder_q0(q0)[:, None], self.decoder_input(z)), dim=1)
        tokens = tokens + self.decoder_position
        for layer in self.decoder:
            tokens = layer(tokens)
        return self.decoder_output(self.decoder_norm(tokens[:, 1:]))

    def decode_actions(self, z: Tensor, q0: Tensor) -> Tensor:
        return self.denormalize_actions(self.decode(z, q0), q0)

    def reconstruction_terms(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Joint-weighted Huber means for position, velocity, and acceleration."""
        terms = {}
        for order, name in enumerate(("position", "velocity", "acceleration")):
            if order:
                pred, target = torch.diff(pred, dim=1), torch.diff(target, dim=1)
            penalty = F.huber_loss(pred, target, reduction="none", delta=1.0)
            terms[name] = (penalty * self.joint_weights).mean()
        return terms

    def reconstruction_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        terms = self.reconstruction_terms(pred, target)
        return terms["position"] + 0.2 * terms["velocity"] + 0.05 * terms["acceleration"]

    def freeze(self) -> HandVAE:
        self.requires_grad_(False)
        self.eval()
        return self

    def validate_state(self) -> None:
        if not bool(self.ready):
            raise ValueError("HandVAE checkpoint is not ready")
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("HandVAE contains nonfinite tensors")
        if not (self.action_std > 0).all():
            raise ValueError("HandVAE action_std must be positive")
        if not (self.joint_weights > 0).all():
            raise ValueError("HandVAE joint_weights must be positive")


def load_hand_vae(path: str | Path) -> tuple[HandVAE, dict]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    codec = HandVAE(**payload["model_config"])
    codec.load_state_dict(payload["state_dict"], strict=True)
    codec.validate_state()
    validate_codec_metadata(payload["metadata"], codec.model_config)
    return codec.freeze(), payload


def validate_codec_metadata(metadata, model_config):
    if not isinstance(metadata, dict):
        raise ValueError("HandVAE checkpoint requires embedded metadata")
    if metadata["model_config"] != model_config:
        raise ValueError("HandVAE metadata/model architecture mismatch")
    contract = metadata["hand_data"]["contract"]
    if any(contract[key] != value for key, value in HAND_ACTION_SEMANTICS.items()):
        raise ValueError("HandVAE action semantics mismatch")
    if contract["horizon"] != model_config["horizon"]:
        raise ValueError("HandVAE metadata horizon mismatch")
    if not math.isfinite(contract["dt"]) or contract["dt"] <= 0:
        raise ValueError("HandVAE requires positive finite dt")
    if metadata["loss_reduction"]["kl"] != "mean_over_batch_time_latent":
        raise ValueError("HandVAE KL reduction mismatch")
