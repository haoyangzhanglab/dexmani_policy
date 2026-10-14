"""Full-chunk temporal hand VAE from arXiv:2609.19666, adapted to one hand.

Both bidirectional time-token Transformers condition on the measured chunk-start
pose. The posterior is target-conditioned and preserves the temporal length.
Pre-LN, GELU, and learned temporal embeddings are explicit implementation choices
where the paper does not specify the Transformer block details.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from dexmani_policy.utils.validation import positive_int


class TemporalHandCodec(nn.Module):
    def __init__(self, hand_dim=12, horizon=16, latent_dim=9, hidden_dim=128,
                 n_layers=3, n_heads=4, mlp_ratio=2):
        super().__init__()
        dimensions = dict(hand_dim=hand_dim, horizon=horizon, latent_dim=latent_dim,
                          hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads)
        self.model_config = {key: positive_int(value, key) for key, value in dimensions.items()}
        if horizon < 3:
            raise ValueError("Temporal codec horizon must be at least 3 for second differences")
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if isinstance(mlp_ratio, bool) or not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
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
        # Construct each layer separately to avoid identical initial weights
        # from TransformerEncoder's cloning. No causal mask is supplied.
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, n_heads, feedforward_dim,
                                       dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, n_heads, feedforward_dim,
                                       dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_output = nn.Linear(hidden_dim, hand_dim)
        self.register_buffer("action_mean", torch.zeros(hand_dim))
        self.register_buffer("action_std", torch.ones(hand_dim))
        self.register_buffer("joint_weights", torch.ones(hand_dim))
        self.register_buffer("ready", torch.tensor(False))

    def _check_chunk(self, chunk, width, name):
        if chunk.ndim != 3 or chunk.shape[1:] != (self.horizon, width):
            raise ValueError(f"{name} must have shape [B,{self.horizon},{width}]")

    def _q0(self, q0, batch_size):
        q0 = torch.as_tensor(q0, device=self.action_mean.device, dtype=self.action_mean.dtype)
        if q0.shape != (batch_size, self.hand_dim):
            raise ValueError(f"q0 must have shape [B,{self.hand_dim}]")
        return q0

    def normalize_actions(self, actions, q0):
        """Physical commands -> standardized displacement from measured q0."""
        actions = torch.as_tensor(actions, device=self.action_mean.device, dtype=self.action_mean.dtype)
        self._check_chunk(actions, self.hand_dim, "actions")
        q0 = self._q0(q0, len(actions))
        return (actions - q0[:, None] - self.action_mean) / self.action_std

    def encode(self, actions, q0):
        relative = self.normalize_actions(actions, q0)
        q0 = self._q0(q0, len(relative))
        tokens = torch.cat((self.encoder_q0(q0)[:, None], self.encoder_input(relative)), dim=1)
        tokens = tokens + self.encoder_position
        for layer in self.encoder:
            tokens = layer(tokens)
        encoded = self.encoder_norm(tokens[:, 1:])
        return self.fc_mu(encoded), self.fc_log_var(encoded)

    def decode(self, z, q0):
        """Latent chunk -> standardized relative commands, with latent gradients."""
        z = torch.as_tensor(z, device=self.action_mean.device, dtype=self.action_mean.dtype)
        self._check_chunk(z, self.latent_dim, "z")
        q0 = self._q0(q0, len(z))
        tokens = torch.cat((self.decoder_q0(q0)[:, None], self.decoder_input(z)), dim=1)
        tokens = tokens + self.decoder_position
        for layer in self.decoder:
            tokens = layer(tokens)
        return self.decoder_output(self.decoder_norm(tokens[:, 1:]))

    def decode_actions(self, z, q0):
        decoded = self.decode(z, q0)
        q0 = self._q0(q0, len(decoded))
        return decoded * self.action_std + self.action_mean + q0[:, None]

    def reconstruction_terms(self, pred, target):
        """Weighted Huber at temporal orders 0/1/2, before order weighting.

        Inputs are standardized relative actions. Each term averages over batch,
        valid frames, and joints; joint weights do not change the denominator.
        """
        self._check_chunk(pred, self.hand_dim, "pred")
        if pred.shape != target.shape:
            raise ValueError("Reconstruction prediction and target shapes must match")
        terms = {}
        for name in ("position", "velocity", "acceleration"):
            penalty = F.huber_loss(pred, target, reduction="none", delta=1.0)
            terms[name] = (penalty * self.joint_weights).mean()
            pred, target = torch.diff(pred, dim=1), torch.diff(target, dim=1)
        return terms

    def reconstruction_loss(self, pred, target):
        terms = self.reconstruction_terms(pred, target)
        return terms["position"] + 0.2 * terms["velocity"] + 0.05 * terms["acceleration"]

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self

    def validate_state(self):
        if not bool(self.ready):
            raise ValueError("Temporal codec checkpoint is not ready")
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Temporal codec contains nonfinite tensors")
        if not (self.action_std > 0).all():
            raise ValueError("Temporal codec action_std must be positive")
        if not (self.joint_weights > 0).all():
            raise ValueError("Temporal codec joint_weights must be positive")


def load_codec(path):
    """Load one immutable artifact snapshot; policy state can embed all tensors."""
    raw = Path(path).expanduser().read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("_format") != "temporal_hand_codec.v1":
        raise ValueError("Expected a temporal_hand_codec.v1 checkpoint")
    codec = TemporalHandCodec(**payload["model_config"])
    state = payload["state_dict"]
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("Temporal codec state_dict must contain only tensors")
    codec.load_state_dict(state, strict=True)
    codec.validate_state()
    payload["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return codec.freeze(), payload
