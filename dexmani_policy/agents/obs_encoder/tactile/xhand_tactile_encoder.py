"""Encode XHand tactile history into five finger tokens.

KineDex CNN: https://github.com/DinoMini00/KineDex_code
Optional ECA: https://github.com/BangguWu/ECANet
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class _EfficientChannelAttention(nn.Module):
    """ECA on pooled channels; equivalent to spatial ECA followed by GAP."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        weights = self.conv(x.unsqueeze(1)).squeeze(1).sigmoid()
        return x * weights


class _ArrayEncoder(nn.Module):
    """Shared single-finger CNN or flat MLP, returning 64 features."""

    def __init__(self, backend: str, use_eca: bool) -> None:
        super().__init__()
        self.is_cnn = backend == "kinedex"
        if self.is_cnn:
            self.net = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(360, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
            )
        self.attention = _EfficientChannelAttention() if use_eca else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.is_cnn:
            features = self.net(x.transpose(1, 2)).squeeze(-1)
        else:
            features = self.net(x.flatten(1))
        return self.attention(features)


class XHandTactileEncoder(nn.Module):
    """[B,T,5,3] (sim) or [B,T,5,120,3] (real) -> [B,5,D].

    Supply exactly T oldest-to-newest normalized frames. Real mode reads only
    tactile_force. ECA is opt-in for the KineDex CNN; other modes reject it.
    """

    num_tokens = 5
    frame_dim = 64

    def __init__(
        self,
        input_key: str,
        n_obs_steps: int = 2,
        out_dim: int = 128,
        array_backend: str = "kinedex",
        validate_finite: bool = True,
        use_eca: bool = False,
    ) -> None:
        super().__init__()
        if input_key not in ("contact_force", "tactile_force"):
            raise ValueError("input_key must be 'contact_force' or 'tactile_force'")
        if array_backend not in ("kinedex", "mlp"):
            raise ValueError("array_backend must be 'kinedex' or 'mlp'")
        for name, flag in (("validate_finite", validate_finite), ("use_eca", use_eca)):
            if type(flag) is not bool:
                raise TypeError(f"{name} must be a bool")
        if input_key == "contact_force" and array_backend != "kinedex":
            raise ValueError("array_backend override is only applicable to tactile_force")
        if use_eca and (input_key != "tactile_force" or array_backend != "kinedex"):
            raise ValueError("use_eca requires tactile_force with array_backend='kinedex'")
        self.input_key = input_key
        self.n_obs_steps = _positive_int(n_obs_steps, "n_obs_steps")
        self.out_dim = _positive_int(out_dim, "out_dim")
        if self.out_dim < 2:
            raise ValueError("out_dim must be at least 2 for token LayerNorm")
        self.validate_finite = validate_finite
        self.input_shape = (5, 3) if input_key == "contact_force" else (5, 120, 3)

        if input_key == "contact_force":
            self.frame_encoder = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
            )
        else:
            self.frame_encoder = _ArrayEncoder(array_backend, use_eca)
        self.history_projector = nn.Sequential(
            nn.Linear(self.n_obs_steps * self.frame_dim, 128), nn.ReLU(),
            nn.Linear(128, self.out_dim),
        )
        self.token_norm = nn.LayerNorm(self.out_dim)
        self.finger_embedding = nn.Embedding(self.num_tokens, self.out_dim)
        nn.init.normal_(self.finger_embedding.weight, std=0.02)

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)

    def _prepare_history(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor) or not x.is_floating_point():
            raise TypeError("tactile history must be a floating-point torch.Tensor")
        if x.ndim != len(self.input_shape) + 2 or tuple(x.shape[2:]) != self.input_shape:
            raise ValueError(
                f"{self.input_key}: expected [B,T,{self.input_shape}], got {tuple(x.shape)}"
            )
        if x.shape[0] <= 0 or x.shape[1] != self.n_obs_steps:
            raise ValueError(f"require B>0 and exactly T={self.n_obs_steps}")
        ref = self.finger_embedding.weight
        if x.device != ref.device:
            raise ValueError("input and encoder must be on the same device")
        x = x.to(dtype=ref.dtype)
        # This check can synchronize CUDA; disable only with upstream validation.
        if self.validate_finite and not bool(torch.isfinite(x).all()):
            raise ValueError(f"{self.input_key}: nonfinite value; missing touch is not zero")
        return x

    def encode_frames(self, x: Tensor) -> Tensor:
        """Return [B,T,5,64] before ordered history projection."""
        x = self._prepare_history(x)
        b, t = x.shape[:2]
        fingers = x.reshape(b * t * self.num_tokens, *self.input_shape[1:])
        features = self.frame_encoder(fingers)
        return features.reshape(b, t, self.num_tokens, self.frame_dim)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encode_frames(x)
        # Concatenate time within each finger, never across fingers.
        history = features.permute(0, 2, 1, 3).flatten(2)
        tokens = self.token_norm(self.history_projector(history))
        return tokens + self.finger_embedding.weight.to(tokens.dtype).unsqueeze(0)

    def forward_from_flat(self, obs: Mapping[str, Tensor]) -> Tensor:
        """Read one field after BaseAgent's normalization and B*T flattening."""
        x = obs[self.input_key]
        if not isinstance(x, Tensor):
            raise TypeError("selected observation must be a torch.Tensor")
        if x.ndim != len(self.input_shape) + 1 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"expected [B*T,{self.input_shape}], got {tuple(x.shape)}")
        if x.shape[0] <= 0 or x.shape[0] % self.n_obs_steps:
            raise ValueError("flattened batch must be positive and divisible by n_obs_steps")
        return self(x.reshape(-1, self.n_obs_steps, *self.input_shape))
