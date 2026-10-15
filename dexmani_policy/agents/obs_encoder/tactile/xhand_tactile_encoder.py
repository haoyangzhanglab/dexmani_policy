"""Independent, per-frame XHand tactile encoders.

The 3D force path follows the state-MLP approach of FTP-1 / ReDex.
The 120-taxel path uses the per-finger 1D CNN from KineDex:
https://github.com/DinoMini00/KineDex_code/blob/main/diffusion_policy/model/vision/multi_image_obs_encoder.py

Both paths share their parameters across fingers and produce five 64D tokens.
Leading [B] or [B,T] dimensions are processed independently: no temporal
modeling, cross-finger interaction, or multimodal fusion is performed here.
"""

import torch
from torch import nn


def _prepare_input(
    x: torch.Tensor,
    tactile_valid: torch.Tensor | None,
    input_shape: tuple[int, ...],
    input_key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate a normalized per-frame input; sanitize masked fingers before encoding."""
    n = len(input_shape)
    if x.ndim not in (n + 1, n + 2) or tuple(x.shape[-n:]) != input_shape or 0 in x.shape[:-n]:
        raise ValueError(f"{input_key}: expected nonempty [B,{input_shape}] or [B,T,{input_shape}], got {tuple(x.shape)}")
    if tactile_valid is not None:
        mask_shape = (*x.shape[:-n], 5)
        if tactile_valid.shape != mask_shape or tactile_valid.dtype != torch.bool:
            raise ValueError(f"tactile_valid must be bool {mask_shape}")
        x = torch.where(tactile_valid.reshape(*mask_shape, *((1,) * (n - 1))), x, 0.0)
    x = x.to(dtype=dtype)
    if not torch.isfinite(x).all():
        raise ValueError(f"{input_key}: nonfinite values in valid observations")
    return x


class XHandForceEncoder(nn.Module):
    """Normalized [B,(T,)5,3] -> [B,(T,)5,64] with one shared 3-layer MLP."""

    num_tokens = 5
    frame_dim = 64
    input_key = "finger_tactile_force"
    input_shape = (5, 3)

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, x: torch.Tensor, tactile_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = _prepare_input(x, tactile_valid, self.input_shape, self.input_key, self.net[0].weight.dtype)
        features = self.net(x)
        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)


class XHandTaxelEncoder(nn.Module):
    """Normalized [B,(T,)5,120,3] -> [B,(T,)5,64] with KineDex 1D CNN."""

    num_tokens = 5
    frame_dim = 64
    input_key = "taxel_tactile_force"
    input_shape = (5, 120, 3)

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor, tactile_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = _prepare_input(x, tactile_valid, self.input_shape, self.input_key, self.net[0].weight.dtype)
        prefix = x.shape[:-3]
        fingers = x.reshape(-1, 120, 3).transpose(1, 2)
        features = self.net(fingers).squeeze(-1).reshape(*prefix, self.num_tokens, self.frame_dim)
        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)
