"""Encode XHand observations without pooling across fingers or time."""

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.tactile.xhand_tactile_encoder import ArrayEncoder


class XHandFrameEncoder(nn.Module):
    """Normalized [B,T,5,3] or [B,T,5,120,3] -> [B,T,5,64].

    Fingers are thumb, index, middle, ring, pinky. A False tactile_valid entry
    means unavailable, not zero contact; mask it before encoding to isolate NaNs.
    ArrayEncoder retains the existing KineDex index-order convolution assumption.
    No history projector or finger embedding is needed before aligned fusion.
    """

    num_tokens = 5
    frame_dim = 64

    def __init__(self, input_key: str = "contact_force", validate_finite: bool = True) -> None:
        super().__init__()
        if input_key not in ("contact_force", "tactile_force"):
            raise ValueError("input_key must be 'contact_force' or 'tactile_force'")
        self.input_key = input_key
        self.validate_finite = validate_finite
        self.input_shape = (5, 3) if input_key == "contact_force" else (5, 120, 3)
        if input_key == "contact_force":
            self.frame_encoder = nn.Sequential(
                nn.Linear(3, self.frame_dim),
                nn.ReLU(),
                nn.Linear(self.frame_dim, self.frame_dim),
                nn.ReLU(),
            )
        else:
            self.frame_encoder = ArrayEncoder(backend="kinedex", use_eca=False)

    def forward(
        self,
        x: torch.Tensor,
        tactile_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != len(self.input_shape) + 2 or x.shape[2:] != self.input_shape:
            raise ValueError(f"expected [B,T,{self.input_shape}], got {tuple(x.shape)}")
        if min(x.shape[:2]) < 1:
            raise ValueError("batch size and time steps must be positive")
        if not x.is_floating_point():
            raise TypeError("tactile observations must be floating point")
        if tactile_valid is not None:
            if (
                tactile_valid.shape != x.shape[:3]
                or tactile_valid.dtype != torch.bool
                or tactile_valid.device != x.device
            ):
                raise ValueError("tactile_valid must be bool [B,T,5] on the input device")
            mask = tactile_valid.reshape(*tactile_valid.shape, *((1,) * (x.ndim - 3)))
            x = torch.where(mask, x, 0.0)
        x = x.to(dtype=next(self.frame_encoder.parameters()).dtype)
        if self.validate_finite and not bool(torch.isfinite(x).all()):
            raise ValueError(f"{self.input_key}: nonfinite values in valid observations")

        batch_size, time_steps = x.shape[:2]
        features = self.frame_encoder(x.reshape(-1, *self.input_shape[1:]))
        features = features.reshape(batch_size, time_steps, self.num_tokens, self.frame_dim)
        if tactile_valid is not None:
            # Encoder biases must not turn a missing reading into an observation.
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)
