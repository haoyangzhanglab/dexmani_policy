"""Encode XHand tactile history into five finger tokens.

KineDex CNN: https://github.com/DinoMini00/KineDex_code
Optional ECA: https://github.com/BangguWu/ECANet
"""
import torch
import torch.nn as nn


class EfficientChannelAttention(nn.Module):
    """ECA on pooled channels."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.conv(features.unsqueeze(1)).squeeze(1).sigmoid()
        return features * weights


class ArrayEncoder(nn.Module):
    """Single-finger array to 64 features."""

    def __init__(self, backend: str, use_eca: bool) -> None:
        super().__init__()
        self.is_cnn = backend == "kinedex"
        if self.is_cnn:
            self.net = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(360, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
        self.attention = EfficientChannelAttention() if use_eca else nn.Identity()

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        if self.is_cnn:
            features = self.net(tactile.transpose(1, 2)).squeeze(-1)
        else:
            features = self.net(tactile.flatten(1))
        return self.attention(features)


class XHandTactileEncoder(nn.Module):
    """[B,T,5,3] (sim) or [B,T,5,120,3] (real) -> [B,5,D].

    Normalized history, oldest first; fingers: thumb, index, middle, ring, pinky.
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
        if input_key == "contact_force" and array_backend != "kinedex":
            raise ValueError("array_backend override is only applicable to tactile_force")
        if use_eca and (input_key != "tactile_force" or array_backend != "kinedex"):
            raise ValueError("use_eca requires tactile_force with array_backend='kinedex'")
        if n_obs_steps < 1 or out_dim < 2:
            raise ValueError("require n_obs_steps >= 1 and out_dim >= 2")

        self.input_key = input_key
        self.n_obs_steps = n_obs_steps
        self.output_channels = out_dim
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
            self.frame_encoder = ArrayEncoder(array_backend, use_eca)
        self.history_projector = nn.Sequential(
            nn.Linear(self.n_obs_steps * self.frame_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_channels),
        )
        self.token_norm = nn.LayerNorm(self.output_channels)
        self.finger_embedding = nn.Embedding(self.num_tokens, self.output_channels)
        nn.init.normal_(self.finger_embedding.weight, std=0.02)

    def _prepare_history(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("tactile history must be floating point")
        expected = (self.n_obs_steps, *self.input_shape)
        if x.ndim != len(expected) + 1 or x.shape[0] == 0 or x.shape[1:] != expected:
            raise ValueError(f"expected [B>0, {expected}], got {tuple(x.shape)}")
        x = x.to(dtype=self.finger_embedding.weight.dtype)
        if self.validate_finite and not bool(torch.isfinite(x).all()):
            raise ValueError(f"{self.input_key}: nonfinite values")
        return x

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B,T,5,64]."""
        x = self._prepare_history(x)
        batch_size, time_steps = x.shape[:2]
        finger_input = x.reshape(batch_size * time_steps * self.num_tokens, *self.input_shape[1:])
        frame_features = self.frame_encoder(finger_input)
        return frame_features.reshape(batch_size, time_steps, self.num_tokens, self.frame_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame_features = self.encode_frames(x)
        history = frame_features.permute(0, 2, 1, 3).flatten(2)
        tokens = self.token_norm(self.history_projector(history))
        return tokens + self.finger_embedding.weight.to(tokens.dtype).unsqueeze(0)

    @property
    def out_dim(self) -> int:
        return self.output_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_tokens, self.output_channels)

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)


def example() -> None:
    batch_size, n_obs_steps = 2, 2
    # Synthetic normalized inputs.
    contact_force = torch.randn(batch_size, n_obs_steps, 5, 3)
    tactile_force = torch.randn(batch_size, n_obs_steps, 5, 120, 3)

    for input_key, history, use_eca in (
        ("contact_force", contact_force, False),
        ("tactile_force", tactile_force, False),
        ("tactile_force", tactile_force, True),
    ):
        print(f"=== XHandTactileEncoder: {input_key}, use_eca={use_eca} ===")
        encoder = XHandTactileEncoder(
            input_key=input_key,
            n_obs_steps=n_obs_steps,
            out_dim=128,
            use_eca=use_eca,
        ).eval()
        with torch.no_grad():
            tokens = encoder(history)
        print("input:", tuple(history.shape))
        print("finger_tokens:", tuple(tokens.shape))
        print("out_dim:", encoder.out_dim)
        print("out_shape:", encoder.out_shape)


if __name__ == "__main__":
    example()
