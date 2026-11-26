"""Shared initialization and conditioning layers for action backbones and pretraining."""

import math

import torch
from torch import nn

POS_ENCODING_BASE = 10000.0
WEIGHT_INIT_STD = 0.02


def batch_timesteps(timestep, reference: torch.Tensor, *, dtype=None) -> torch.Tensor:
    return (
        torch.as_tensor(timestep, device=reference.device, dtype=dtype)
        .reshape(-1)
        .expand(reference.shape[0])
    )


def add_temporal_embedding(tokens, frame_embedding):
    """Frame-major observation tokens share one temporal embedding per frame."""
    if tokens.shape[1] == 0:
        raise ValueError("context must contain observation tokens")
    tokens = tokens.unflatten(1, (frame_embedding.shape[1], -1))
    return (tokens + frame_embedding.unsqueeze(2)).flatten(1, 2)


def initialize_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SinusoidalTimestepEmbedding(nn.Module):
    """Discrete diffusion sin/cos features with endpoint-inclusive log frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("sinusoidal embedding dimension must be positive and even")
        half_dim = dim // 2
        frequency_step = math.log(POS_ENCODING_BASE) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -frequency_step)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x[:, None] * self.frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class FlowTimestepEmbedding(nn.Module):
    """Continuous sin/cos features with OpenPI's 0.004–4.0 periods."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("flow embedding dimension must be positive and even")
        self.half_dim = dim // 2
        self.register_buffer("frequencies", self._frequencies(), persistent=False)

    def _frequencies(self, device=None):
        fraction = torch.linspace(0, 1, self.half_dim, device=device, dtype=torch.float32)
        period = 0.004 * (4.0 / 0.004) ** fraction
        return 2 * math.pi / period

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Rebuild after dtype conversion: casting rounded BF16 frequencies back
        # to FP32 would retain phase errors at the shortest periods.
        self.frequencies = self._frequencies(self.frequencies.device)
        return self

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = timestep.float()[:, None] * self.frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, pos_emb_dim: int, output_dim: int, *, embedding: nn.Module | None = None):
        super().__init__()
        hidden_dim = pos_emb_dim * 4
        self.net = nn.Sequential(
            SinusoidalTimestepEmbedding(pos_emb_dim) if embedding is None else embedding,
            nn.Linear(pos_emb_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        features = self.net[0](t).to(self.net[1].weight.dtype)
        return self.net[3](self.net[2](self.net[1](features)))


class DiTTimestepEmbedding(nn.Module):
    """Cos/sin timestep features with SiLU projection for AFRO dynamics."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(POS_ENCODING_BASE) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("frequencies", freqs, persistent=False)

    def forward(self, t):
        args = t[:, None].float() * self.frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)
