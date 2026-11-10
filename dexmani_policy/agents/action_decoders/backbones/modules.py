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


def approximate_gelu():
    return nn.GELU(approximate="tanh")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalTimestepEmbedding(nn.Module):
    """Mish conditioning uses sin/cos with endpoint-inclusive log frequencies."""

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


class TimestepMLP(nn.Module):
    def __init__(self, pos_emb_dim: int, output_dim: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimestepEmbedding(pos_emb_dim),
            nn.Linear(pos_emb_dim, pos_emb_dim * expansion),
            nn.Mish(),
            nn.Linear(pos_emb_dim * expansion, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


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
