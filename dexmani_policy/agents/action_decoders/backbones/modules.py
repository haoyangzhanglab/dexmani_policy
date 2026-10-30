"""Shared conditioning and attention layers for action backbones."""

import math

import torch
from torch import nn
from torch.nn import functional as F

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


class SelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be divisible by positive num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, length, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            self.q_norm(q),
            self.k_norm(k),
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(x))


class AdaLNOutput(nn.Module):
    def __init__(self, hidden_size, output_dim, norm):
        super().__init__()
        self.norm_final = norm
        self.linear = nn.Linear(hidden_size, output_dim)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, condition):
        shift, scale = self.adaln_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SinusoidalTimestepEmbedding(nn.Module):
    """Mish conditioning uses sin/cos with endpoint-inclusive log frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("sinusoidal embedding dimension must be positive and even")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequency_step = math.log(POS_ENCODING_BASE) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -frequency_step)
        angles = x[:, None] * frequencies[None, :]
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
    """DiT uses cos/sin, a different frequency spacing and SiLU projection."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=POS_ENCODING_BASE):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
