"""Standalone SmolVLA-style action expert for the repository's flow decoders.

Reference: huggingface/lerobot at 45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6,
src/lerobot/policies/smolvla/{modeling_smolvla,smolvlm_with_expert}.py.
Each layer adapts fixed observation features in place of the original VLM's
layerwise prefix K/V. This is a from-scratch backbone, not checkpoint-compatible
with SmolVLA. Time and velocity follow the calling decoder's convention.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_int

from .modules import WEIGHT_INIT_STD, add_temporal_embedding, batch_timesteps


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.register_buffer("indices", torch.arange(head_dim // 2), persistent=False)

    def forward(self, x, positions):
        # SmolVLA rotates the two halves, rather than adjacent even/odd channels.
        frequencies = 10000.0 ** (-2.0 * self.indices.float() / self.head_dim)
        angles = positions.float()[:, None] * frequencies[None]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(x.dtype)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, min_period=4e-3, max_period=4.0):
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer("indices", torch.arange(dim // 2), persistent=False)

    def forward(self, timestep):
        # Integer indices retain precision even when the model is cast to BF16.
        fraction = self.indices.double() / max(self.dim // 2 - 1, 1)
        periods = self.min_period * (self.max_period / self.min_period) ** fraction
        angles = timestep.double()[:, None] * (2 * math.pi / periods)[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1).float()


class GatedMLP(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertBlock(nn.Module):
    def __init__(
        self, dim, n_head, n_kv_head, head_dim, intermediate_dim,
        cross_attention, p_drop_attn, qkv_bias,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.cross_attention = cross_attention
        self.p_drop_attn = p_drop_attn
        kv_dim = n_kv_head * head_dim

        self.input_norm = RMSNorm(dim)
        self.context_norm = RMSNorm(dim)
        self.context_k_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.context_v_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, n_head * head_dim, bias=qkv_bias)
        # Cross-attention adapts the already rotated prefix keys, as in SmolVLA.
        kv_input_dim = kv_dim if cross_attention else dim
        self.k_proj = nn.Linear(kv_input_dim, kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_input_dim, kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_head * head_dim, dim, bias=False)
        self.post_attention_norm = RMSNorm(dim)
        self.mlp = GatedMLP(dim, intermediate_dim)
        self.rope = RotaryEmbedding(head_dim)

    def split_heads(self, x, n_head):
        return x.unflatten(-1, (n_head, self.head_dim)).transpose(1, 2)

    def prepare_context_kv(self, context):
        context = self.context_norm(context)
        key = self.split_heads(self.context_k_proj(context), self.n_kv_head)
        value = self.split_heads(self.context_v_proj(context), self.n_kv_head)
        key = self.rope(key, torch.arange(context.shape[1], device=context.device))
        if self.cross_attention:
            key = self.k_proj(key.transpose(1, 2).flatten(2))
            value = self.v_proj(value.transpose(1, 2).flatten(2))
            key = self.split_heads(key, self.n_kv_head)
            value = self.split_heads(value, self.n_kv_head)
        return key, value

    def forward(self, x, context, *, context_kv=None):
        key, value = self.prepare_context_kv(context) if context_kv is None else context_kv
        prefix_length = key.shape[2]
        hidden = self.input_norm(x)
        positions = torch.arange(x.shape[1], device=x.device)
        query = self.split_heads(self.q_proj(hidden), self.n_head)
        if self.cross_attention:
            query = self.rope(query, positions)
            attention_mask = None
        else:
            query = self.rope(query, positions + prefix_length)
            action_key = self.split_heads(self.k_proj(hidden), self.n_kv_head)
            action_key = self.rope(action_key, positions + prefix_length)
            action_value = self.split_heads(self.v_proj(hidden), self.n_kv_head)
            key = torch.cat((key, action_key), dim=2)
            value = torch.cat((value, action_value), dim=2)
            # SDPA's non-square is_causal mask would hide part of the prefix.
            attention_mask = torch.arange(key.shape[2], device=x.device)[None] <= (
                prefix_length + positions[:, None]
            )

        groups = self.n_head // self.n_kv_head
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = F.scaled_dot_product_attention(
                query.float(), key.float(), value.float(), attn_mask=attention_mask,
                dropout_p=self.p_drop_attn if self.training else 0.0,
            ).to(query.dtype)
        x = x + self.o_proj(output.transpose(1, 2).flatten(2))
        return x + self.mlp(self.post_attention_norm(x))


class DiT(nn.Module):
    """Action/time fusion followed by alternating joint SA and prefix-only CA.

    ``x`` is [B, horizon, action_dim]; ``context`` is frame-major
    [B, n_obs_steps * tokens_per_frame, obs_token_dim], without padding.
    ``forward`` returns a velocity of the same shape as ``x``. A cache belongs
    to one fixed context and is reused unchanged across flow integration steps.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        timestep_embed_dim: int | None = None,
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        n_kv_head: int = 2,
        head_dim: int | None = None,
        self_attn_every_n_layers: int = 2,
    ) -> None:
        super().__init__()
        for name, value in (
            ("horizon", horizon), ("action_dim", action_dim), ("n_obs_steps", n_obs_steps),
            ("obs_token_dim", obs_token_dim), ("n_layers", n_layers), ("hidden_dim", hidden_dim),
            ("n_head", n_head), ("n_kv_head", n_kv_head),
            ("self_attn_every_n_layers", self_attn_every_n_layers),
        ):
            positive_int(value, name)
        if n_head % n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if head_dim is None:
            if hidden_dim % n_head:
                raise ValueError("hidden_dim must be divisible by n_head when head_dim is omitted")
            head_dim = hidden_dim // n_head
        if positive_int(head_dim, "head_dim") % 2:
            raise ValueError("head_dim must be even for RoPE")
        timestep_embed_dim = hidden_dim if timestep_embed_dim is None else timestep_embed_dim
        if positive_int(timestep_embed_dim, "timestep_embed_dim") % 2:
            raise ValueError("timestep_embed_dim must be even")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")
        if not 0 <= p_drop_attn < 1:
            raise ValueError("p_drop_attn must be in [0, 1)")
        # Match SmolVLA's gated-MLP parameter budget and 256-wide rounding.
        intermediate_dim = 256 * math.ceil(int(int(2 * hidden_dim / 3) * mlp_ratio) / 256)
        if intermediate_dim == 0:
            raise ValueError("hidden_dim and mlp_ratio produce an empty MLP")

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.obs_token_dim = obs_token_dim
        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.timestep_embedder = TimeEmbedding(timestep_embed_dim)
        self.action_time_mlp = nn.Sequential(
            nn.Linear(hidden_dim + timestep_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dit_blocks = nn.ModuleList(
            ExpertBlock(
                hidden_dim, n_head, n_kv_head, head_dim, intermediate_dim,
                cross_attention=index % self_attn_every_n_layers != 0,
                p_drop_attn=p_drop_attn, qkv_bias=qkv_bias,
            )
            for index in range(n_layers)
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=WEIGHT_INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed_context(self, context):
        tokens = self.context_embedder(context)
        return add_temporal_embedding(tokens, self.context_frame_pos_embed.to(tokens.dtype))

    def prepare_context_kv(self, context):
        if self.training:
            return None
        context = self.embed_context(context)
        return tuple(block.prepare_context_kv(context) for block in self.dit_blocks)

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor, *, context_kv=None):
        if x.ndim != 3 or x.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("actions must have shape [B, horizon, action_dim]")
        hidden = self._forward_hidden(x, timestep, context, context_kv=context_kv)
        return self.output_proj(self.final_norm(hidden))

    def forward_prefix_features(self, x: torch.Tensor, timestep, context: torch.Tensor):
        if x.ndim != 3 or not 1 <= x.shape[1] <= self.horizon or x.shape[2] != self.action_dim:
            raise ValueError("prefix actions must have shape [B, L, action_dim], 1 <= L <= horizon")
        return self._forward_hidden(x, timestep, context)

    def _forward_hidden(self, x, timestep, context, *, context_kv=None):
        if context.shape[0] != x.shape[0]:
            raise ValueError("actions and context must have the same batch size")
        context_c = self.embed_context(context) if context_kv is None else None
        timestep = batch_timesteps(timestep, x, dtype=torch.float32)
        hidden = self.input_embedder(x)
        time = self.timestep_embedder(timestep).to(hidden.dtype)
        hidden = self.action_time_mlp(torch.cat((
            hidden, time[:, None].expand(-1, x.shape[1], -1),
        ), dim=-1))
        for index, block in enumerate(self.dit_blocks):
            hidden = block(
                hidden, context_c, context_kv=None if context_kv is None else context_kv[index],
            )
        return hidden
