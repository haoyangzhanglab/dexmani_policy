from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP


WEIGHT_INIT_STD = 0.02


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized.to(dtype=x.dtype)


class AdaptiveRMSNorm(nn.Module):
    """Parameter-free RMSNorm modulated by a flow-timestep embedding."""

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int | None = None,
        with_gate: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        cond_dim = hidden_dim if cond_dim is None else cond_dim
        self.hidden_dim = hidden_dim
        self.with_gate = with_gate
        self.eps = eps
        num_outputs = 3 if with_gate else 2
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_outputs * hidden_dim),
        )

    def initialize_weights(self) -> None:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        modulation = self.modulation(cond)
        if self.with_gate:
            scale, shift, gate = modulation.chunk(3, dim=-1)
        else:
            scale, shift = modulation.chunk(2, dim=-1)

        x = _rms_norm(x, self.eps)
        x = x * (1 + scale[:, None]) + shift[:, None]
        if self.with_gate:
            return x, gate
        return x


class GEGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.out_proj = nn.Linear(ffn_hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        value = self.value_proj(x)
        return self.out_proj(gate * value)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm:
            q = _rms_norm(q)
            k = _rms_norm(k)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        return self.out_proj(output)


class GQACrossAttentionWithCache(nn.Module):
    """Action-to-observation GQA with a static per-generation KV cache.

    The cache retains the compact ``num_kv_heads`` representation and expands K/V heads immediately before scaled dot-product attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim)

        self._cached_k: torch.Tensor | None = None
        self._cached_v: torch.Tensor | None = None

    def _project_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_len, _ = context.shape
        k = self.k_proj(context).reshape(
            batch_size,
            context_len,
            self.num_kv_heads,
            self.head_dim,
        )
        v = self.v_proj(context).reshape(
            batch_size,
            context_len,
            self.num_kv_heads,
            self.head_dim,
        )
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.qk_norm:
            k = _rms_norm(k)
        return k, v

    def setup_kv_cache(self, context: torch.Tensor) -> None:
        self._cached_k, self._cached_v = self._project_kv(context)

    def clear_kv_cache(self) -> None:
        self._cached_k = None
        self._cached_v = None

    def _expand_kv_heads(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_kv_heads == self.num_heads:
            return k, v
        repeats = self.num_heads // self.num_kv_heads
        return (
            k.repeat_interleave(repeats, dim=1),
            v.repeat_interleave(repeats, dim=1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        q = q.permute(0, 2, 1, 3)
        if self.qk_norm:
            q = _rms_norm(q)

        if self._cached_k is None:
            k, v = self._project_kv(context)
        else:
            k, v = self._cached_k, self._cached_v
        k, v = self._expand_kv_heads(k, v)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        )
        return self.out_proj(output)


class ActionFlowTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        ffn_hidden_dim: int = 896,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if attention_type not in {"self", "cross"}:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.attention_type = attention_type
        if attention_type == "self":
            self.attention = SelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
            )
        else:
            self.attention = GQACrossAttentionWithCache(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
            )

        self.ffn = GEGLU(hidden_dim, ffn_hidden_dim)
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

    def initialize_weights(self) -> None:
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        (
            scale_attn,
            shift_attn,
            gate_attn,
            scale_ffn,
            shift_ffn,
            gate_ffn,
        ) = self.ada_mod(timestep_embedding).chunk(6, dim=-1)

        hidden = _rms_norm(x)
        hidden = hidden * (1 + scale_attn[:, None]) + shift_attn[:, None]
        if self.attention_type == "self":
            attention_output = self.attention(hidden)
        else:
            attention_output = self.attention(hidden, context)
        x = x + gate_attn[:, None] * attention_output

        hidden = _rms_norm(x)
        hidden = hidden * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * self.ffn(hidden)
        return x


class ActionFlowDiT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        ffn_hidden_dim: int = 896,
        timestep_embed_dim: int = 128,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.timestep_embedder = TimestepMLP(
            pos_emb_dim=timestep_embed_dim,
            output_dim=hidden_dim,
        )

        attention_pattern = ["self" if i % 2 == 0 else "cross" for i in range(depth)]
        self.layers = nn.ModuleList(
            [
                ActionFlowTransformerLayer(
                    hidden_dim=hidden_dim,
                    attention_type=attention_type,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                )
                for attention_type in attention_pattern
            ]
        )
        self.final_norm = AdaptiveRMSNorm(
            hidden_dim, cond_dim=hidden_dim, with_gate=False
        )
        self.action_out = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        nn.init.normal_(self.action_in.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.action_in.bias)
        nn.init.normal_(self.action_pos, std=WEIGHT_INIT_STD)

        for layer in self.layers:
            layer.initialize_weights()
        self.final_norm.initialize_weights()
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["action_pos"],
        )

    def _prepare_timestep(
        self, timestep, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device)
        else:
            timestep = timestep.to(device)
        return timestep.expand(batch_size)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.horizon or x.shape[2] != self.action_dim:
            raise ValueError(
                f"x must have shape [B, {self.horizon}, {self.action_dim}], got {tuple(x.shape)}"
            )
        if (
            context.ndim != 3
            or context.shape[0] != x.shape[0]
            or context.shape[2] != self.hidden_dim
        ):
            raise ValueError(
                f"context must have shape [B, L, {self.hidden_dim}] with B={x.shape[0]}, "
                f"got {tuple(context.shape)}"
            )

        hidden = self.action_in(x) + self.action_pos.to(dtype=x.dtype)
        timestep = self._prepare_timestep(timestep, x.shape[0], x.device)
        timestep_embedding = self.timestep_embedder(timestep)

        for layer in self.layers:
            hidden = layer(hidden, timestep_embedding, context)

        hidden = self.final_norm(hidden, timestep_embedding)
        return self.action_out(hidden)

    def setup_kv_cache(self, context: torch.Tensor) -> None:
        for layer in self.layers:
            if layer.attention_type == "cross":
                layer.attention.setup_kv_cache(context)

    def clear_kv_cache(self) -> None:
        for layer in self.layers:
            if layer.attention_type == "cross":
                layer.attention.clear_kv_cache()
