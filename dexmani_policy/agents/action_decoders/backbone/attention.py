"""Reusable attention primitives for action-decoder backbones."""

from __future__ import annotations

from torch.jit import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import use_fused_attn


class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, dim = x.shape
        _, context_len, _ = context.shape

        q = self.q(x).reshape(
            batch_size, query_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(
            batch_size, context_len, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = None
        if mask is not None:
            if mask.shape != (batch_size, context_len):
                raise ValueError(
                    "cross-attention mask must have shape "
                    f"(B, L_context), got {tuple(mask.shape)}"
                )
            attn_mask = mask.to(torch.bool).reshape(
                batch_size, 1, 1, context_len
            ).expand(-1, -1, query_len, -1)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            scores = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                scores = scores.masked_fill(
                    attn_mask.logical_not(),
                    float("-inf"),
                )
            weights = scores.softmax(dim=-1)
            if self.attn_drop.p > 0:
                weights = self.attn_drop(weights)
            out = weights @ v

        out = out.permute(0, 2, 1, 3).reshape(batch_size, query_len, dim)
        out = self.proj(out)
        return self.proj_drop(out)
