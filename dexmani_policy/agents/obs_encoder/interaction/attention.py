"""交互 token 的 self-attention 与带米制相对几何的 cross-attention。"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class InteractionSelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(token)
        update, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        token = token + update
        return token + self.ffn(self.norm2(token))


class GeometryCrossAttention(nn.Module):
    """按米制边特征和距离先验检索；null 权重只是模型读出，未经置信度校准。"""

    def __init__(
        self,
        channels: int,
        sigmas: Sequence[float],
        metric_scale: float,
        proximity_scales: tuple[float, float],
        radius_multiple: float | None,
    ):
        super().__init__()
        self.channels = channels
        self.heads = len(sigmas)
        self.head_dim = channels // self.heads
        self.metric_scale = metric_scale
        self.radius_multiple = radius_multiple
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.register_buffer(
            "proximity_scales", torch.tensor(proximity_scales, dtype=torch.float32)
        )
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.edge = nn.Sequential(nn.Linear(6, channels), nn.GELU())
        self.edge_bias = nn.Linear(channels, self.heads)
        self.edge_value = nn.Linear(channels, channels)
        self.null_key = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.out = nn.Linear(channels, channels)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        relative: torch.Tensor,
        distance: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, queries, channels = query.shape
        patches = memory.shape[1]
        q = self.q(self.query_norm(query)).reshape(
            batch, queries, self.heads, self.head_dim
        )
        memory = self.memory_norm(memory)
        k = self.k(memory).reshape(batch, patches, self.heads, self.head_dim)
        v = self.v(memory).reshape(batch, patches, self.heads, self.head_dim)
        d = torch.where(valid, distance.float(), 0.0)
        delta = torch.where(valid[..., None], relative.float(), 0.0)
        proximity = torch.exp(-0.5 * (d[..., None] / self.proximity_scales).square())
        geometry = torch.cat(
            (delta / self.metric_scale, d[..., None] / self.metric_scale, proximity), -1
        )
        edge = self.edge(geometry.to(memory.dtype))
        edge_value = self.edge_value(edge).reshape(
            batch, queries, patches, self.heads, self.head_dim
        )

        edge_bias = self.edge_bias(edge).permute(0, 1, 3, 2)
        # AMP 下 logits 和 softmax 仍使用 FP32。
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bqhd,bmhd->bqhm", q.float(), k.float())
            logits = logits / math.sqrt(self.head_dim) + edge_bias.float()
            ratios = d[:, :, None, :] / self.sigmas.float()[None, None, :, None]
            logits = logits - 0.5 * ratios.square()
            allowed = valid[:, :, None, :].expand(-1, -1, self.heads, -1)
            if self.radius_multiple is not None:
                allowed = allowed & (ratios <= self.radius_multiple)
            logits = logits.masked_fill(~allowed, float("-inf"))
            null_logit = torch.einsum("bqhd,hd->bqh", q.float(), self.null_key.float())
            null_logit = null_logit / math.sqrt(self.head_dim)
            attention = torch.cat((logits, null_logit[..., None]), -1).softmax(-1)
        real_weight = attention[..., :-1].to(v.dtype)
        semantic = torch.einsum("bqhm,bmhd->bqhd", real_weight, v)
        relational = torch.einsum("bqhm,bqmhd->bqhd", real_weight, edge_value)
        null = attention[..., -1:].to(v.dtype) * self.null_value[None, None].to(v.dtype)
        output = self.out(
            (semantic + relational + null).reshape(batch, queries, channels)
        )
        return output, attention, allowed.any(-1)
