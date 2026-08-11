"""Geometric Transformer stack — faithful port of OPFA's
``geotransformer/modules/geotransformer/`` + ``transformer/`` + ``layers/``.

Consolidates 6 OPFA files:
  - ``pe_transformer.py``  → PEMultiHeadAttention, PEAttentionLayer, PETransformerLayer
  - ``output_layer.py``    → AttentionOutput
  - ``layers/factory.py``  → build_dropout_layer, build_act_layer
  - ``conditional_transformer.py`` → SinglePointCloudPEConditionalTransformer
  - ``geotransformer.py``  → GeometricTransformer

Key OPFA design:
  PE-MHA: Q = proj_q(x) + proj_p(embed), K = proj_k(x) + proj_p(embed)
  AttentionOutput: expand(d→2d) → act → squeeze(2d→d)  (NOT standard MLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# =========================================================================
# Dropout factory (port of layers/factory.py:build_dropout_layer)
# =========================================================================

def build_dropout_layer(p: float | None) -> nn.Module:
    """Return ``nn.Dropout(p)`` or ``nn.Identity()`` if p is None/0."""
    if p is None or p == 0:
        return nn.Identity()
    return nn.Dropout(p=p)


def build_act_layer(name: str) -> nn.Module:
    """Return activation module by name."""
    return {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "GELU": nn.GELU,
        "Identity": nn.Identity,
    }[name]()


# =========================================================================
# AttentionOutput (port of transformer/output_layer.py — identical)
# =========================================================================

class AttentionOutput(nn.Module):
    """OPFA-style FFN: expand→act→squeeze with pre-norm residual.

    .. math::
        h = LayerNorm(x + Dropout(Squeeze(Act(Expand(x)))))

    Note: this is NOT the standard Transformer FFN (which uses two Linear
    layers with the same hidden dim).  OPFA doubles the dimension.
    """

    def __init__(self, d_model: int, dropout: float | None = None, activation_fn: str = "ReLU"):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.norm(input_states + hidden_states)


# =========================================================================
# PEMultiHeadAttention (port of transformer/pe_transformer.py — identical)
# =========================================================================

class PEMultiHeadAttention(nn.Module):
    """Position-Encoded Multi-Head Attention.

    Key OPFA detail: positional embedding is ADDED to the Q and K projections,
    NOT added to the input tokens.

    .. math::
        Q = proj_q(x) + proj_p(emb)
        K = proj_k(x) + proj_p(emb)
        V = proj_v(x)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be a multiple of num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self,
        input_q: torch.Tensor,  # (B, N, C)
        input_k: torch.Tensor,  # (B, M, C)
        input_v: torch.Tensor,  # (B, M, C)
        embed_q: torch.Tensor,  # (B, N, C)
        embed_k: torch.Tensor,  # (B, M, C)
        key_masks: torch.Tensor | None = None,  # (B, M) bool
        attention_factors: torch.Tensor | None = None,  # (B, N, M)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = rearrange(self.proj_q(input_q) + self.proj_p(embed_q), "b n (h c) -> b h n c", h=self.num_heads)
        k = rearrange(self.proj_k(input_k) + self.proj_p(embed_k), "b m (h c) -> b h m c", h=self.num_heads)
        v = rearrange(self.proj_v(input_v), "b m (h c) -> b h m c", h=self.num_heads)

        attention_scores = torch.einsum("bhnc,bhmc->bhnm", q, k) / self.d_model_per_head**0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, "b h n c -> b n (h c)")

        return hidden_states, attention_scores


# =========================================================================
# PEAttentionLayer (port of transformer/pe_transformer.py — identical)
# =========================================================================

class PEAttentionLayer(nn.Module):
    """PE-MHA → Linear → Dropout → LayerNorm(+residual)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float | None = None):
        super().__init__()
        self.attention = PEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states: torch.Tensor,  # (B, N, C)
        memory_states: torch.Tensor,  # (B, M, C)
        input_embeddings: torch.Tensor,  # (B, N, C)
        memory_embeddings: torch.Tensor,  # (B, M, C)
        memory_masks: torch.Tensor | None = None,
        attention_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            input_embeddings,
            memory_embeddings,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


# =========================================================================
# PETransformerLayer (port of transformer/pe_transformer.py — identical)
# =========================================================================

class PETransformerLayer(nn.Module):
    """PEAttentionLayer + AttentionOutput (OPFA-style FFN)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float | None = None, activation_fn: str = "ReLU"):
        super().__init__()
        self.attention = PEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states: torch.Tensor,
        memory_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        memory_embeddings: torch.Tensor,
        memory_masks: torch.Tensor | None = None,
        attention_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            input_embeddings,
            memory_embeddings,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


# =========================================================================
# SinglePointCloudPEConditionalTransformer (port of conditional_transformer.py)
# =========================================================================

class SinglePointCloudPEConditionalTransformer(nn.Module):
    """PE-conditional transformer for a single point cloud (no cross-attn).

    For ``blocks=['self','self','self']`` (OPFA default), creates 3
    ``PETransformerLayer`` layers — all self-attention with PE.
    """

    def __init__(
        self,
        blocks: list[str],
        d_model: int,
        num_heads: int,
        dropout: float | None = None,
        activation_fn: str = "ReLU",
        return_attention_scores: bool = False,
    ):
        super().__init__()
        self.blocks = blocks
        layers = []
        for block in blocks:
            if block not in ("self", "cross"):
                raise ValueError(f'Unsupported block type "{block}".')
            if block == "self":
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                # Standard TransformerLayer for cross-attn (not used in OPFA config,
                # but kept for API compatibility — defined below in this module).
                layers.append(_TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(
        self,
        feats0: torch.Tensor,  # (B, N, C)
        embeddings0: torch.Tensor,  # (B, N, C)
        masks0: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == "self":
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
            else:
                # Cross-attention path (unused for OPFA's ['self','self','self'])
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append(scores0)
        if self.return_attention_scores:
            return feats0, attention_scores
        return feats0


# =========================================================================
# TransformerLayer (vanilla, for cross-attn path — simplified port)
# =========================================================================

class MultiHeadAttention(nn.Module):
    """Standard MHA (no PE)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be a multiple of num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.dropout = build_dropout_layer(dropout)

    def forward(
        self,
        input_q: torch.Tensor,
        input_k: torch.Tensor,
        input_v: torch.Tensor,
        key_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = rearrange(self.proj_q(input_q), "b n (h c) -> b h n c", h=self.num_heads)
        k = rearrange(self.proj_k(input_k), "b m (h c) -> b h m c", h=self.num_heads)
        v = rearrange(self.proj_v(input_v), "b m (h c) -> b h m c", h=self.num_heads)

        attention_scores = torch.einsum("bhnc,bhmc->bhnm", q, k) / self.d_model_per_head**0.5
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, "b h n c -> b n (h c)")
        return hidden_states, attention_scores


class AttentionLayer(nn.Module):
    """MHA → Linear → Dropout → LayerNorm(+residual)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float | None = None):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states: torch.Tensor,
        memory_states: torch.Tensor,
        memory_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, attention_scores = self.attention(input_states, memory_states, memory_states, key_masks=memory_masks)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class _TransformerLayer(nn.Module):
    """See TransformerLayer — same implementation, module-private name."""
    """AttentionLayer + AttentionOutput (OPFA-style FFN)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float | None = None, activation_fn: str = "ReLU"):
        super().__init__()
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states: torch.Tensor,
        memory_states: torch.Tensor,
        memory_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, attention_scores = self.attention(input_states, memory_states, memory_masks=memory_masks)
        output_states = self.output(hidden_states)
        return output_states, attention_scores


# =========================================================================
# GeometricTransformer (port of geotransformer/geotransformer.py — identical)
# =========================================================================

class GeometricTransformer(nn.Module):
    """OPFA Geometric Transformer — PE-based attention on point clouds.

    Architecture (identical to OPFA):
      1. ``xyz_embedding``: Linear(3, input_dim) — spatial PE from coordinates.
      2. PE fusion: ``pos_emb = link_emb + xyz_embedding(points)``.
      3. ``in_proj`` / ``in_proj_pos_emb``: project features & PE to hidden_dim.
      4. ``SinglePointCloudPEConditionalTransformer``: 3× PETransformerLayer.
      5. ``out_proj``: project back to output_dim.

    Note: ``GeometricStructureEmbedding`` is intentionally NOT included —
    it is defined but commented out in OPFA's forward pass and was never used.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        blocks: list[str],
        dropout: float | None = None,
        activation_fn: str = "ReLU",
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.in_proj_pos_emb = nn.Linear(input_dim, hidden_dim)
        self.transformer = SinglePointCloudPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.xyz_embedding = nn.Linear(3, input_dim)

    def forward(
        self,
        points_c: torch.Tensor,  # (B, N, 3)
        feats_c: torch.Tensor,  # (B, N, input_dim)
        link_emb_c: torch.Tensor,  # (B, N, input_dim)
    ) -> torch.Tensor:
        """GeometricTransformer forward — identical to OPFA.

        Returns:
            ``(B, N, output_dim)``
        """
        xyz_pos_embedding = self.xyz_embedding(points_c)  # (B, N, input_dim)
        pos_emb = link_emb_c + xyz_pos_embedding
        feats = self.in_proj(feats_c)
        pos_emb = self.in_proj_pos_emb(pos_emb)
        feats = self.transformer(feats, pos_emb)
        feats = self.out_proj(feats)
        return feats
