"""Cross-attention Transformer for CordViP's feature correspondences."""
# Source: xuanxuanzzzii/cordvip@ad6d441abe7b0480cf79b0311db266965c1313b6, train_policy/policy/CordViP/CordViP/CordViP/model/vision/transformer.py.
# Upstream: https://github.com/r-pad/taxpose and https://github.com/WangYueFt/dcp/blob/master/model.py.

import copy

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttentionTransformer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        attention = MultiHeadedAttention(4, emb_dim)
        feed_forward = PositionwiseFeedForward(emb_dim, 1024)
        self.encoder_attention = copy.deepcopy(attention)
        self.encoder_ffn = copy.deepcopy(feed_forward)
        self.decoder_attention = copy.deepcopy(attention)
        self.cross_attention = copy.deepcopy(attention)
        self.decoder_ffn = copy.deepcopy(feed_forward)
        self.encoder_norm1 = LayerNorm(emb_dim)
        self.encoder_norm2 = LayerNorm(emb_dim)
        self.encoder_norm = LayerNorm(emb_dim)
        self.decoder_norm1 = LayerNorm(emb_dim)
        self.decoder_norm2 = LayerNorm(emb_dim)
        self.decoder_norm3 = LayerNorm(emb_dim)
        self.decoder_norm = LayerNorm(emb_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized = self.encoder_norm1(context)
        context = context + self.encoder_attention(normalized, normalized, normalized)
        context = self.encoder_norm(context + self.encoder_ffn(self.encoder_norm2(context)))
        normalized = self.decoder_norm1(query)
        query = query + self.decoder_attention(normalized, normalized, normalized)
        query = query + self.cross_attention(self.decoder_norm2(query), context, context)
        return self.decoder_norm(query + self.decoder_ffn(self.decoder_norm3(query)))


def clones(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # Sample std and eps outside the square root are part of the original model.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, features):
        super().__init__()
        self.head_dim = features // heads
        self.heads = heads
        self.linears = clones(nn.Linear(features, features), 4)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        x = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * self.head_dim)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, features, hidden_dim):
        super().__init__()
        self.w_1 = nn.Linear(features, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, features)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))
