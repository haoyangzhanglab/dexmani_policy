"""Cross-attention Transformer for CordViP's feature correspondences.

Commit: ad6d441abe7b0480cf79b0311db266965c1313b6
Source: train_policy/policy/CordViP/CordViP/CordViP/model/vision/transformer.py
Upstream attribution: https://github.com/r-pad/taxpose and
https://github.com/WangYueFt/dcp/blob/master/model.py.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttentionTransformer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        attention = MultiHeadedAttention(4, emb_dim)
        feed_forward = PositionwiseFeedForward(emb_dim, 1024)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(emb_dim, copy.deepcopy(attention), copy.deepcopy(feed_forward))),
            Decoder(DecoderLayer(emb_dim, copy.deepcopy(attention), copy.deepcopy(attention),
                                 copy.deepcopy(feed_forward))),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.model(context, query)


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


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, context, query):
        return self.decoder(query, self.encoder(context))


class Decoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = clones(layer, 1)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 3)

    def forward(self, x, memory):
        x = self.sublayer[0](x, lambda value: self.self_attn(value, value, value))
        x = self.sublayer[1](x, lambda value: self.src_attn(value, memory, memory))
        return self.sublayer[2](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class Encoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = clones(layer, 1)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x, lambda value: self.self_attn(value, value, value))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, features):
        super().__init__()
        self.head_dim = features // heads
        self.heads = heads
        self.linears = clones(nn.Linear(features, features), 4)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2).contiguous()
            for linear, x in zip(self.linears, (query, key, value))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(self.head_dim)
        x = torch.matmul(F.softmax(scores, dim=-1), value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.head_dim)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, features, hidden_dim):
        super().__init__()
        self.w_1 = nn.Linear(features, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, features)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))
