"""Cross-attention with Base-frame relative position bias and values."""

import torch
from torch import nn
from torch.nn import functional as F


class Relative3DCrossAttention(nn.Module):
    embed_dim = 192
    num_heads = 4
    head_dim = embed_dim // num_heads
    geometry_scale = 0.10

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.memory_norm = nn.LayerNorm(self.embed_dim)
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv = nn.Linear(self.embed_dim, 2 * self.embed_dim)
        self.position_bias = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, self.num_heads)
        )
        self.out = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.relative_value = nn.Linear(3, self.embed_dim, bias=False)
        nn.init.zeros_(self.relative_value.weight)

    def forward(self, body_tokens, scene_tokens, anchors_base, scene_xyz_base):
        batch_size, num_queries, _ = body_tokens.shape
        num_scene_tokens = scene_tokens.shape[1]

        q = self.q(self.query_norm(body_tokens)).reshape(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.kv(self.memory_norm(scene_tokens)).reshape(
            batch_size, num_scene_tokens, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)

        with torch.autocast(device_type=anchors_base.device.type, enabled=False):
            anchors = anchors_base.float()
            scene_xyz = scene_xyz_base.float()
            wrist = anchors[:, :1]

            relative_position = scene_xyz[:, None] - anchors[:, :, None]
            bias = self.position_bias(relative_position / self.geometry_scale)
            bias = bias.permute(0, 3, 1, 2)

            scene_rel = (scene_xyz - wrist) / self.geometry_scale
            anchor_rel = (anchors - wrist) / self.geometry_scale
            scene_geometry = self.relative_value(scene_rel).reshape(
                batch_size, num_scene_tokens, self.num_heads, self.head_dim
            ).transpose(1, 2)
            anchor_geometry = self.relative_value(anchor_rel).reshape(
                batch_size, num_queries, self.num_heads, self.head_dim
            ).transpose(1, 2)

        value = v + scene_geometry.to(dtype=v.dtype)
        context = F.scaled_dot_product_attention(
            q, k, value, attn_mask=bias.to(dtype=q.dtype), dropout_p=0.0
        )
        context = context - anchor_geometry
        context = context.transpose(1, 2).reshape(
            batch_size, num_queries, self.embed_dim
        )
        return self.norm(body_tokens.float() + self.out(context).float())
