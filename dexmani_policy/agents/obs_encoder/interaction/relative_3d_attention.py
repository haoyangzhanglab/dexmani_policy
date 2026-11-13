"""Cross-attention with Base-frame relative position bias and values."""

import torch
from torch import nn
from torch.nn import functional as F


class Relative3DCrossAttention(nn.Module):
    embed_dim, num_heads, geometry_scale = 192, 4, 0.10
    head_dim = embed_dim // num_heads

    def __init__(self):
        super().__init__()
        dim = self.embed_dim
        self.query_norm, self.memory_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)
        self.position_bias = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, self.num_heads)
        )
        self.out, self.norm = nn.Linear(dim, dim), nn.LayerNorm(dim)
        self.relative_value = nn.Linear(3, dim, bias=False)
        nn.init.zeros_(self.relative_value.weight)

    def forward(self, body_tokens, scene_tokens, anchors_base, scene_xyz_base):
        b, l, _ = body_tokens.shape
        n = scene_tokens.shape[1]
        q = self.q(self.query_norm(body_tokens)).reshape(
            b, l, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.kv(self.memory_norm(scene_tokens)).reshape(
            b, n, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)

        with torch.autocast(device_type=anchors_base.device.type, enabled=False):
            anchors, scene_xyz = anchors_base.float(), scene_xyz_base.float()
            wrist = anchors[:, :1]
            delta = scene_xyz[:, None] - anchors[:, :, None]
            bias = self.position_bias(delta / self.geometry_scale).permute(0, 3, 1, 2)
            scene_rel = (scene_xyz - wrist) / self.geometry_scale
            anchor_rel = (anchors - wrist) / self.geometry_scale
            relative_values = self.relative_value(torch.cat((scene_rel, anchor_rel), dim=1))
            scene_geometry, anchor_geometry = relative_values.split((n, l), dim=1)
            scene_geometry = scene_geometry.reshape(
                b, n, self.num_heads, self.head_dim
            ).transpose(1, 2)
            anchor_geometry = anchor_geometry.reshape(
                b, l, self.num_heads, self.head_dim
            ).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            q, k, v + scene_geometry.to(v.dtype),
            attn_mask=bias.to(q.dtype), dropout_p=0.0,
        )
        context = (context - anchor_geometry).transpose(1, 2).reshape(b, l, self.embed_dim)
        return self.norm(body_tokens.float() + self.out(context).float())
