"""InterAct-only cross-attention with Base-frame relative position bias."""

import torch
from torch import nn
from torch.nn import functional as F


class Relative3DCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(192)
        self.memory_norm = nn.LayerNorm(192)
        self.q = nn.Linear(192, 192)
        self.kv = nn.Linear(192, 384)
        self.position_bias = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 4))
        self.out = nn.Linear(192, 192)
        self.norm = nn.LayerNorm(192)

    def relative_bias(self, anchors_base, scene_xyz_base):
        with torch.autocast(device_type=anchors_base.device.type, enabled=False):
            delta = scene_xyz_base.float()[:, None] - anchors_base.float()[:, :, None]
            return self.position_bias(delta / 0.10).permute(0, 3, 1, 2)

    def forward(self, body_tokens, scene_tokens, anchors_base, scene_xyz_base):
        b, n, _ = scene_tokens.shape
        q = self.q(self.query_norm(body_tokens)).reshape(b, 6, 4, 48).transpose(1, 2)
        k, v = self.kv(self.memory_norm(scene_tokens)).reshape(b, n, 2, 4, 48).permute(2, 0, 3, 1, 4)
        bias = self.relative_bias(anchors_base, scene_xyz_base).to(q.dtype)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        context = context.transpose(1, 2).reshape(b, 6, 192)
        return self.norm(body_tokens.float() + self.out(context).float())
