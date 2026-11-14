"""Cross-attention with metric Base-frame relative position bias."""

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_float


class Relative3DCrossAttention(nn.Module):
    embed_dim = 192
    num_heads = 4
    head_dim = embed_dim // num_heads

    def __init__(self, geometry_scale: float = 0.10) -> None:
        super().__init__()
        self.geometry_scale = positive_float(geometry_scale, "geometry_scale")
        dim = self.embed_dim
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.position_bias = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, self.num_heads)
        )
        nn.init.zeros_(self.position_bias[-1].weight)
        nn.init.zeros_(self.position_bias[-1].bias)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        body_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        anchors_base: torch.Tensor,
        scene_xyz_base: torch.Tensor,
    ) -> torch.Tensor:
        """Read scene [B,N,192] into body [B,L,192], returning [B,L,192].

        anchors_base [B,L,3] and scene_xyz_base [B,N,3] must share a
        Base frame and be expressed in meters, without normalization.
        """
        for name, tokens in (("body_tokens", body_tokens), ("scene_tokens", scene_tokens)):
            if (
                tokens.ndim != 3
                or tokens.shape[-1] != self.embed_dim
                or min(tokens.shape[:2]) == 0
            ):
                raise ValueError(
                    f"{name} requires [B>0,T>0,{self.embed_dim}], got {tuple(tokens.shape)}"
                )
        b, l, _ = body_tokens.shape
        if scene_tokens.shape[0] != b:
            raise ValueError(
                f"scene_tokens and body_tokens must share B, got "
                f"{tuple(scene_tokens.shape)} and {tuple(body_tokens.shape)}"
            )
        n = scene_tokens.shape[1]
        for name, xyz, shape in (
            ("anchors_base", anchors_base, (b, l, 3)),
            ("scene_xyz_base", scene_xyz_base, (b, n, 3)),
        ):
            if xyz.shape != shape:
                raise ValueError(f"{name} requires {shape}, got {tuple(xyz.shape)}")

        q = self.q(self.query_norm(body_tokens)).reshape(
            b, l, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.kv(self.memory_norm(scene_tokens)).reshape(
            b, n, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)

        with torch.autocast(device_type=anchors_base.device.type, enabled=False):
            delta = scene_xyz_base.float()[:, None] - anchors_base.float()[:, :, None]
            bias = self.position_bias(delta / self.geometry_scale).permute(0, 3, 1, 2)

        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype), dropout_p=0.0,
        )
        context = context.transpose(1, 2).reshape(b, l, self.embed_dim)
        return self.norm(body_tokens + self.out(context))
