import torch
from torch import nn
from torch.nn import functional as F

from ..pointcloud.rope import RotaryPositionEmbedding3D


class GeoAwareRelationAttention(nn.Module):
    """Query Base-frame scene patches with spatially rotated hand queries and scene keys."""

    embed_dim = 192
    num_heads = 4
    head_dim = embed_dim // num_heads

    def __init__(self, rope_scale: float = 0.05, rope_base: float = 100.0) -> None:
        super().__init__()
        self.hand_norm = nn.LayerNorm(self.embed_dim)
        self.hand_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.scene_norm = nn.LayerNorm(self.embed_dim)
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.scene_kv = nn.Linear(self.embed_dim, 2 * self.embed_dim)
        self.rope = RotaryPositionEmbedding3D(self.head_dim, rope_scale, rope_base)
        self.context_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.ffn_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(nn.Linear(self.embed_dim, 384), nn.GELU(), nn.Linear(384, self.embed_dim))
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hand_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        hand_landmarks_base: torch.Tensor,
        scene_points_base: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hand_tokens.shape[0] != scene_tokens.shape[0]
            or hand_landmarks_base.shape != (*hand_tokens.shape[:2], 3)
            or scene_points_base.shape != (*scene_tokens.shape[:2], 3)
        ):
            raise ValueError("Hand and scene tokens must share a batch and have one XYZ per token")
        normalized_hand = self.hand_norm(hand_tokens)
        hand_update, _ = self.hand_attention(
            normalized_hand, normalized_hand, normalized_hand, need_weights=False,
        )
        hand_tokens = hand_tokens + hand_update

        query = self.query_projection(self.query_norm(hand_tokens)).unflatten(
            -1, (self.num_heads, self.head_dim),
        ).transpose(1, 2)
        scene_kv = self.scene_kv(self.scene_norm(scene_tokens)).unflatten(
            -1, (2, self.num_heads, self.head_dim),
        )
        key, value = scene_kv.permute(2, 0, 3, 1, 4).unbind(0)

        origin = scene_points_base.float().mean(dim=1, keepdim=True)
        query = self.rope.apply_rotary(query, *self.rope(hand_landmarks_base, origin=origin))
        key = self.rope.apply_rotary(key, *self.rope(scene_points_base, origin=origin))
        scene_context = F.scaled_dot_product_attention(query, key, value)
        scene_context = scene_context.transpose(1, 2).flatten(2)
        hand_tokens = hand_tokens + self.context_projection(scene_context)
        return self.norm(hand_tokens + self.ffn(self.ffn_norm(hand_tokens)))
