import torch
from torch import nn
from torch.nn import functional as F


class GeoAwareRelationAttention(nn.Module):
    """Update hand tokens with geometry-biased attention to Base-frame scene tokens."""

    embed_dim = 192
    num_heads = 4
    head_dim = embed_dim // num_heads
    relative_position_scale = 0.10

    def __init__(self) -> None:
        super().__init__()
        self.hand_norm = nn.LayerNorm(self.embed_dim)
        self.hand_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.scene_norm = nn.LayerNorm(self.embed_dim)
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.scene_kv = nn.Linear(self.embed_dim, 2 * self.embed_dim)
        self.geometry_bias = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, self.num_heads))
        nn.init.zeros_(self.geometry_bias[-1].weight)
        nn.init.zeros_(self.geometry_bias[-1].bias)
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

        with torch.autocast(device_type=hand_landmarks_base.device.type, enabled=False):
            relative_xyz_base = scene_points_base.float()[:, None] - hand_landmarks_base.float()[:, :, None]
            relative_distance = torch.linalg.vector_norm(relative_xyz_base, dim=-1, keepdim=True)
            geometry_input = torch.cat(
                (
                    relative_xyz_base / self.relative_position_scale,
                    relative_distance / self.relative_position_scale,
                ),
                dim=-1,
            )
            geometry_bias = self.geometry_bias(geometry_input)
        geometry_bias = geometry_bias.permute(0, 3, 1, 2).to(
            dtype=query.dtype, memory_format=torch.contiguous_format,
        )
        scene_context = F.scaled_dot_product_attention(query, key, value, attn_mask=geometry_bias)
        scene_context = scene_context.transpose(1, 2).flatten(2)
        hand_tokens = hand_tokens + self.context_projection(scene_context)
        return self.norm(hand_tokens + self.ffn(self.ffn_norm(hand_tokens)))
