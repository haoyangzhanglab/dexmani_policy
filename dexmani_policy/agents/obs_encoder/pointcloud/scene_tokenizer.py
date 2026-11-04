import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_int

from .ops import farthest_point_sample, index_points, resolve_fps_random_config
from .rope import RotaryPositionEmbedding3D


class SceneTokenizer(nn.Module):
    def __init__(
        self,
        scene_channels: int,
        num_queries: int = 12,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_queries = positive_int(num_queries, "scene_num_queries")
        self.fps_random_config = dict(fps_random_config or {})
        self.fps_random_config["use_shuffle_output"] = False
        self.query_projection = nn.Linear(scene_channels, 192)
        self.query_norm = nn.LayerNorm(192)
        self.memory_norm = nn.LayerNorm(scene_channels)
        self.q = nn.Linear(192, 192)
        self.kv = nn.Linear(scene_channels, 384)
        self.out = nn.Linear(192, 192)
        self.ffn_norm = nn.LayerNorm(192)
        self.ffn = nn.Sequential(nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192))
        self.output_norm = nn.LayerNorm(192)
        self.rope = RotaryPositionEmbedding3D(48, scale=0.10, base=100.0)

    def forward(self, patch_features: torch.Tensor, patch_xyz: torch.Tensor) -> torch.Tensor:
        batch, patches, _ = patch_features.shape
        if patches < self.num_queries:
            raise ValueError("SceneTokenizer requires at least num_queries patches")
        with torch.no_grad(), torch.autocast(device_type=patch_xyz.device.type, enabled=False):
            xyz = patch_xyz.float().contiguous()
            query_xyz, indices = farthest_point_sample(
                xyz, self.num_queries,
                **resolve_fps_random_config(self.fps_random_config, self.training),
            )
            # RoPE recenters its input; Q and K must use the same origin.
            cos, sin = self.rope(torch.cat((query_xyz, xyz), dim=1))
        seed = index_points(patch_features, indices)
        z = self.query_projection(seed.to(self.query_projection.weight.dtype)).float()
        q = self.q(self.query_norm(z.to(self.query_norm.weight.dtype)))
        q = q.reshape(batch, self.num_queries, 4, 48).transpose(1, 2)
        kv = self.kv(self.memory_norm(patch_features.to(self.memory_norm.weight.dtype)))
        k, v = kv.reshape(batch, patches, 2, 4, 48).permute(2, 0, 3, 1, 4).unbind(0)
        q = self.rope.apply_rotary(q, cos[:, :, :self.num_queries], sin[:, :, :self.num_queries])
        k = self.rope.apply_rotary(k, cos[:, :, self.num_queries:], sin[:, :, self.num_queries:])
        update = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        update = update.transpose(1, 2).reshape(batch, self.num_queries, 192)
        z = z + self.out(update.to(self.out.weight.dtype)).float()
        z = z + self.ffn(self.ffn_norm(z.to(self.ffn_norm.weight.dtype))).float()
        return self.output_norm(z.to(self.output_norm.weight.dtype)).float()
