import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.obs_encoder.interaction.attention import (
    InteractionSelfAttentionBlock,
)


@torch.no_grad()
def _spatial_partition(
    centers: torch.Tensor, num_regions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = centers.detach().float()
    batch_size, num_points, _ = xyz.shape
    batch = torch.arange(batch_size, device=xyz.device)
    anchor_idx = torch.empty(
        batch_size, num_regions, dtype=torch.long, device=xyz.device
    )
    selected = torch.zeros(batch_size, num_points, dtype=torch.bool, device=xyz.device)
    min_dist = torch.full((batch_size, num_points), float("inf"), device=xyz.device)
    farthest = (xyz - xyz.mean(dim=1, keepdim=True)).square().sum(-1).argmax(-1)
    for region in range(num_regions):
        anchor_idx[:, region] = farthest
        selected[batch, farthest] = True
        distance = (xyz - xyz[batch, farthest, None]).square().sum(-1)
        min_dist = torch.minimum(min_dist, distance)
        if region + 1 < num_regions:
            farthest = min_dist.masked_fill(selected, -1).argmax(-1)

    anchors = xyz.gather(1, anchor_idx[..., None].expand(-1, -1, 3))
    distance = (xyz[:, :, None] - anchors[:, None]).square().sum(-1)
    assignment = distance.argmin(-1)
    own_region = torch.arange(num_regions, device=xyz.device).expand(batch_size, -1)
    # 重合锚点也保留各自区域，避免空区域。
    assignment.scatter_(1, anchor_idx, own_region)
    return anchor_idx, assignment


class SceneContextPool(nn.Module):
    """将已编码的点云 patches 汇聚为少量场景上下文 token；中心使用米制基座/世界坐标。"""

    def __init__(
        self,
        token_channels: int = 192,
        num_scene_tokens: int = 16,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        self_depth: int = 1,
        position_scale: float = 1.0,
    ):
        super().__init__()
        hidden_dim = int(token_channels * mlp_ratio)
        if num_scene_tokens < 1 or self_depth < 0 or hidden_dim < 1:
            raise ValueError(
                "num_scene_tokens and hidden_dim must be positive; self_depth must be >= 0"
            )
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")

        self.token_channels = token_channels
        self.num_scene_tokens = num_scene_tokens
        self.position_scale = float(position_scale)
        self.position_embed = nn.Sequential(
            nn.Linear(3, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.query_norm = nn.LayerNorm(token_channels)
        self.context_norm = nn.LayerNorm(token_channels)
        self.cross_attn = nn.MultiheadAttention(
            token_channels, num_heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(token_channels)
        self.ffn = nn.Sequential(
            nn.Linear(token_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_channels),
        )
        self.blocks = nn.ModuleList(
            [
                InteractionSelfAttentionBlock(token_channels, num_heads, hidden_dim)
                for _ in range(self_depth)
            ]
        )
        self.norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        patch_token: torch.Tensor,
        patch_center: torch.Tensor,
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        if patch_token.ndim != 3 or patch_token.shape[-1] != self.token_channels:
            raise ValueError("patch_token must have shape [B,M,token_channels]")
        if patch_center.shape != (*patch_token.shape[:2], 3):
            raise ValueError("patch_center must have shape [B,M,3]")
        if patch_token.shape[1] < self.num_scene_tokens:
            raise ValueError("num_scene_tokens must not exceed the number of patches")

        anchor_idx, assignment = _spatial_partition(patch_center, self.num_scene_tokens)
        scene_center = patch_center.gather(1, anchor_idx[..., None].expand(-1, -1, 3))
        membership = F.one_hot(assignment, self.num_scene_tokens).transpose(1, 2)
        region_size = membership.sum(-1)
        pool_weight = membership.float() / region_size[..., None].float()
        # 关闭 autocast，低精度输入以 FP32 累积。
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pooling_dtype = (
                torch.float32
                if patch_token.dtype in (torch.float16, torch.bfloat16)
                else patch_token.dtype
            )
            pooled = torch.bmm(
                pool_weight.to(pooling_dtype), patch_token.to(pooling_dtype)
            ).to(patch_token.dtype)

        position = self.position_embed(
            patch_center.to(dtype=patch_token.dtype) / self.position_scale
        )
        anchor_position = position.gather(
            1, anchor_idx[..., None].expand(-1, -1, self.token_channels)
        )
        query = pooled + anchor_position
        context = self.context_norm(patch_token + position)
        update, attention = self.cross_attn(
            self.query_norm(query),
            context,
            context,
            need_weights=return_intermediate,
            average_attn_weights=False,
        )
        scene_token = query + update
        scene_token = scene_token + self.ffn(self.ffn_norm(scene_token))
        for block in self.blocks:
            scene_token = block(scene_token)
        scene_token = self.norm(scene_token)

        outputs = {"scene_token": scene_token, "scene_center": scene_center}
        if return_intermediate:
            outputs.update(
                scene_center_idx=anchor_idx,
                patch_to_scene=assignment,
                region_size=region_size,
                pool_weight=pool_weight,
                pooled_token=pooled,
                cross_attention=attention,
            )
        return outputs

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_scene_tokens, self.token_channels)
