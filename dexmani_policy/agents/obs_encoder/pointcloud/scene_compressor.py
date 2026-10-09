"""将 PointPatchEncoder 的 patch tokens 压缩成少量场景 tokens。

结构：FPS 锚点 -> Voronoi 区域均值 -> 全局 cross-attention -> self-attention。
FPS 只选择摘要的位置；所有输入 patch 均参与区域汇聚和全局注意力。
该模块是待验证的研究实现，不包含交互 token、重建监督或策略接线。

输入：patch_token [B, M, D]，patch_center [B, M, 3]。
输出：scene_token [B, K, D]，scene_center [B, K, 3]。
scene_center 是锚点参考坐标；经过全局注意力后的 token 不再严格局部。

坐标约定：同一固定坐标系，推荐使用米制坐标。position_scale 是固定的
各向同性尺度；不要按每帧点云居中/缩放，否则会丢失场景的位置/尺度。
若 point_patch 的输入经过逐轴归一化，应先恢复 patch_center 的原坐标，
或显式接受在归一化坐标中进行空间划分的含义。输入须有限且没有 padding。
FPS/分区不可导，特征汇聚和注意力可导。FPS 从离质心最远的点开始，
距离并列时按输入索引打破平局；不保证严格排列不变或跨帧 token 对应。

用法（compressor 应在策略 __init__ 中构造一次）：
    compressor = PointPatchSceneCompressor(token_channels=192, num_scene_tokens=16)
    patch = point_patch(pointcloud)
    scene = compressor(patch["patch_token"], patch["patch_center"])
    # 将来在策略中沿 token 维拼接 scene["scene_token"] 与 interaction tokens。

输入接口核对自 haoyangzhanglab/dexmani_policy 的 point_patch.py，
commit 3019d6e618c1b95f0a5e27101d86d75efc35689c。
设计参考（本文件为独立实现，并非原论文方法的完整复现）：
  Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
  3DShape2VecSet: https://arxiv.org/abs/2301.11445
  官方编码器: https://github.com/1zb/3DShape2VecSet/blob/
      8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py

依赖 PyTorch 2.x；CPU 核验版本 2.5.1。直接运行本文件可执行示例。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@torch.no_grad()
def _spatial_partition(centers: Tensor, num_regions: int) -> tuple[Tensor, Tensor]:
    """返回互异锚点索引 [B,K] 和每个 patch 的区域编号 [B,M]。

    此纯 PyTorch FPS 适用于约 128 个 patch；大规模点云应使用专门的算子。
    在坐标重合时，仍将每个锚点自身分配给自己的区域，以避免空区域。
    """
    xyz = centers.detach().float()
    batch_size, num_points, _ = xyz.shape
    batch = torch.arange(batch_size, device=xyz.device)
    anchor_idx = torch.empty(
        batch_size, num_regions, dtype=torch.long, device=xyz.device
    )
    selected = torch.zeros(
        batch_size, num_points, dtype=torch.bool, device=xyz.device
    )
    min_dist = torch.full(
        (batch_size, num_points), float("inf"), device=xyz.device
    )
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
    assignment.scatter_(1, anchor_idx, own_region)
    return anchor_idx, assignment


class _SceneSelfAttentionBlock(nn.Module):
    """显式 prenorm block，便于与 point_patch 的残差结构对照。"""

    def __init__(self, channels: int, heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, channels)
        )

    def forward(self, token: Tensor) -> Tensor:
        normalized = self.norm1(token)
        update, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        token = token + update
        return token + self.ffn(self.norm2(token))


class PointPatchSceneCompressor(nn.Module):
    """固定数量的场景摘要；不接收语言、机器人状态或交互 queries。

    Args:
        token_channels: 与 PointPatchEncoder.out_dim 一致。
        num_scene_tokens: 输出 token 数 K，须满足 1 <= K <= 输入 patch 数。
        num_heads: 注意力头数；token_channels 须可整除 num_heads。
        mlp_ratio: cross/self-attention 的 FFN 隐藏维度比例。
        self_depth: 压缩后的 self-attention 层数，可设为 0 做消融。
        position_scale: 用于坐标嵌入的固定尺度，单位与输入坐标相同。

    只处理 [B,M,D]。多帧可先合并 B、T，再恢复 [B,T,K,D]；不得假设
    不同帧中编号相同的 scene token 对应同一物理区域。
    """

    def __init__(
        self,
        token_channels: int = 192,
        num_scene_tokens: int = 16,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        self_depth: int = 1,
        position_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if token_channels < 1 or num_heads < 1 or token_channels % num_heads:
            raise ValueError("token_channels must be positive and divisible by num_heads")
        if num_scene_tokens < 1 or self_depth < 0:
            raise ValueError("num_scene_tokens must be positive; self_depth must be >= 0")
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")
        if not math.isfinite(mlp_ratio) or int(token_channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")

        self.token_channels = token_channels
        self.num_scene_tokens = num_scene_tokens
        self.position_scale = float(position_scale)
        hidden_dim = int(token_channels * mlp_ratio)
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
            _SceneSelfAttentionBlock(token_channels, num_heads, hidden_dim)
            for _ in range(self_depth)
        )
        self.norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        patch_token: Tensor,
        patch_center: Tensor,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        if patch_token.ndim != 3 or patch_token.shape[-1] != self.token_channels:
            raise ValueError("patch_token must have shape [B,M,token_channels]")
        if patch_center.shape != (*patch_token.shape[:2], 3):
            raise ValueError("patch_center must have shape [B,M,3]")
        if patch_center.device != patch_token.device:
            raise ValueError("patch_token and patch_center must be on the same device")
        if not patch_token.is_floating_point() or not patch_center.is_floating_point():
            raise ValueError("patch_token and patch_center must have floating-point dtype")
        if patch_token.shape[0] == 0 or patch_token.shape[1] < self.num_scene_tokens:
            raise ValueError("B must be positive and M must be >= num_scene_tokens")

        anchor_idx, assignment = _spatial_partition(patch_center, self.num_scene_tokens)
        scene_center = patch_center.gather(
            1, anchor_idx[..., None].expand(-1, -1, 3)
        )
        membership = F.one_hot(assignment, self.num_scene_tokens).transpose(1, 2)
        region_size = membership.sum(-1)
        pool_weight = membership.float() / region_size[..., None].float()
        # FP32 累积用于低精度输入，防止较大区域求和时溢出。
        # 与 autocast 分离，避免 float() 后的 bmm 又被自动降精度。
        with torch.autocast(device_type=patch_token.device.type, enabled=False):
            pooling_dtype = (
                torch.float32 if patch_token.dtype in (torch.float16, torch.bfloat16)
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


def example() -> None:
    torch.manual_seed(0)
    compressor = PointPatchSceneCompressor().eval()
    patch_token = torch.randn(2, 128, 192)
    patch_center = torch.rand(2, 128, 3)
    with torch.no_grad():
        outputs = compressor(patch_token, patch_center, return_intermediate=True)
    for name, value in outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("parameters:", sum(p.numel() for p in compressor.parameters()))


if __name__ == "__main__":
    example()
