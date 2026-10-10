"""由读取前的交互状态调节逐指几何/触觉贡献，无显式阶段标签。"""

import torch
from torch import nn


class FingerModalityGate(nn.Module):
    """独立的 geometry/tactile gain；2*sigmoid 初始为 1，非概率或可靠性。

    输入必须独立于 query/late 的几何读取结果，保证两个实验共享门控路径。
    腕部 geometry=1 / tactile=0；缺失指的 tactile=0，几何仍可使用。
    """

    def __init__(self, token_dim: int = 192):
        super().__init__()
        self.hand_norm = nn.LayerNorm(token_dim)
        self.evidence_norm = nn.LayerNorm(token_dim)
        self.scene_norm = nn.LayerNorm(token_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * token_dim + 4, max(token_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(token_dim // 2, 16), 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, hand_context, evidence, scene_summary, force, valid, *, enabled=True):
        if hand_context.ndim != 3 or hand_context.shape[1] != 6:
            raise ValueError("hand_context must be [BT,6,D]")
        batch, _, dim = hand_context.shape
        if evidence.shape != hand_context.shape or scene_summary.shape != (batch, dim):
            raise ValueError("evidence must be [BT,6,D] and scene_summary [BT,D]")
        if force.shape != (batch, 5, 3) or valid.shape != (batch, 5) or valid.dtype != torch.bool:
            raise ValueError("force must be [BT,5,3] and valid bool [BT,5]")
        # 先清除无效通道；NaN*0 无法防止污染视觉 gain。
        touch = torch.where(valid[..., None], evidence[:, 1:], 0.0)
        force = torch.where(valid[..., None], force, 0.0)
        if enabled:
            if not all(torch.isfinite(x).all() for x in (hand_context, touch, scene_summary, force)):
                raise ValueError("modality gate inputs must be finite after validity masking")
            dtype = self.gate[0].weight.dtype
            inputs = torch.cat((
                self.hand_norm(hand_context[:, 1:].to(dtype)),
                self.evidence_norm(touch.to(dtype)),
                self.scene_norm(scene_summary.to(dtype))[:, None].expand(-1, 5, -1),
                force.to(dtype), valid[..., None].to(dtype),
            ), dim=-1)
            logits = self.gate(inputs)
            # Preserve small learned deviations from the unit initialization.
            # A BF16 sigmoid rounds near-zero logits to gain=1 while still
            # backpropagating, silently removing the actual modulation.
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            gains = 2.0 * logits.sigmoid()
        else:
            gains = hand_context.new_ones(batch, 5, 2)
        geometry = gains[..., :1]
        tactile = torch.where(valid[..., None], gains[..., 1:], 0.0)
        return {
            "geometry_gain": torch.cat((geometry.new_ones(batch, 1, 1), geometry), dim=1),
            "tactile_gain": torch.cat((tactile.new_zeros(batch, 1, 1), tactile), dim=1),
        }
