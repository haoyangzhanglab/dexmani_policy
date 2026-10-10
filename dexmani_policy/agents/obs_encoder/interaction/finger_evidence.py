"""将已经编码的触觉证据对应到五个手指；腕部槽位不构造触觉。"""

import torch
from torch import nn


class FingerEvidenceEncoder(nn.Module):
    """[BT,5,F] + normalized sensor channels -> [BT,6,D].

    三轴通道是传感器读数，不假设已变换为世界系物理合力。
    valid embedding 区分有效零读数和缺失；缺失证据严格为零。
    """

    def __init__(self, frame_dim: int = 64, token_dim: int = 192):
        super().__init__()
        self.frame_norm = nn.LayerNorm(frame_dim)
        self.frame_projection = nn.Linear(frame_dim, token_dim, bias=False)
        self.force_projection = nn.Linear(3, token_dim, bias=False)
        self.valid_embedding = nn.Parameter(torch.zeros(token_dim))
        nn.init.normal_(self.valid_embedding, std=0.02)

    def forward(self, frames, force, valid):
        if frames.ndim != 3 or frames.shape[1] != 5:
            raise ValueError("frames must be [BT,5,F]")
        if force.shape != (*frames.shape[:2], 3):
            raise ValueError("force must be [BT,5,3]")
        if valid.dtype != torch.bool or valid.shape != frames.shape[:2]:
            raise ValueError("valid must be bool [BT,5]")
        frames = torch.where(valid[..., None], frames, 0.0)
        force = torch.where(valid[..., None], force, 0.0)
        if not torch.isfinite(frames).all() or not torch.isfinite(force).all():
            raise ValueError("nonfinite tactile evidence in valid fingers")
        dtype = self.frame_projection.weight.dtype
        evidence = (
            self.frame_projection(self.frame_norm(frames.to(dtype)))
            + self.force_projection(force.to(dtype))
            + self.valid_embedding
        )
        evidence = torch.where(valid[..., None], evidence, 0.0)
        wrist = evidence.new_zeros(evidence.shape[0], 1, evidence.shape[-1])
        return torch.cat((wrist, evidence), dim=1)
