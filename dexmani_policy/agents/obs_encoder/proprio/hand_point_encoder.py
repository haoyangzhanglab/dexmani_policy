import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.position_encodings import NeRFSinusoidalPosEmb3D


class WristFingertipEncoder(nn.Module):
    """编码米制世界系 wrist / fingertip，输出 [B,6,D] token 与 [B,6,3] center。

    eef_pose [B,9] 为 XYZ + 有效 rot6d（旋转矩阵前两列依次拼接）；
    fingertip_points 为 [B,5,3] 或 [B,15]，顺序 thumb/index/middle/ring/pinky。
    输出按 wrist、五指排列；位置与手形位移均沿世界轴，max_wavelength 单位为米。
    """

    def __init__(
        self,
        token_channels: int = 192,
        position_scale: float = 1.0,
        hand_scale: float = 0.1,
        num_frequencies: int = 6,
        max_wavelength: float = 0.64,
    ):
        super().__init__()
        if min(position_scale, hand_scale, max_wavelength) <= 0:
            raise ValueError("position_scale, hand_scale and max_wavelength must be positive")

        self.token_channels = token_channels
        self.position_scale = float(position_scale)
        self.hand_scale = float(hand_scale)
        self.max_wavelength = float(max_wavelength)
        self.position_pe = NeRFSinusoidalPosEmb3D(num_frequencies)
        self.body_identity = nn.Parameter(torch.randn(6, token_channels) * 0.02)
        self.position_embed = nn.Sequential(
            nn.Linear(3 + self.position_pe.out_dim, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.orientation_embed = nn.Sequential(
            nn.Linear(6, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.shape_embed = nn.Sequential(
            nn.Linear(15, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.output_norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        eef_pose: torch.Tensor,
        fingertip_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if eef_pose.ndim != 2 or eef_pose.shape[-1] != 9:
            raise ValueError("eef_pose must have shape [B,9]")
        batch = eef_pose.shape[0]
        if fingertip_points.shape == (batch, 15):
            fingertip_points = fingertip_points.reshape(batch, 5, 3)
        if fingertip_points.shape != (batch, 5, 3):
            raise ValueError("fingertip_points must have shape [B,5,3] or [B,15]")

        # 几何和 Fourier 相位保持 FP32。
        with torch.autocast(device_type=eef_pose.device.type, enabled=False):
            wrist = eef_pose.float()
            fingertips = fingertip_points.float()
            centers = torch.cat((wrist[:, None, :3], fingertips), dim=1)
            position_features = torch.cat(
                (centers / self.position_scale,
                 self.position_pe(centers * (2 * math.pi / self.max_wavelength))), dim=-1
            )
            first, second = wrist[:, 3:6], wrist[:, 6:9]
            x = F.normalize(first, dim=-1)
            y = F.normalize(second - (second * x).sum(-1, keepdim=True) * x, dim=-1)
            orientation = torch.cat((x, y), dim=-1)
            hand_offsets = (fingertips - wrist[:, None, :3]) / self.hand_scale

        dtype = self.position_embed[0].weight.dtype
        position = self.position_embed(position_features.to(dtype))
        context = self.orientation_embed(orientation.to(dtype))
        context = context + self.shape_embed(hand_offsets.flatten(1).to(dtype))
        tokens = position + self.body_identity[None].to(position.dtype) + context[:, None]
        return {"hand_token": self.output_norm(tokens), "hand_center": centers}

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (6, self.token_channels)
