"""世界坐标系下的 wrist / fingertip 空间编码，不执行坐标系变换。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.position_encodings import NeRFSinusoidalPosEmb3D


def rotation_from_columns_6d(rotation: torch.Tensor) -> torch.Tensor:
    """前两列依次拼接的 rot6d -> 旋转矩阵；仅用于编码腕部朝向。"""
    rotation = rotation.float()
    first, second = rotation[..., :3], rotation[..., 3:]
    if not torch.isfinite(rotation).all() or (first.norm(dim=-1) < 1e-6).any():
        raise ValueError("rotation contains invalid first columns")
    x = F.normalize(first, dim=-1)
    y = second - (second * x).sum(-1, keepdim=True) * x
    tolerance = 1e-6 * second.norm(dim=-1).clamp_min(1.0)
    if (y.norm(dim=-1) < tolerance).any():
        raise ValueError("rotation columns must be linearly independent")
    y = F.normalize(y, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


class WristFingertipEncoder(nn.Module):
    """位置 Fourier 特征 + 身份 + 腕部朝向 + 整体手形，输出六个 token。

    输入使用同一世界坐标系、米制、未做逐轴数据归一化的观测：
    ``eef_pose: [B,9]`` = wrist XYZ + rot6d（旋转矩阵前两列依次拼接）；
    ``fingertip_points: [B,5,3]``，也接受仿真存储的 ``[B,15]``。
    固定顺序为 wrist / thumb / index / middle / ring / pinky。

    手形用五指到腕部的位移编码，位移仍沿世界坐标轴，不乘腕部旋转。
    所有 token 均保留绝对世界位置；朝向和整体手形作为共享上下文。
    ``hand_center`` 返回原始米制世界位置，可直接作为场景交互锚点。
    ``max_wavelength`` 单位为米，默认六频率对应 0.64 至 0.02 m。
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
        if type(token_channels) is not int or token_channels <= 0:
            raise ValueError("token_channels must be a positive integer")
        if any(not math.isfinite(s) or s <= 0 for s in (position_scale, hand_scale, max_wavelength)):
            raise ValueError("position_scale, hand_scale and max_wavelength must be finite and positive")
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
        if not eef_pose.is_floating_point() or not fingertip_points.is_floating_point():
            raise ValueError("eef_pose and fingertip_points must be floating-point tensors")
        if eef_pose.device != fingertip_points.device:
            raise ValueError("eef_pose and fingertip_points must be on the same device")
        if not torch.isfinite(eef_pose).all() or not torch.isfinite(fingertip_points).all():
            raise ValueError("eef_pose and fingertip_points must contain finite values")

        # 几何和 Fourier 相位使用 FP32；只在进入可学习投影时转换精度。
        with torch.autocast(device_type=eef_pose.device.type, enabled=False):
            wrist = eef_pose.float()
            fingertips = fingertip_points.float()
            centers = torch.cat((wrist[:, None, :3], fingertips), dim=1)
            position_features = torch.cat(
                (centers / self.position_scale,
                 self.position_pe(centers * (2 * math.pi / self.max_wavelength))), dim=-1
            )
            rotation = rotation_from_columns_6d(wrist[:, 3:])
            orientation = torch.cat((rotation[..., 0], rotation[..., 1]), dim=-1)
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
