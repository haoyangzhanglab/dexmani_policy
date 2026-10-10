"""按腕部和指尖查询可见场景几何，检索更新与手部残差分开返回。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.interaction.attention import GeometryCrossAttention
from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.proprio.hand_kinematics import HandKinematicsEncoder


def nearest_patch_evidence(
    anchors: torch.Tensor,
    members: torch.Tensor,
    member_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回最近有效成员的相对向量、米制距离和有效位；无观测距离为 inf。

    该证据仅来自可见点，不表示真实接触点，也不证明未观测区域为空。
    """
    valid = torch.isfinite(members).all(-1)
    if member_valid is not None:
        valid = valid & member_valid
    safe_members = torch.where(valid[..., None], members.float(), 0.0)
    delta = safe_members[:, None] - anchors.float()[:, :, None, None]
    squared = delta.square().sum(-1).masked_fill(~valid[:, None], float("inf"))
    squared_min, index = squared.min(dim=-1)
    observed = torch.isfinite(squared_min)
    selected = delta.gather(3, index[..., None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    selected = torch.where(observed[..., None], selected, 0.0)
    return selected, squared_min.sqrt(), observed


class HandSceneRelationEncoder(nn.Module):
    """为 wrist + 五指读取场景上下文与近场几何，不在此融合残差或协调手指。

    query_context 只改变检索 query；输出 hand_query 始终是直接运动学编码。
    上层统一构造 kin + state + touch + relation_update，并共享归一化与手部注意力。
    """

    def __init__(
        self,
        token_channels: int = 192,
        context_sigmas: Sequence[float] = (0.08, 0.08, 0.20, float("inf")),
        near_sigmas: Sequence[float] = (0.015, 0.015, 0.04, 0.04),
        metric_scale: float = 0.10,
        position_scale: float = 1.0,
        radius_multiple: float = 3.0,
        num_heads: int = 4,
        position_num_frequencies: int = 6,
        position_max_wavelength: float = 0.64,
        edge_frame: str = "wrist",
    ):
        super().__init__()
        if token_channels <= 0 or num_heads <= 0 or token_channels % num_heads:
            raise ValueError("token_channels must be positive and divisible by num_heads")
        if len(context_sigmas) != num_heads or len(near_sigmas) != num_heads:
            raise ValueError("context_sigmas and near_sigmas must have length num_heads")
        scales = (metric_scale, position_scale, radius_multiple, *near_sigmas)
        if any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("scales and near_sigmas must be finite and positive")
        if any(math.isnan(s) or s <= 0 for s in context_sigmas):
            raise ValueError("context_sigmas must be positive; infinity is allowed")
        if edge_frame not in ("wrist", "base"):
            raise ValueError("edge_frame must be 'wrist' or 'base'")

        self.token_channels = token_channels
        self.metric_scale = metric_scale
        self.edge_frame = edge_frame
        proximity_scales = (min(near_sigmas), max(near_sigmas))
        self.hand_encoder = HandKinematicsEncoder(
            token_channels=token_channels,
            position_scale=position_scale,
            hand_scale=metric_scale,
            num_frequencies=position_num_frequencies,
            max_wavelength=position_max_wavelength,
        )
        self.context = GeometryCrossAttention(
            token_channels, context_sigmas, metric_scale, proximity_scales, None
        )
        self.near = GeometryCrossAttention(
            token_channels, near_sigmas, metric_scale, proximity_scales, radius_multiple
        )

    def forward(
        self,
        pointcloud: torch.Tensor,
        patches: Mapping[str, torch.Tensor],
        eef_pose: torch.Tensor,
        fingertip_points: torch.Tensor,
        *,
        member_valid: torch.Tensor | None = None,
        query_context: torch.Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """patches 来自 GeometryPatchEncoder(return_intermediate=True)。

        pointcloud、eef_pose XYZ 与 fingertip_points 必须共用米制基座/世界坐标。
        rot6d 为腕部旋转矩阵前两列依次拼接；指序为 thumb/index/middle/ring/pinky。
        edge_frame 仅选择相对边的轴向，不变换输入点云或输出 anchor，也不保证网络等变。
        member_valid 只屏蔽几何成员；编码前仍需处理无效点，patch 特征必须有限。
        query_context 接受 [B,D] 的共享状态或 [B,6,D] 的逐锚点条件。
        """
        if pointcloud.ndim != 3 or pointcloud.shape[-1] < 3 or 0 in pointcloud.shape[:2]:
            raise ValueError("pointcloud must be nonempty [B,N,C>=3]")
        batch = pointcloud.shape[0]
        if eef_pose.shape != (batch, 9):
            raise ValueError("eef_pose must have shape [B,9] with the pointcloud batch size")
        global_memory, local_memory = patches["patch_token"], patches["local_token"]
        centers, indices = patches["patch_center"], patches["neighbor_idx"]
        if indices.ndim != 3 or indices.dtype != torch.long or indices.shape[0] != batch:
            raise ValueError("neighbor_idx must be a torch.long tensor with shape [B,M,K]")
        if indices.numel() == 0 or indices.min() < 0 or indices.max() >= pointcloud.shape[1]:
            raise ValueError("neighbor_idx contains empty or out-of-bounds indices")
        memory_shape = (batch, indices.shape[1], self.token_channels)
        if global_memory.shape != memory_shape or local_memory.shape != memory_shape:
            raise ValueError("patch_token and local_token must have shape [B,M,token_channels]")
        if centers.shape != (batch, indices.shape[1], 3):
            raise ValueError("patch_center must have shape [B,M,3]")
        if not torch.isfinite(global_memory).all() or not torch.isfinite(local_memory).all():
            raise ValueError("patch features must be finite; member_valid only masks geometry")
        if member_valid is not None and (
            member_valid.shape != indices.shape or member_valid.dtype != torch.bool
        ):
            raise ValueError("member_valid must be boolean with shape [B,M,K]")

        hand = self.hand_encoder(eef_pose, fingertip_points)
        anchors = hand["hand_center"]
        hand_query = hand["hand_token"]
        query = hand_query
        if query_context is not None:
            if query_context.shape == (batch, self.token_channels):
                query_context = query_context[:, None]
            elif query_context.shape != (batch, 6, self.token_channels):
                raise ValueError("query_context must have shape [B,D] or [B,6,D]")
            if not torch.isfinite(query_context).all():
                raise ValueError("query_context must be finite")
            query = query + query_context.to(query.dtype)

        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            members = index_points(pointcloud[..., :3], indices)
            relative, distance, observed = nearest_patch_evidence(anchors, members, member_valid)
            center_valid = torch.isfinite(centers).all(-1)
            safe_centers = torch.where(center_valid[..., None], centers.float(), 0.0)
            center_relative = safe_centers[:, None] - anchors[:, :, None]
            center_distance = center_relative.norm(dim=-1)
            context_valid = center_valid[:, None] & observed
            if self.edge_frame == "wrist":
                # 行向量乘 R，相当于列向量左乘 R^T：base/world -> wrist。
                rotation = hand["wrist_rotation"]
                relative = torch.einsum("bqmi,bij->bqmj", relative, rotation)
                center_relative = torch.einsum("bqmi,bij->bqmj", center_relative, rotation)

        context, context_attention, _ = self.context(
            query, global_memory, center_relative, center_distance, context_valid
        )
        near, near_attention, support = self.near(query, local_memory, relative, distance, observed)
        # 即使 null_value/out bias 非零，没有局部观测支撑时也不能注入近场更新。
        near_update = near * support.any(-1, keepdim=True).to(near.dtype)
        outputs = {
            "hand_query": hand_query,
            "relation_update": context + near_update,
            "interaction_center": anchors,
        }
        if return_intermediate:
            outputs.update(
                context_update=context,
                near_update=near_update,
                min_observed_distance=distance.min(-1).values,
                near_has_support=support,
                near_null_mass=near_attention[..., -1],
                context_null_mass=context_attention[..., -1],
                near_attention=near_attention,
                context_attention=context_attention,
                context_relative=center_relative,
                surface_relative=relative,
                surface_distance=distance,
                surface_valid=observed,
            )
        return outputs

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (6, self.token_channels)
