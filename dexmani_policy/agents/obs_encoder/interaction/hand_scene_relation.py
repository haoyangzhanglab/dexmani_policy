"""按腕部和指尖查询可见场景几何，检索更新与手部残差分开返回。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.interaction.attention import (
    GeometryCrossAttention,
)
from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points


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
    selected = delta.gather(
        3,
        index[..., None, None].expand(-1, -1, -1, 1, 3),
    ).squeeze(3)
    selected = torch.where(observed[..., None], selected, 0.0)
    return selected, squared_min.sqrt(), observed


class HandSceneRelationEncoder(nn.Module):
    """为 wrist + 五指读取场景上下文与近场几何，不在此融合残差或协调手指。

    query_context 只改变检索 query；hand_query 原样返回调用方提供的空间 token。
    上层统一构造 spatial + state + touch + relation_update；此处不重复编码手部。
    """

    def __init__(
        self,
        token_channels: int = 192,
        context_sigmas: Sequence[float] = (0.08, 0.08, 0.20, float("inf")),
        near_sigmas: Sequence[float] = (0.015, 0.015, 0.04, 0.04),
        metric_scale: float = 0.10,
        radius_multiple: float = 3.0,
        num_heads: int = 4,
    ):
        super().__init__()
        if token_channels <= 0 or num_heads <= 0 or token_channels % num_heads:
            raise ValueError(
                "token_channels must be positive and divisible by num_heads"
            )
        if len(context_sigmas) != num_heads or len(near_sigmas) != num_heads:
            raise ValueError(
                "context_sigmas and near_sigmas must have length num_heads"
            )
        scales = (metric_scale, radius_multiple, *near_sigmas)
        if any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("scales and near_sigmas must be finite and positive")
        if any(math.isnan(s) or s <= 0 for s in context_sigmas):
            raise ValueError("context_sigmas must be positive; infinity is allowed")

        self.token_channels = token_channels
        self.metric_scale = metric_scale
        proximity_scales = (min(near_sigmas), max(near_sigmas))
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
        anchor_xyz: torch.Tensor,
        hand_tokens: torch.Tensor,
        *,
        member_valid: torch.Tensor | None = None,
        query_context: torch.Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """patches 来自 GeometryAwarePointTransformer(return_geometry=True)，tokens 已由调用层投影。

        anchor_xyz [BT,6,3] 与点云共用米制世界坐标；hand_tokens [BT,6,D] 已编码。
        两者顺序均为 wrist/thumb/index/middle/ring/pinky。关系边始终沿世界轴。
        member_valid 只屏蔽几何成员；编码前仍需处理无效点，patch 特征必须有限。
        query_context 接受 [B,D] 的共享状态或 [B,6,D] 的逐锚点条件。
        物理锚点与特征分开传递；位置缩放只用于编码器，不得用于几何距离。
        """
        # Both readers use the SAME contextual descriptors. Near support is
        # still determined by observed raw members, not a separate local head.
        memory = patches["tokens"]
        centers, indices = patches["xyz"], patches["neighbor_indices"]
        token_valid = patches["mask"]
        if token_valid.dtype != torch.bool or token_valid.shape != centers.shape[:2]:
            raise ValueError("patch mask must be boolean [B,M]")
        if not torch.isfinite(memory).all():
            raise ValueError(
                "patch features must be finite; member_valid only masks geometry"
            )
        if member_valid is not None and (
            member_valid.shape != indices.shape or member_valid.dtype != torch.bool
        ):
            raise ValueError("member_valid must be boolean with shape [B,M,K]")

        batch = memory.shape[0]
        if anchor_xyz.shape != (batch, 6, 3) or not anchor_xyz.is_floating_point():
            raise ValueError("anchor_xyz must be floating point [BT,6,3]")
        if hand_tokens.shape != (batch, 6, self.token_channels):
            raise ValueError("hand_tokens must have shape [BT,6,D]")
        if anchor_xyz.device != memory.device or hand_tokens.device != memory.device:
            raise ValueError("anchors, hand tokens and patch memory must share a device")
        if not torch.isfinite(anchor_xyz).all() or not torch.isfinite(hand_tokens).all():
            raise ValueError("hand anchors and tokens must be finite")
        anchors = anchor_xyz.float()
        hand_query = hand_tokens
        query = hand_query
        if query_context is not None:
            if query_context.shape == (batch, self.token_channels):
                query_context = query_context[:, None]
            if query_context.shape not in (
                (batch, 1, self.token_channels), (batch, 6, self.token_channels)
            ):
                raise ValueError("query_context must be [BT,D], [BT,1,D] or [BT,6,D]")
            if not torch.isfinite(query_context).all():
                raise ValueError("query_context must be finite")
            query = query + query_context.to(query.dtype)

        with (
            torch.no_grad(),
            torch.autocast(device_type=pointcloud.device.type, enabled=False),
        ):
            members = index_points(pointcloud[..., :3], indices)
            valid_members = token_valid.unsqueeze(-1).expand_as(indices)
            if member_valid is not None:
                valid_members = valid_members & member_valid
            relative, distance, observed = nearest_patch_evidence(
                anchors, members, valid_members
            )
            center_valid = torch.isfinite(centers).all(-1) & token_valid
            safe_centers = torch.where(center_valid[..., None], centers.float(), 0.0)
            center_relative = safe_centers[:, None] - anchors[:, :, None]
            center_distance = center_relative.norm(dim=-1)
            context_valid = center_valid[:, None] & observed

        context, context_attention, _ = self.context(
            query, memory, center_relative, center_distance, context_valid
        )
        near, near_attention, support = self.near(
            query, memory, relative, distance, observed
        )
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
