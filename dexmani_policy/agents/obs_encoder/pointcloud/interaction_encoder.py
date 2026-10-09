"""身体锚定的点云交互编码：宽域上下文与多尺度近表面读取。

几何输入使用同一坐标系、单位米；eef_pose=[pos3, rot6d]，rot6d 为旋转
矩阵前两列拼接。五指顺序为 thumb/index/middle/ring/pinky。

与 PointPatchEncoder(return_intermediate=True) 配合：
    encoder = PointPatchInteractionEncoder(token_channels=192)
    interaction = encoder(pointcloud, patches, eef_pose, fingertip_points)

输出 interaction_token [B,6,D] 与 interaction_center [B,6,3]。B 可合并
时间维。距离仅表示锚点到已观测点的关系，不是 SDF、接触概率或接触力。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.pointcloud.scene_compressor import SceneSelfAttentionBlock


def _rotation_from_columns_6d(rotation: torch.Tensor) -> torch.Tensor:
    """[B,6] -> [B,3,3]；输入前后两个三维向量分别为前两列。"""
    if rotation.ndim != 2 or rotation.shape[-1] != 6:
        raise ValueError("rotation must have shape [B,6]")
    rotation = rotation.float()
    first, second = rotation[..., :3], rotation[..., 3:]
    if not torch.isfinite(rotation).all() or (first.norm(dim=-1) < 1e-6).any():
        raise ValueError("rotation contains invalid first columns")
    x = F.normalize(first, dim=-1)
    y = second - (second * x).sum(-1, keepdim=True) * x
    if (y.norm(dim=-1) < 1e-6).any():
        raise ValueError("rotation columns must be linearly independent")
    y = F.normalize(y, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


def _nearest_patch_evidence(
    anchors: torch.Tensor,
    members: torch.Tensor,
    rotation: torch.Tensor,
    member_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """找到每个身体锚点在每个 patch 内最近的有效观测点。

    anchors: [B,Q,3]，members: [B,M,K,3]，rotation: [B,3,3]。
    返回 eef 坐标系相对向量 [B,Q,M,3]、米制距离 [B,Q,M]、有效位。
    无有效点时相对向量为零、距离为 inf。等距离点按输入顺序打破平局。
    对采样表面的离散最近邻求值，不对 argmin 的选择过程求导。
    """
    if members.shape[2] == 0:
        raise ValueError("patches must contain at least one member slot")
    valid = torch.isfinite(members).all(-1)
    if member_valid is not None:
        valid = valid & member_valid.bool()
    safe_members = torch.where(valid[..., None], members.float(), 0.0)
    delta = safe_members[:, None] - anchors.float()[:, :, None, None]
    squared = delta.square().sum(-1).masked_fill(~valid[:, None], float("inf"))
    squared_min, index = squared.min(dim=-1)
    observed = torch.isfinite(squared_min)
    selected = delta.gather(3, index[..., None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    selected = torch.where(observed[..., None], selected, 0.0)
    # 行向量 delta @ R 等价于列向量 R^T delta。
    relative = torch.einsum("bqmc,bcd->bqmd", selected, rotation.float())
    return relative, squared_min.sqrt(), observed


class GeometryCrossAttention(nn.Module):
    """几何同时影响 attention logits 和 value，附带一个可学习的 null 项。"""

    def __init__(
        self,
        channels: int,
        sigmas: Sequence[float],
        metric_scale: float,
        proximity_scales: tuple[float, float],
        radius_multiple: float | None,
    ):
        super().__init__()
        if channels <= 0 or not sigmas or channels % len(sigmas):
            raise ValueError("channels must be divisible by the number of sigmas/heads")
        if any(math.isnan(s) or s <= 0 for s in sigmas):
            raise ValueError("sigmas must be positive; infinity is allowed for a global head")
        if radius_multiple is not None and (not math.isfinite(radius_multiple) or radius_multiple <= 0):
            raise ValueError("radius_multiple must be finite and positive")
        self.channels = channels
        self.heads = len(sigmas)
        self.head_dim = channels // self.heads
        self.metric_scale = metric_scale
        self.radius_multiple = radius_multiple
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.register_buffer("proximity_scales", torch.tensor(proximity_scales, dtype=torch.float32))
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.edge = nn.Sequential(nn.Linear(6, channels), nn.GELU())
        self.edge_bias = nn.Linear(channels, self.heads)
        self.edge_value = nn.Linear(channels, channels)
        self.null_key = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.out = nn.Linear(channels, channels)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        relative: torch.Tensor,
        distance: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, queries, channels = query.shape
        patches = memory.shape[1]
        q = self.q(self.query_norm(query)).reshape(batch, queries, self.heads, self.head_dim)
        memory = self.memory_norm(memory)
        k = self.k(memory).reshape(batch, patches, self.heads, self.head_dim)
        v = self.v(memory).reshape(batch, patches, self.heads, self.head_dim)
        d = torch.where(valid, distance.float(), 0.0)
        delta = torch.where(valid[..., None], relative.float(), 0.0)
        proximity = torch.exp(-0.5 * (d[..., None] / self.proximity_scales).square())
        geometry = torch.cat((delta / self.metric_scale, d[..., None] / self.metric_scale, proximity), -1)
        edge = self.edge(geometry.to(memory.dtype))
        edge_value = self.edge_value(edge).reshape(batch, queries, patches, self.heads, self.head_dim)

        edge_bias = self.edge_bias(edge).permute(0, 1, 3, 2)
        # 显式关闭 autocast：仅 .float() 不能阻止 einsum 被 AMP 降精度。
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bqhd,bmhd->bqhm", q.float(), k.float())
            logits = logits / math.sqrt(self.head_dim) + edge_bias.float()
            ratios = d[:, :, None, :] / self.sigmas.float()[None, None, :, None]
            logits = logits - 0.5 * ratios.square()
            allowed = valid[:, :, None, :].expand(-1, -1, self.heads, -1)
            if self.radius_multiple is not None:
                allowed = allowed & (ratios <= self.radius_multiple)
            logits = logits.masked_fill(~allowed, float("-inf"))
            null_logit = torch.einsum("bqhd,hd->bqh", q.float(), self.null_key.float())
            null_logit = null_logit / math.sqrt(self.head_dim)
            attention = torch.cat((logits, null_logit[..., None]), -1).softmax(-1)
        real_weight = attention[..., :-1].to(v.dtype)
        semantic = torch.einsum("bqhm,bmhd->bqhd", real_weight, v)
        relational = torch.einsum("bqhm,bqmhd->bqhd", real_weight, edge_value)
        null = attention[..., -1:].to(v.dtype) * self.null_value[None, None].to(v.dtype)
        output = self.out((semantic + relational + null).reshape(batch, queries, channels))
        return output, attention, allowed.any(-1)


class PointPatchInteractionEncoder(nn.Module):
    """6 个查询：eef + 五指；宽域上下文 + 多尺度近表面证据。

    near sigmas 默认 1.5/4 cm，各两个头；它们是未调优的初始化参数。
    局部分支使用 3*sigma 的截断半径。宽域分支保留一个无距离衰减的头。
    gate 表示局部分支的学习权重；null_mass 表示拒绝读取权重，二者均不
    是接触概率或经过校准的可见性概率。

    member_valid 仅标记原始采样中有效的成员；若用于去除机器人或目标分割，
    必须同步重新生成 local_token，不能用它清除已经混入特征的手/背景信息。
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
        mlp_ratio: float = 2.0,
        self_depth: int = 1,
    ):
        super().__init__()
        if token_channels <= 0 or num_heads <= 0 or token_channels % num_heads:
            raise ValueError("token_channels must be positive and divisible by num_heads")
        if len(context_sigmas) != num_heads or len(near_sigmas) != num_heads:
            raise ValueError("context_sigmas and near_sigmas must have length num_heads")
        if not math.isfinite(mlp_ratio) or int(token_channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be finite and produce a positive hidden dimension")
        if self_depth < 0:
            raise ValueError("self_depth must be nonnegative")
        hidden_dim = int(token_channels * mlp_ratio)
        if not math.isfinite(metric_scale) or metric_scale <= 0:
            raise ValueError("metric_scale must be finite and positive")
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")
        if not near_sigmas or any(not math.isfinite(s) or s <= 0 for s in near_sigmas):
            raise ValueError("near_sigmas must be finite and positive")
        self.token_channels = token_channels
        self.metric_scale = metric_scale
        self.position_scale = position_scale
        proximity_scales = (min(near_sigmas), max(near_sigmas))
        self.register_buffer("proximity_scales", torch.tensor(proximity_scales, dtype=torch.float32))
        self.body_identity = nn.Parameter(torch.randn(6, token_channels) * 0.02)
        self.position = nn.Sequential(
            nn.Linear(3, token_channels), nn.GELU(), nn.Linear(token_channels, token_channels)
        )
        self.context = GeometryCrossAttention(
            token_channels, context_sigmas, metric_scale, proximity_scales, None
        )
        self.near = GeometryCrossAttention(
            token_channels, near_sigmas, metric_scale, proximity_scales, radius_multiple
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * token_channels + 4, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.blocks = nn.ModuleList(
            [SceneSelfAttentionBlock(token_channels, num_heads, hidden_dim) for _ in range(self_depth)]
        )
        self.output_norm = nn.LayerNorm(token_channels)

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
        """输入的 batch 维度可由 [B,T] flatten 得到；所有模态必须同步。

        pointcloud [BT,N,>=3]；eef_pose [BT,9]；fingertip_points [BT,5,3]。
        patches 包含 patch_token/local_token [BT,M,D]、patch_center [BT,M,3]、
        neighbor_idx [BT,M,K]。可选 query_context [BT,D] 为已有任务/状态条件。
        return_intermediate=True 时额外返回距离、掩码、gate 和逐头 attention；
        attention 最后一项为 null，无观测时距离为 inf。完整网络不保证 SE(3)
        不变性，只有相对几何在同步变换坐标系时保持一致。
        """
        if pointcloud.ndim != 3 or pointcloud.shape[-1] < 3:
            raise ValueError("pointcloud must have shape [B,N,C>=3]")
        batch = pointcloud.shape[0]
        if batch == 0 or pointcloud.shape[1] == 0:
            raise ValueError("pointcloud must have nonempty batch and point dimensions")
        if eef_pose.shape != (batch, 9) or fingertip_points.shape != (batch, 5, 3):
            raise ValueError("eef_pose must be [B,9] and fingertip_points [B,5,3]")
        for key in ("patch_token", "local_token", "patch_center", "neighbor_idx"):
            if key not in patches:
                raise ValueError(f"missing {key}; use PointPatchEncoder(return_intermediate=True)")
        global_memory, local_memory = patches["patch_token"], patches["local_token"]
        if (global_memory.ndim != 3 or global_memory.shape[0] != batch
                or global_memory.shape[2] != self.token_channels):
            raise ValueError("patch_token must have shape [B,M,token_channels]")
        if local_memory.shape != global_memory.shape:
            raise ValueError("local_token and patch_token must have equal shapes")
        if global_memory.shape[1] == 0:
            raise ValueError("at least one patch is required")
        tensors = (global_memory, local_memory, eef_pose, fingertip_points)
        if any(t.device != pointcloud.device for t in tensors):
            raise ValueError("pointcloud, patch tokens and body poses must share one device")
        if any(not t.is_floating_point() for t in (pointcloud, *tensors)):
            raise ValueError("coordinates and patch features must be floating-point tensors")
        if local_memory.dtype != global_memory.dtype:
            raise ValueError("local_token and patch_token must have equal dtypes")
        if not torch.isfinite(global_memory).all() or not torch.isfinite(local_memory).all():
            raise ValueError("patch features must be finite; clean invalid raw points before point_patch")
        centers, indices = patches["patch_center"], patches["neighbor_idx"]
        if centers.device != pointcloud.device or indices.device != pointcloud.device:
            raise ValueError("patch centers and indices must share the pointcloud device")
        if not centers.is_floating_point():
            raise ValueError("patch_center must be floating point")
        if centers.shape != (*global_memory.shape[:2], 3) or indices.shape[:2] != global_memory.shape[:2]:
            raise ValueError("patch centers and neighbor indices do not match patch tokens")
        if indices.ndim != 3 or indices.dtype != torch.long:
            raise ValueError("neighbor_idx must be a torch.long tensor with shape [B,M,K]")
        if indices.numel() == 0 or indices.min() < 0 or indices.max() >= pointcloud.shape[1]:
            raise ValueError("neighbor_idx contains empty or out-of-bounds indices")
        if member_valid is not None and member_valid.shape != indices.shape:
            raise ValueError("member_valid must have shape [B,M,K]")
        if member_valid is not None and (
            member_valid.device != pointcloud.device or member_valid.dtype != torch.bool
        ):
            raise ValueError("member_valid must be a boolean mask on the pointcloud device")
        anchors = torch.cat((eef_pose[:, None, :3], fingertip_points), 1).float()
        if not torch.isfinite(anchors).all():
            raise ValueError("body anchors must be finite; use explicit modality validity, not zero-dropout")
        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            rotation = _rotation_from_columns_6d(eef_pose[:, 3:])
            members = index_points(pointcloud[..., :3], indices)
            relative, distance, observed = _nearest_patch_evidence(
                anchors, members, rotation, member_valid
            )
            center_valid = torch.isfinite(centers).all(-1)
            safe_centers = torch.where(center_valid[..., None], centers.float(), 0.0)
            center_delta = safe_centers[:, None] - anchors[:, :, None]
            center_relative = torch.einsum("bqmc,bcd->bqmd", center_delta, rotation)
            center_distance = center_delta.norm(dim=-1)
            # 全空 patch 的几何和特征都不可作为证据使用。
            context_valid = center_valid[:, None] & observed
            nearest = distance.min(-1).values
            has_observation = torch.isfinite(nearest)
            d_safe = nearest.nan_to_num(posinf=10 * self.metric_scale)
            proximity = torch.exp(-0.5 * (d_safe[..., None] / self.proximity_scales).square())
            gate_geometry = torch.cat(
                ((d_safe / self.metric_scale).clamp(max=10)[..., None],
                 proximity, has_observation[..., None].float()), dim=-1
            )

        position = (anchors / self.position_scale).to(self.position[0].weight.dtype)
        query = self.body_identity[None] + self.position(position)
        if query_context is not None:
            if query_context.shape != (batch, self.token_channels):
                raise ValueError("query_context must have shape [B,token_channels]")
            if query_context.device != query.device or not query_context.is_floating_point():
                raise ValueError("query_context must be floating point on the query device")
            if not torch.isfinite(query_context).all():
                raise ValueError("query_context must be finite")
            query = query + query_context[:, None].to(query.dtype)
        context, context_attention, _ = self.context(
            query, global_memory, center_relative, center_distance, context_valid
        )
        near, near_attention, support = self.near(query, local_memory, relative, distance, observed)
        gate_input = torch.cat((query, context, near, gate_geometry.to(query.dtype)), -1)
        near_gate = self.gate(gate_input).sigmoid() * support.any(-1, keepdim=True).to(query.dtype)
        tokens = query + context + near_gate * near
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        outputs = {"interaction_token": tokens, "interaction_center": anchors}
        if return_intermediate:
            outputs.update(
                near_gate=near_gate.squeeze(-1),
                min_observed_distance=nearest,
                near_has_support=support,
                near_null_mass=near_attention[..., -1],
                context_null_mass=context_attention[..., -1],
                near_attention=near_attention,
                context_attention=context_attention,
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


def example() -> None:
    """仅依赖 PyTorch 的合成输入前向示例，不触发 FPS 或预训练模型。"""
    torch.manual_seed(0)
    batch_size, num_points, num_patches, group_size, channels = 2, 64, 16, 8, 192
    pointcloud = torch.rand(batch_size, num_points, 3) * 0.2
    indices = torch.randint(num_points, (batch_size, num_patches, group_size))
    patches = {
        "patch_token": torch.randn(batch_size, num_patches, channels),
        "local_token": torch.randn(batch_size, num_patches, channels),
        "patch_center": index_points(pointcloud, indices[..., 0]),
        "neighbor_idx": indices,
    }
    eef_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    eef_pose = eef_pose.expand(batch_size, -1)
    fingertips = torch.rand(batch_size, 5, 3) * 0.1
    encoder = PointPatchInteractionEncoder(token_channels=channels).eval()
    with torch.no_grad():
        outputs = encoder(pointcloud, patches, eef_pose, fingertips, return_intermediate=True)
    for name, value in outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("out_shape:", encoder.out_shape)
    print("parameters:", sum(p.numel() for p in encoder.parameters()))


if __name__ == "__main__":
    example()
