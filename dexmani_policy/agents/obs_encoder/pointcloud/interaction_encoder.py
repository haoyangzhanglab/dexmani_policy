"""世界坐标系下的 PointPatch 交互编码，几何输入使用未归一化的米制值。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.pointcloud.scene_compressor import SceneSelfAttentionBlock
from dexmani_policy.agents.obs_encoder.proprio.hand_point_encoder import WristFingertipEncoder


def nearest_patch_evidence(
    anchors: torch.Tensor,
    members: torch.Tensor,
    member_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """最近有效成员相对锚点的世界轴向量、距离和有效位；无观测距离为 inf。"""
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


class GeometryCrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        sigmas: Sequence[float],
        metric_scale: float,
        proximity_scales: tuple[float, float],
        radius_multiple: float | None,
    ):
        super().__init__()
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
        # AMP 下 logits 和 softmax 仍使用 FP32。
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
    """将世界系 PointPatch 特征编码为 wrist + 五指的交互 token [B,6,D]。"""

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
        position_num_frequencies: int = 6,
        position_max_wavelength: float = 0.64,
    ):
        super().__init__()
        if token_channels <= 0 or num_heads <= 0 or token_channels % num_heads:
            raise ValueError("token_channels must be positive and divisible by num_heads")
        if len(context_sigmas) != num_heads or len(near_sigmas) != num_heads:
            raise ValueError("context_sigmas and near_sigmas must have length num_heads")
        hidden_dim = int(token_channels * mlp_ratio)
        if self_depth < 0 or hidden_dim < 1:
            raise ValueError("self_depth must be nonnegative and hidden_dim positive")
        scales = (metric_scale, position_scale, radius_multiple, *near_sigmas)
        if any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("scales and near_sigmas must be finite and positive")
        if any(math.isnan(s) or s <= 0 for s in context_sigmas):
            raise ValueError("context_sigmas must be positive; infinity is allowed")

        self.token_channels = token_channels
        self.metric_scale = metric_scale
        proximity_scales = (min(near_sigmas), max(near_sigmas))
        self.register_buffer("proximity_scales", torch.tensor(proximity_scales, dtype=torch.float32))
        self.hand_encoder = WristFingertipEncoder(
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
        """patches 来自 PointPatchEncoder(return_intermediate=True)。

        eef_pose = pos3 + rot6d；指尖顺序为 thumb/index/middle/ring/pinky。
        所有位置及相对向量均沿世界坐标轴；腕部朝向仅用于查询特征。
        patches 必须由同一世界系 pointcloud 提取，不在此组装点云编码器。
        member_valid 仅屏蔽几何成员；attention 最后一项为 null。
        """
        if pointcloud.ndim != 3 or pointcloud.shape[-1] < 3:
            raise ValueError("pointcloud must have shape [B,N,C>=3]")
        batch = pointcloud.shape[0]
        if eef_pose.shape != (batch, 9):
            raise ValueError("eef_pose must have shape [B,9] with the pointcloud batch size")
        global_memory, local_memory = patches["patch_token"], patches["local_token"]
        centers, indices = patches["patch_center"], patches["neighbor_idx"]
        if indices.ndim != 3 or indices.dtype != torch.long:
            raise ValueError("neighbor_idx must be a torch.long tensor with shape [B,M,K]")
        if indices.numel() == 0 or indices.min() < 0 or indices.max() >= pointcloud.shape[1]:
            raise ValueError("neighbor_idx contains empty or out-of-bounds indices")
        if member_valid is not None and (
            member_valid.shape != indices.shape or member_valid.dtype != torch.bool
        ):
            raise ValueError("member_valid must be boolean with shape [B,M,K]")

        hand = self.hand_encoder(eef_pose, fingertip_points)
        anchors = hand["hand_center"]
        query = hand["hand_token"]
        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            members = index_points(pointcloud[..., :3], indices)
            relative, distance, observed = nearest_patch_evidence(
                anchors, members, member_valid
            )
            center_valid = torch.isfinite(centers).all(-1)
            safe_centers = torch.where(center_valid[..., None], centers.float(), 0.0)
            center_delta = safe_centers[:, None] - anchors[:, :, None]
            center_relative = center_delta
            center_distance = center_delta.norm(dim=-1)
            context_valid = center_valid[:, None] & observed
            nearest = distance.min(-1).values
            has_observation = torch.isfinite(nearest)
            d_safe = nearest.nan_to_num(posinf=10 * self.metric_scale)
            proximity = torch.exp(-0.5 * (d_safe[..., None] / self.proximity_scales).square())
            gate_geometry = torch.cat(
                ((d_safe / self.metric_scale).clamp(max=10)[..., None],
                 proximity, has_observation[..., None].float()), dim=-1
            )

        if query_context is not None:
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


def example() -> None:
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

