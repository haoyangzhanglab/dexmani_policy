"""Hand-shape-conditioned wrist context and fingertip-relative surface geometry.

One shared scene attention reads encoded patches. A small shared point MLP
reads raw surface neighborhoods for the five fingers, without re-encoding the
scene. XYZ must share a metric frame; observed proximity is not physical contact.
"""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


@torch.no_grad()
def gather_surface_support(
    points: Tensor,
    anchors: Tensor,
    k: int,
    radius: float,
    metric_scale: float,
    point_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Gather observed neighbors; invalid/empty clouds have no near-field support.

    Geometry is observation-only (no coordinate gradients). A masked dummy point
    handles N=0 and fewer than k valid points without fake surface observations.
    """
    with torch.autocast(device_type=points.device.type, enabled=False):
        valid = torch.isfinite(points).all(-1)
        if point_mask is not None:
            valid = valid & point_mask
        points = torch.where(valid[..., None], points.float(), 0.0)
        points = torch.cat((points, points.new_zeros(points.shape[0], 1, 3)), 1)
        valid = torch.cat((valid, valid.new_zeros(valid.shape[0], 1)), 1)
        relative = points[:, None] - anchors.float()[:, :, None]
        distance2 = relative.square().sum(-1).masked_fill(~valid[:, None], torch.inf)
        distance2, indices = distance2.topk(min(k, points.shape[1]), largest=False, sorted=True)
        relative = relative.gather(2, indices[..., None].expand(-1, -1, -1, 3))
        distance = distance2.sqrt()
        observed = torch.isfinite(distance)
        local_valid = observed & (distance <= radius)
        has_reference = observed[..., 0]
        nearest_distance = distance[..., 0]
        nearest_relative = torch.where(has_reference[..., None], relative[..., 0, :], 0.0)
        safe_distance = torch.where(has_reference, nearest_distance, 0.0)
        # Bounded geometry retains a metric scale; missing != near/contact.
        denominator = metric_scale + safe_distance[..., None]
        proximity = torch.exp(-0.5 * (safe_distance / radius).square()) * has_reference
        near_supported = local_valid.any(-1)
        descriptor = torch.cat((
            nearest_relative / denominator,
            safe_distance[..., None] / denominator,
            proximity[..., None],
            has_reference[..., None].float(),
            near_supported[..., None].float(),
        ), -1)
        return {
            "relative": torch.where(local_valid[..., None], relative, 0.0),
            "valid": local_valid,
            "descriptor": descriptor,
            "nearest_distance": nearest_distance,
            "has_reference": has_reference,
            "near_supported": near_supported,
        }


class SpatialCrossAttention(nn.Module):
    """Shared metric context, with relative geometry in logits AND values.

    A fixed zero key/value null slot makes empty memory safe. Its attention mass
    is a model diagnostic, not a calibrated contact or visibility probability.
    """

    def __init__(self, channels: int, sigmas: Sequence[float], metric_scale: float):
        super().__init__()
        self.heads = len(sigmas)
        self.head_dim = channels // self.heads
        self.metric_scale = metric_scale
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        # Small per-edge bottleneck instead of a channels -> channels edge MLP.
        self.edge = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, channels + self.heads))
        self.out = nn.Linear(channels, channels)

    def forward(self, query: Tensor, memory: Tensor, relative: Tensor, valid: Tensor):
        batch, queries, channels = query.shape
        patches = memory.shape[1]
        # Clear masked NaN/Inf BEFORE LayerNorm, projections and weighted sums.
        memory = torch.where(valid[..., None], memory, 0.0).to(self.k.weight.dtype)
        query = self.q(self.query_norm(query.to(self.q.weight.dtype)))
        query = query.reshape(batch, queries, self.heads, self.head_dim)
        memory = self.memory_norm(memory)
        key = self.k(memory).reshape(batch, patches, self.heads, self.head_dim)
        value = self.v(memory).reshape(batch, patches, self.heads, self.head_dim)
        with torch.autocast(device_type=query.device.type, enabled=False):
            relative = torch.where(valid[:, None, :, None], relative.float(), 0.0)
            distance = relative.norm(dim=-1)
            geometry = torch.cat((relative, distance[..., None]), -1) / self.metric_scale
        edge = self.edge(geometry.to(self.edge[0].weight.dtype))
        bias, edge_value = edge.split((self.heads, channels), -1)
        edge_value = edge_value.reshape(batch, queries, patches, self.heads, self.head_dim)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bqhd,bmhd->bqhm", query.float(), key.float()) / math.sqrt(self.head_dim)
            ratios = distance[:, :, None] / self.sigmas.float()[None, None, :, None]
            logits = logits + bias.float().permute(0, 1, 3, 2) - 0.5 * ratios.square()
            logits = logits.masked_fill(~valid[:, None, None], -torch.inf)
            attention = torch.cat((logits, logits.new_zeros(batch, queries, self.heads, 1)), -1).softmax(-1)
            weights = attention[..., :-1]
            message = torch.einsum("bqhm,bmhd->bqhd", weights, value.float())
            message = message + torch.einsum("bqhm,bqmhd->bqhd", weights, edge_value.float())
        output = self.out(message.reshape(batch, queries, channels).to(self.out.weight.dtype))
        output = torch.where(valid.any(-1)[:, None, None], output, 0.0)
        return output, attention


class SurfaceSetEncoder(nn.Module):
    """Shared fingertip point MLP + masked mean/max; no attention or extra tokens."""

    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.mlp = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())

    def forward(self, relative: Tensor, valid: Tensor) -> Tensor:
        with torch.autocast(device_type=relative.device.type, enabled=False):
            relative = torch.where(valid[..., None], relative.float(), 0.0) / self.radius
            features = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
        point = self.mlp(features.to(self.mlp[0].weight.dtype))
        with torch.autocast(device_type=point.device.type, enabled=False):
            point = point.float()
            mean = point.masked_fill(~valid[..., None], 0.0).sum(-2)
            mean = mean / valid.sum(-1, keepdim=True).clamp_min(1)
            maximum = point.masked_fill(~valid[..., None], -torch.inf).amax(-2)
            maximum = torch.where(valid.any(-1, keepdim=True), maximum, 0.0)
        return torch.cat((mean, maximum), -1)


class HandSceneAttention(nn.Module):
    """Read wrist layout and fingertip surface relations in wrist/thumb/... order.

    Requires projected patch tokens [BT,M,D], xyz [BT,M,3], bool mask [BT,M].
    pointcloud [BT,N,C>=3] supplies raw geometry, independently of patch grouping.
    point_mask restricts that geometry only; it does NOT segment patch semantics.
    The caller must remove robot self-points/select objects when needed.
    Returns updates, not hand_query + updates; the caller owns the residual.
    """

    def __init__(
        self,
        token_channels: int = 192,
        context_sigmas: Sequence[float] = (0.08, 0.08, 0.20, float("inf")),
        metric_scale: float = 0.10,
        num_heads: int = 4,
        surface_k: int = 32,
        surface_radius: float = 0.04,
    ) -> None:
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (token_channels, num_heads, surface_k)):
            raise ValueError("token_channels, num_heads and surface_k must be positive integers")
        if token_channels % num_heads or len(context_sigmas) != num_heads:
            raise ValueError("channels must divide into heads; context_sigmas must have one entry per head")
        if any(not math.isfinite(x) or x <= 0 for x in (metric_scale, surface_radius)):
            raise ValueError("metric_scale and surface_radius must be finite and positive")
        if any(math.isnan(x) or x <= 0 for x in context_sigmas):
            raise ValueError("context_sigmas must be positive; infinity is allowed")
        self.token_channels = token_channels
        self.metric_scale = metric_scale
        self.surface_k = surface_k
        self.surface_radius = surface_radius
        self.context = SpatialCrossAttention(token_channels, context_sigmas, metric_scale)
        self.hand_shape = nn.Linear(15, token_channels, bias=False)
        self.surface = SurfaceSetEncoder(surface_radius)
        self.wrist_fusion = nn.Linear(token_channels + 7, token_channels)
        self.finger_fusion = nn.Linear(token_channels + 64 + 7, token_channels)

    def forward(
        self,
        pointcloud: Tensor,
        patches: Mapping[str, Tensor],
        anchor_xyz: Tensor,
        hand_tokens: Tensor,
        *,
        point_mask: Tensor | None = None,
        query_context: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        memory, centers, token_valid = patches["tokens"], patches["xyz"], patches["mask"]
        if memory.ndim != 3 or memory.shape[-1] != self.token_channels or memory.shape[0] == 0:
            raise ValueError("patch tokens must be nonempty-batch [BT,M,D]")
        batch, patch_count, _ = memory.shape
        if centers.shape != (batch, patch_count, 3):
            raise ValueError("patch xyz must be [BT,M,3]")
        if token_valid.shape != (batch, patch_count) or token_valid.dtype != torch.bool:
            raise ValueError("patch mask must be bool [BT,M]")
        if pointcloud.ndim != 3 or pointcloud.shape[0] != batch or pointcloud.shape[-1] < 3:
            raise ValueError("pointcloud must be [BT,N,C>=3]")
        if anchor_xyz.shape != (batch, 6, 3) or hand_tokens.shape != (batch, 6, self.token_channels):
            raise ValueError("anchors/hand_tokens must be [BT,6,3] / [BT,6,D]")
        if any(not x.is_floating_point() for x in (memory, centers, pointcloud, anchor_xyz, hand_tokens)):
            raise TypeError("features and XYZ must be floating point")
        if any(x.device != memory.device for x in (centers, token_valid, pointcloud, anchor_xyz, hand_tokens)):
            raise ValueError("inputs must share a device")
        if not torch.isfinite(anchor_xyz).all() or not torch.isfinite(hand_tokens).all():
            raise ValueError("anchors and hand_tokens must be finite")
        if point_mask is not None and (
            point_mask.shape != pointcloud.shape[:2] or point_mask.dtype != torch.bool or point_mask.device != memory.device
        ):
            raise ValueError("point_mask must be bool [BT,N] on the input device")
        query = hand_tokens.float()
        if query_context is not None:
            if query_context.shape == (batch, self.token_channels):
                query_context = query_context[:, None]
            if query_context.shape not in ((batch, 1, self.token_channels), (batch, 6, self.token_channels)):
                raise ValueError("query_context must be [BT,D], [BT,1,D] or [BT,6,D]")
            if query_context.device != memory.device or not query_context.is_floating_point() or not torch.isfinite(query_context).all():
                raise ValueError("query_context must be finite floating point on the input device")
            query = query + query_context.float()
        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            anchors = anchor_xyz.float()
            shape = ((anchors[:, 1:] - anchors[:, :1]) / self.metric_scale).flatten(1)
            valid = token_valid & torch.isfinite(centers).all(-1) & torch.isfinite(memory).all(-1)
            centers = torch.where(valid[..., None], centers.float(), 0.0)
            relative = centers[:, None] - anchors[:, :, None]
        shape = self.hand_shape(shape.to(self.hand_shape.weight.dtype)).float()
        query = torch.cat((query[:, :1] + shape[:, None], query[:, 1:]), 1)
        context, attention = self.context(query, memory, relative, valid)
        support = gather_surface_support(
            pointcloud[..., :3], anchors, self.surface_k, self.surface_radius, self.metric_scale, point_mask,
        )
        geometry = support["descriptor"]
        local = self.surface(support["relative"][:, 1:], support["valid"][:, 1:])
        wrist = self.wrist_fusion(torch.cat((context[:, :1].float(), geometry[:, :1]), -1).to(self.wrist_fusion.weight.dtype))
        finger = self.finger_fusion(torch.cat((context[:, 1:].float(), local, geometry[:, 1:]), -1).to(self.finger_fusion.weight.dtype))
        has_evidence = valid.any(-1)[:, None] | support["has_reference"]
        update = torch.where(has_evidence[..., None], torch.cat((wrist, finger), 1), 0.0)
        output = {
            "hand_query": hand_tokens,
            "relation_update": update,
            "wrist_relation": update[:, :1],
            "finger_relation": update[:, 1:],
            "interaction_center": anchors,
            "geometry_descriptor": geometry,
            "min_observed_distance": support["nearest_distance"],
            "has_reference": support["has_reference"],
            "near_has_support": support["near_supported"],
        }
        if return_intermediate:
            output.update(
                context_update=context, context_attention=attention,
                context_null_mass=attention[..., -1], finger_local=local,
                surface_relative=support["relative"], surface_valid=support["valid"],
            )
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
