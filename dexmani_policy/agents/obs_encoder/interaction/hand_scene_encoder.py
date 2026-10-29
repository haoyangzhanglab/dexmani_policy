"""Hand-scene encoding with geometric attention and local surface features."""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


@torch.no_grad()
def query_surface_neighbors(
    points: Tensor,
    anchors: Tensor,
    k: int,
    radius: float,
    metric_scale: float,
    point_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Query surface neighborhoods and proximity descriptors without coordinate gradients."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        valid = torch.isfinite(points).all(-1)
        if point_mask is not None:
            valid = valid & point_mask
        points = torch.where(valid[..., None], points.float(), 0.0)
        anchors = anchors.float()

        # A masked sentinel keeps empty clouds and top-k reductions well-defined.
        points = torch.cat((points, points.new_zeros(points.shape[0], 1, 3)), 1)
        valid = torch.cat((valid, valid.new_zeros(valid.shape[0], 1)), 1)

        # Accumulate squared offsets without a full [B,Q,N,3] temporary.
        distance2 = (points[:, None, :, 0] - anchors[:, :, None, 0]).square_()
        for axis in (1, 2):
            distance2.add_((points[:, None, :, axis] - anchors[:, :, None, axis]).square_())
        distance2.masked_fill_(~valid[:, None], torch.inf)
        distance2, indices = distance2.topk(min(k, points.shape[1]), largest=False)
        distance = distance2.sqrt()
        batch_indices = torch.arange(points.shape[0], device=points.device)[:, None, None]
        relative = points[batch_indices, indices] - anchors[:, :, None]

        nearest_distance = distance[..., 0]
        has_reference = torch.isfinite(nearest_distance)
        local_valid = distance <= radius
        near_supported = local_valid[..., 0]
        nearest_relative = torch.where(has_reference[..., None], relative[..., 0, :], 0.0)
        safe_distance = torch.where(has_reference, nearest_distance, 0.0)
        denominator = metric_scale + safe_distance[..., None]
        proximity = torch.exp(-0.5 * (safe_distance / radius).square()) * has_reference
        descriptor = torch.cat(
            (
                nearest_relative / denominator,
                safe_distance[..., None] / denominator,
                proximity[..., None],
                has_reference[..., None].float(),
                near_supported[..., None].float(),
            ),
            -1,
        )

        return {
            "relative": torch.where(local_valid[..., None], relative, 0.0),
            "valid": local_valid,
            "descriptor": descriptor,
            "nearest_distance": nearest_distance,
            "has_reference": has_reference,
            "near_supported": near_supported,
        }


class GeometricCrossAttention(nn.Module):
    """Cross-attend with EEF-frame offsets; invalid directions are already zeroed."""

    def __init__(self, channels: int, sigmas: Sequence[float], metric_scale: float) -> None:
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
        self.edge = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, channels + self.heads))
        self.out = nn.Linear(channels, channels)

    def forward(
        self, query: Tensor, memory: Tensor, relative: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, queries, channels = query.shape
        patches = memory.shape[1]

        # Remove masked NaN/Inf before normalization and projections.
        memory = torch.where(valid[..., None], memory, 0.0).to(self.k.weight.dtype)
        query = self.q(self.query_norm(query.to(self.q.weight.dtype)))
        query = query.reshape(batch, queries, self.heads, self.head_dim)
        memory = self.memory_norm(memory)
        key = self.k(memory).reshape(batch, patches, self.heads, self.head_dim)
        value = self.v(memory).reshape(batch, patches, self.heads, self.head_dim)

        with torch.autocast(device_type=query.device.type, enabled=False):
            relative = relative.float()
            distance = relative.norm(dim=-1)
            geometry = torch.cat((relative, distance[..., None]), -1) / self.metric_scale
        edge = self.edge(geometry.to(self.edge[0].weight.dtype))
        bias, edge_value = edge.split((self.heads, channels), -1)
        edge_value = edge_value.reshape(batch, queries, patches, self.heads, self.head_dim)

        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bqhd,bmhd->bqhm", query.float(), key.float())
            ratios = distance[:, :, None] / self.sigmas.float()[None, None, :, None]
            logits = logits / math.sqrt(self.head_dim) + bias.float().permute(0, 1, 3, 2)
            logits = logits - 0.5 * ratios.square()
            logits = logits.masked_fill(~valid[:, None, None], -torch.inf)

            # A zero key/value slot gives empty scenes a finite softmax.
            null_logits = logits.new_zeros(batch, queries, self.heads, 1)
            attention = torch.cat((logits, null_logits), -1).softmax(-1)
            weights = attention[..., :-1]
            message = torch.einsum("bqhm,bmhd->bqhd", weights, value.float())
            message = message + torch.einsum("bqhm,bqmhd->bqhd", weights, edge_value.float())

        output = self.out(message.reshape(batch, queries, channels).to(self.out.weight.dtype))
        output = torch.where(valid.any(-1)[:, None, None], output, 0.0)
        return output, attention


class LocalSurfaceEncoder(nn.Module):
    """Encode EEF-frame neighborhoods; invalid coordinates are zeroed by the query."""

    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius
        self.mlp = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())

    def forward(self, relative: Tensor, valid: Tensor) -> Tensor:
        with torch.autocast(device_type=relative.device.type, enabled=False):
            relative = relative.float() / self.radius
            features = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
        features = self.mlp(features.to(self.mlp[0].weight.dtype))

        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            count = valid.sum(-1, keepdim=True)
            mean = features.masked_fill(~valid[..., None], 0.0).sum(-2) / count.clamp_min(1)
            maximum = features.masked_fill(~valid[..., None], -torch.inf).amax(-2)
            maximum = torch.where(count > 0, maximum, 0.0)

        return torch.cat((mean, maximum), -1)


class HandSceneEncoder(nn.Module):
    """Fuse hand-conditioned scene context and wrist/fingertip surface geometry."""

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
        if min(token_channels, num_heads, surface_k) < 1:
            raise ValueError("token_channels, num_heads and surface_k must be positive")
        if token_channels % num_heads or len(context_sigmas) != num_heads:
            raise ValueError("channels must be divisible by heads, with one sigma per head")
        if any(not math.isfinite(x) or x <= 0 for x in (metric_scale, surface_radius)):
            raise ValueError("metric_scale and surface_radius must be finite and positive")
        if any(math.isnan(x) or x <= 0 for x in context_sigmas):
            raise ValueError("context_sigmas must be positive; infinity is allowed")

        self.token_channels = token_channels
        self.metric_scale = metric_scale
        self.surface_k = surface_k
        self.surface_radius = surface_radius
        self.context = GeometricCrossAttention(token_channels, context_sigmas, metric_scale)
        self.hand_shape = nn.Linear(15, token_channels, bias=False)
        self.surface = LocalSurfaceEncoder(surface_radius)
        self.wrist_fusion = nn.Linear(token_channels + 7, token_channels)
        self.finger_fusion = nn.Linear(token_channels + 64 + 7, token_channels)

    def forward(
        self,
        pointcloud: Tensor,
        patches: Mapping[str, Tensor],
        anchor_xyz: Tensor,
        hand_tokens: Tensor,
        *,
        rotation_world_from_eef: Tensor,
        point_mask: Tensor | None = None,
        query_context: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """Return scene increments alongside the original hand_tokens as hand_query.

        Patches contain tokens [N,M,D], xyz [N,M,3] and bool mask [N,M], as
        returned by Sparse3DEncoder; token width must match hand_tokens.
        Anchors, patch centers and pointcloud XYZ remain in world meters. The
        measured [N,3,3] rotation maps EEF axes into the world; all relative
        directions, including geometry_descriptor[..., :3], use EEF axes.
        GatedTactileFusion applies the scene and tactile residuals exactly once.
        """
        memory, centers, token_valid = patches["tokens"], patches["xyz"], patches["mask"]
        batch, channels = memory.shape[0], self.token_channels
        if anchor_xyz.shape != (batch, 6, 3) or hand_tokens.shape != (batch, 6, channels):
            raise ValueError("anchor_xyz/hand_tokens must be [N,6,3] / [N,6,D]")
        if rotation_world_from_eef.shape != (batch, 3, 3):
            raise ValueError("rotation_world_from_eef must be [N,3,3]")

        with torch.no_grad(), torch.autocast(device_type=pointcloud.device.type, enabled=False):
            anchors = anchor_xyz.float()
            rotation = rotation_world_from_eef.float()
            shape = torch.einsum("n...c,ncd->n...d", anchors[:, 1:] - anchors[:, :1], rotation)
            shape = (shape / self.metric_scale).flatten(1)
            valid = token_valid & torch.isfinite(centers).all(-1) & torch.isfinite(memory).all(-1)
            patch_relative = torch.where(
                valid[:, None, :, None], centers.float()[:, None] - anchors[:, :, None], 0.0
            )
            patch_relative = torch.einsum(
                "n...c,ncd->n...d", patch_relative, rotation
            )
            support = query_surface_neighbors(
                pointcloud[..., :3], anchors, self.surface_k, self.surface_radius,
                self.metric_scale, point_mask,
            )
            surface_relative = torch.einsum("n...c,ncd->n...d", support["relative"], rotation)
            geometry = torch.cat(
                (torch.einsum("n...c,ncd->n...d", support["descriptor"][..., :3], rotation),
                 support["descriptor"][..., 3:]),
                -1,
            )
            has_evidence = valid.any(-1)[:, None] | support["has_reference"]

        query = hand_tokens.float()
        if query_context is not None:
            if query_context.ndim == 2:
                query_context = query_context[:, None]
            if query_context.shape not in ((batch, 1, channels), (batch, 6, channels)):
                raise ValueError("query_context must be [N,D], [N,1,D] or [N,6,D]")
            query = query + query_context.float()

        shape = self.hand_shape(shape.to(self.hand_shape.weight.dtype)).float()
        query = torch.cat((query[:, :1] + shape[:, None], query[:, 1:]), 1)
        context, attention = self.context(query, memory, patch_relative, valid)
        local = self.surface(surface_relative[:, 1:], support["valid"][:, 1:])

        wrist = torch.cat((context[:, :1].float(), geometry[:, :1]), -1)
        finger = torch.cat((context[:, 1:].float(), local, geometry[:, 1:]), -1)
        wrist = self.wrist_fusion(wrist.to(self.wrist_fusion.weight.dtype))
        finger = self.finger_fusion(finger.to(self.finger_fusion.weight.dtype))
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
                context_update=context,
                context_attention=attention,
                context_null_mass=attention[..., -1],
                finger_local=local,
                surface_relative=surface_relative,
                surface_valid=support["valid"],
            )
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
