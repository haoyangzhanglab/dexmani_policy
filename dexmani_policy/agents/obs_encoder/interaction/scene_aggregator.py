"""Geometry-Aware Scene Aggregation."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


@torch.no_grad()
def sample_region_seeds(points: Tensor, count: int, mask: Tensor | None = None) -> Tensor:
    """Centroid-initialized FPS with unique indices; masked coordinates must already be zero."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.float()
        excluded = torch.zeros(points.shape[:2], device=points.device, dtype=torch.bool) if mask is None else ~mask
        valid_count = points.shape[1] if mask is None else mask.sum(1)[:, None, None]
        centroid = points.sum(1, keepdim=True) / valid_count
        distance = (points - centroid).square().sum(-1)
        farthest = distance.masked_fill(excluded, -torch.inf).argmax(-1)
        batch = torch.arange(points.shape[0], device=points.device)
        nearest = torch.full_like(distance, torch.inf)
        seeds = []

        for step in range(count):
            seeds.append(farthest)
            if step + 1 < count:
                excluded.scatter_(1, farthest[:, None], True)
                distance = (points - points[batch, farthest, None]).square().sum(-1)
                nearest = torch.minimum(nearest, distance)
                farthest = nearest.masked_fill(excluded, -torch.inf).argmax(-1)

        return torch.stack(seeds, 1)


class RegionSelfAttention(nn.Module):
    """Region self-attention with relative geometry in both attention biases and values."""

    def __init__(self, channels: int, attention_dim: int, heads: int, scale: float) -> None:
        super().__init__()
        if min(channels, attention_dim, heads) < 1:
            raise ValueError("relation dimensions must be positive")
        if attention_dim % heads or attention_dim // heads < 2:
            raise ValueError("relation head dimension must be an integer >= 2")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("relation scale must be finite and positive")

        self.heads = heads
        self.head_dim = attention_dim // heads
        self.scale = scale
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * attention_dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.edge = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads + attention_dim))
        self.out = nn.Linear(attention_dim, channels)
        self.gain = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, tokens: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
        batch, regions, _ = tokens.shape
        qkv = self.qkv(self.norm(tokens)).reshape(batch, regions, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)

        with torch.autocast(device_type=tokens.device.type, enabled=False):
            relative = (centers[:, None].float() - centers[:, :, None].float()) / self.scale
            edge_input = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
        edge = self.edge(edge_input.to(self.edge[0].weight.dtype))
        bias, edge_value = edge.split((self.heads, self.heads * self.head_dim), -1)
        edge_value = edge_value.reshape(batch, regions, regions, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)

        with torch.autocast(device_type=tokens.device.type, enabled=False):
            logits = query.float() @ key.float().transpose(-2, -1)
            logits = logits / math.sqrt(self.head_dim) + bias.float().permute(0, 3, 1, 2)
            attention = logits.softmax(-1)
            message = attention @ value.float()
            message = message + (attention[..., None] * edge_value.float()).sum(-2)
            message = message.transpose(1, 2).reshape(batch, regions, -1)

        update = self.out(message.to(self.out.weight.dtype))
        return tokens + self.gain.to(tokens.dtype) * update.to(tokens.dtype), attention


class RoutingPriorCrossAttention(nn.Module):
    """Refine geometrically routed regions using content-aware patch attention."""

    def __init__(self, channels: int, attention_dim: int, heads: int) -> None:
        super().__init__()
        if min(channels, attention_dim, heads) < 1 or attention_dim % heads:
            raise ValueError("attention_dim must be positive and divisible by heads")

        self.heads = heads
        self.head_dim = attention_dim // heads
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, attention_dim, bias=False)
        self.kv = nn.Linear(channels, 2 * attention_dim, bias=False)
        self.out = nn.Linear(attention_dim, channels)

    def forward(self, query: Tensor, memory: Tensor, prior: Tensor) -> Tensor:
        batch, regions, _ = query.shape
        patches = memory.shape[1]
        query = self.q(self.query_norm(query.to(self.q.weight.dtype)))
        key, value = self.kv(self.memory_norm(memory.to(self.kv.weight.dtype))).chunk(2, dim=-1)
        query = query.reshape(batch, regions, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)

        # Routing is trained through pooled tokens; detach its prior for efficient SDPA.
        weight = prior.detach().float()
        bias = weight.clamp_min(1e-30).log().masked_fill(weight <= 0, -torch.inf)
        refined = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias[:, None].to(query.dtype), dropout_p=0.0
        )
        return self.out(refined.transpose(1, 2).reshape(batch, regions, -1))

class SceneAggregator(nn.Module):
    """Compress spatial patches into geometric regions, then encode inter-region relations."""

    def __init__(
        self,
        token_channels: int = 192,
        num_scene_tokens: int = 12,
        *,
        route_dim: int = 64,
        routing_steps: int = 2,
        nearest_slots: int = 3,
        geometry_scale: float = 0.10,
        position_scale: float = 1.0,
        use_relation: bool = True,
        relation_dim: int = 64,
        relation_heads: int = 4,
        mlp_ratio: float = 2.0,
        use_cross_refinement: bool = False,
        refinement_dim: int = 128,
        refinement_heads: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (
            ("token_channels", token_channels),
            ("num_scene_tokens", num_scene_tokens),
            ("route_dim", route_dim),
            ("routing_steps", routing_steps),
            ("nearest_slots", nearest_slots),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("geometry_scale", geometry_scale), ("position_scale", position_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        hidden = int(token_channels * mlp_ratio)
        if hidden < 1:
            raise ValueError("mlp_ratio creates an empty hidden layer")

        self.token_channels = token_channels
        self.num_scene_tokens = num_scene_tokens
        self.routing_steps = routing_steps
        self.nearest_slots = nearest_slots
        self.geometry_scale = float(geometry_scale)
        self.position_scale = float(position_scale)

        self.route_norm = nn.LayerNorm(token_channels)
        self.route_projection = nn.Linear(token_channels, route_dim, bias=False)
        self.semantic_scale_logit = nn.Parameter(torch.zeros(()))
        self.geometry_embed = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, token_channels))
        self.cross_refinement = (
            RoutingPriorCrossAttention(token_channels, refinement_dim, refinement_heads)
            if use_cross_refinement else None
        )
        self.refinement_gain = (
            nn.Parameter(torch.full((token_channels,), 0.1)) if use_cross_refinement else None
        )
        self.relation = (
            RegionSelfAttention(token_channels, relation_dim, relation_heads, self.geometry_scale)
            if use_relation
            else None
        )
        self.ffn_norm = nn.LayerNorm(token_channels)
        self.ffn = nn.Sequential(nn.Linear(token_channels, hidden), nn.GELU(), nn.Linear(hidden, token_channels))
        self.ffn_gain = nn.Parameter(torch.full((token_channels,), 0.1))
        self.norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        patch_token: Tensor,
        patch_center: Tensor,
        return_intermediate: bool = False,
        *,
        patch_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if patch_token.ndim != 3 or patch_token.shape[-1] != self.token_channels:
            raise ValueError("patch_token must be [B,M,token_channels]")
        if not patch_token.shape[0] or patch_token.shape[1] < self.num_scene_tokens:
            raise ValueError("nonempty batch and M >= num_scene_tokens are required")
        if patch_center.shape != (*patch_token.shape[:2], 3):
            raise ValueError("patch_center must be [B,M,3]")
        if not patch_token.is_floating_point() or not patch_center.is_floating_point():
            raise TypeError("tokens and XYZ must be floating point")

        tokens, points = patch_token, patch_center.float()
        valid = None
        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2] or patch_mask.dtype != torch.bool:
                raise ValueError("patch_mask must be bool [B,M]")
            if not (patch_mask.sum(-1) >= self.num_scene_tokens).all():
                raise ValueError("each sample needs at least K valid patches")
            valid = patch_mask[:, None]
            tokens = torch.where(patch_mask[..., None], tokens, 0.0)
            points = torch.where(patch_mask[..., None], points, 0.0)

        regions = self.num_scene_tokens
        seed_indices = sample_region_seeds(points, regions, patch_mask)
        anchors = gather_tokens(points, seed_indices)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            distance2 = (anchors[:, :, None] - points[:, None]).square().sum(-1)
            # The Cauchy prior retains feature-dependent routing at larger spatial separations.
            geometry_bias = -torch.log1p(distance2 / self.geometry_scale**2)

        with torch.no_grad():
            nearest = distance2.topk(min(self.nearest_slots, regions), dim=1, largest=False).indices
            allowed = torch.zeros_like(distance2, dtype=torch.bool).scatter_(1, nearest, True)
            ownership = torch.zeros_like(allowed).scatter_(2, seed_indices[..., None], True)
            allowed = torch.where(ownership.any(dim=1, keepdim=True), ownership, allowed)
            if valid is not None:
                allowed = allowed & valid
            excluded = ~allowed

        keys = self.route_projection(self.route_norm(tokens.to(self.route_projection.weight.dtype)))
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            keys = F.normalize(keys.float(), dim=-1, eps=1e-6)
            transposed_keys = keys.transpose(-2, -1)
            prototypes = gather_tokens(keys, seed_indices)
            scale = 4.0 * self.semantic_scale_logit.float().sigmoid()

            for step in range(self.routing_steps):
                logits = scale * (prototypes @ transposed_keys) + geometry_bias
                logits = logits.masked_fill(excluded, -torch.inf)
                if valid is not None:
                    logits = torch.where(valid, logits, 0.0)  # Avoid all-inf softmax on padding.
                assignment = logits.softmax(dim=1)
                if valid is not None:
                    assignment = assignment * valid
                mass = assignment.sum(-1)
                weight = assignment / mass.clamp_min(1e-12)[..., None]
                if step + 1 < self.routing_steps:
                    prototypes = F.normalize(weight @ keys, dim=-1, eps=1e-6)

            pooled = weight @ tokens.float()
            centers = weight @ points
            # Centered moments avoid cancellation in E[xx^T] - E[x]E[x]^T.
            delta = points[:, None] - centers[:, :, None]
            covariance = (weight[..., None] * delta).transpose(-2, -1) @ delta
            normalized_covariance = covariance / self.geometry_scale**2
            upper = torch.stack(
                tuple(normalized_covariance[..., i, j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))),
                -1,
            )
            descriptor = torch.cat((centers / self.position_scale, upper), -1)

        geometry = self.geometry_embed(descriptor.to(self.geometry_embed[0].weight.dtype))
        base_tokens = pooled.to(geometry.dtype) + geometry
        scene_tokens = base_tokens
        refinement_update = None
        if self.cross_refinement is not None:
            refinement_update = self.cross_refinement(scene_tokens, tokens, weight)
            scene_tokens = scene_tokens + self.refinement_gain.to(scene_tokens.dtype) * refinement_update.to(
                scene_tokens.dtype
            )
        relation_attention = None
        if self.relation is not None:
            scene_tokens, relation_attention = self.relation(scene_tokens, centers)
        scene_tokens = scene_tokens + self.ffn_gain.to(scene_tokens.dtype) * self.ffn(self.ffn_norm(scene_tokens))

        output = {
            "scene_token": self.norm(scene_tokens),
            "scene_center": centers,
            "scene_covariance": covariance,
            "scene_mass": mass,
            "scene_mask": torch.ones_like(mass, dtype=torch.bool),
        }
        if return_intermediate:
            output = {
                **output,
                "assignment": assignment,
                "pool_weight": weight,
                "anchor_indices": seed_indices,
                "routing_anchors": anchors,
                "routing_allowed": allowed,
                "pooled_token": pooled,
                "base_scene_token": base_tokens,
            }
            if refinement_update is not None:
                output["refinement_update"] = refinement_update
            if relation_attention is not None:
                output["relation_attention"] = relation_attention
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return self.num_scene_tokens, self.token_channels
