"""Geometry-only region pooling followed by optional context-conditioned patch reading."""

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


@torch.no_grad()
def sample_region_seeds(points: Tensor, count: int, mask: Tensor | None = None) -> Tensor:
    """Centroid-initialized FPS with unique indices; masked coordinates must already be zero.

    Exact distance ties use input order. Distinct indices can have identical coordinates.
    """
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.float()
        excluded = (
            torch.zeros(points.shape[:2], device=points.device, dtype=torch.bool)
            if mask is None
            else ~mask
        )
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


class LocalGlobalCrossAttention(nn.Module):
    """One patch read: soft local distance bias and distance-free global heads.

    The hard mode is an ablation using the sparse pooling prior for every head.
    """

    def __init__(
        self, channels: int, attention_dim: int, heads: int, global_heads: int, mode: str
    ) -> None:
        super().__init__()
        if min(channels, attention_dim, heads) < 1 or attention_dim % heads:
            raise ValueError("attention_dim must be positive and divisible by heads")
        if not 0 <= global_heads <= heads:
            raise ValueError("global_heads must be between zero and heads")
        if mode not in ("soft", "hard"):
            raise ValueError("refinement_mode must be 'soft' or 'hard'")

        self.heads = heads
        self.head_dim = attention_dim // heads
        self.global_heads = global_heads
        self.mode = mode
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, attention_dim, bias=False)
        self.kv = nn.Linear(channels, 2 * attention_dim, bias=False)
        self.out = nn.Linear(attention_dim, channels)

    def attention_bias(
        self, geometry_bias: Tensor, pool_weight: Tensor, patch_mask: Tensor | None
    ) -> Tensor:
        if self.mode == "hard":
            bias = pool_weight.clamp_min(1e-30).log().masked_fill(pool_weight <= 0, -torch.inf)
            bias = bias[:, None].expand(-1, self.heads, -1, -1)
        else:
            local = geometry_bias[:, None].expand(-1, self.heads - self.global_heads, -1, -1)
            global_bias = torch.zeros_like(geometry_bias[:, None]).expand(
                -1, self.global_heads, -1, -1
            )
            bias = torch.cat((local, global_bias), dim=1)
        if patch_mask is not None:
            bias = bias.masked_fill(~patch_mask[:, None, None], -torch.inf)
        return bias

    def forward(self, query: Tensor, memory: Tensor, bias: Tensor) -> Tensor:
        batch, regions, _ = query.shape
        patches = memory.shape[1]
        query = self.q(self.query_norm(query.to(self.q.weight.dtype)))
        memory = self.memory_norm(memory.to(self.kv.weight.dtype))
        key, value = self.kv(memory).chunk(2, dim=-1)
        query = query.reshape(batch, regions, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, patches, self.heads, self.head_dim).transpose(1, 2)
        refined = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias.to(query.dtype), dropout_p=0.0
        )
        return self.out(refined.transpose(1, 2).reshape(batch, regions, -1))


class SceneAggregator(nn.Module):
    """Compress [B,M,D] patches into K regions in the input metric coordinate frame.

    Valid tokens/XYZ must be finite; masked entries may contain NaN/Inf. Every
    sample needs at least K valid patches. Centers (m), covariance (m^2), and
    mass (patch assignment count) describe only the geometric partition.
    FPS indices are not persistent identities. This is not SE(3)-equivariant.
    """

    def __init__(
        self,
        token_channels: int = 192,
        num_scene_tokens: int = 12,
        *,
        nearest_slots: int = 3,
        geometry_scale: float = 0.10,
        position_scale: float = 1.0,
        mlp_ratio: float = 2.0,
        use_cross_refinement: bool = True,
        refinement_dim: int = 128,
        refinement_heads: int = 4,
        global_heads: int = 1,
        refinement_mode: str = "soft",
        query_source: str = "region",
    ) -> None:
        super().__init__()
        for name, value in (
            ("token_channels", token_channels),
            ("num_scene_tokens", num_scene_tokens),
            ("nearest_slots", nearest_slots),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, scale in (("geometry_scale", geometry_scale), ("position_scale", position_scale)):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name} must be finite and positive")
        hidden = int(token_channels * mlp_ratio)
        if hidden < 1:
            raise ValueError("mlp_ratio creates an empty hidden layer")
        if query_source not in ("region", "learned"):
            raise ValueError("query_source must be 'region' or 'learned'")

        self.token_channels = token_channels
        self.num_scene_tokens = num_scene_tokens
        self.nearest_slots = nearest_slots
        self.geometry_scale = float(geometry_scale)
        self.position_scale = float(position_scale)

        self.geometry_embed = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(), nn.Linear(64, token_channels)
        )
        self.cross_refinement = (
            LocalGlobalCrossAttention(
                token_channels, refinement_dim, refinement_heads, global_heads, refinement_mode
            )
            if use_cross_refinement
            else None
        )
        self.refinement_gain = (
            nn.Parameter(torch.full((token_channels,), 0.1)) if use_cross_refinement else None
        )
        self.learned_query = None
        if use_cross_refinement and query_source == "learned":
            self.learned_query = nn.Parameter(torch.empty(1, num_scene_tokens, token_channels))
            nn.init.normal_(self.learned_query, std=0.02)
        self.ffn_norm = nn.LayerNorm(token_channels)
        self.ffn = nn.Sequential(
            nn.Linear(token_channels, hidden), nn.GELU(), nn.Linear(hidden, token_channels)
        )
        self.ffn_gain = nn.Parameter(torch.full((token_channels,), 0.1))
        self.norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        patch_token: Tensor,
        patch_center: Tensor,
        return_intermediate: bool = False,
        *,
        patch_mask: Tensor | None = None,
        query_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """query_context [B,D] is caller-encoded current information, used only in Q."""
        if patch_token.ndim != 3 or patch_token.shape[-1] != self.token_channels:
            raise ValueError("patch_token must be [B,M,token_channels]")
        if not patch_token.shape[0] or patch_token.shape[1] < self.num_scene_tokens:
            raise ValueError("nonempty batch and M >= num_scene_tokens are required")
        if patch_center.shape != (*patch_token.shape[:2], 3):
            raise ValueError("patch_center must be [B,M,3]")
        if not patch_token.is_floating_point() or not patch_center.is_floating_point():
            raise TypeError("tokens and XYZ must be floating point")
        if query_context is not None:
            if query_context.shape != (patch_token.shape[0], self.token_channels):
                raise ValueError("query_context must be [B,token_channels]")
            if not query_context.is_floating_point():
                raise TypeError("query_context must be floating point")

        tokens, points = patch_token, patch_center.float()
        valid = None
        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2] or patch_mask.dtype != torch.bool:
                raise ValueError("patch_mask must be bool [B,M]")
            # Torch 2.4 Inductor drops _assert_async; keep this boundary check
            # in Python, accepting a graph break for masked compiled calls.
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
            geometry_bias = -torch.log1p(distance2 / self.geometry_scale**2)

        with torch.no_grad():
            nearest = distance2.topk(min(self.nearest_slots, regions), dim=1, largest=False).indices
            allowed = torch.zeros_like(distance2, dtype=torch.bool).scatter_(1, nearest, True)
            # Each unique seed index owns itself, even for coincident coordinates.
            ownership = torch.zeros_like(allowed).scatter_(2, seed_indices[..., None], True)
            allowed = torch.where(ownership.any(dim=1, keepdim=True), ownership, allowed)
            if valid is not None:
                allowed = allowed & valid

        with torch.autocast(device_type=tokens.device.type, enabled=False):
            logits = geometry_bias.masked_fill(~allowed, -torch.inf)
            if valid is not None:
                logits = torch.where(valid, logits, 0.0)
            assignment = logits.softmax(dim=1)
            if valid is not None:
                assignment = assignment * valid
            mass = assignment.sum(-1)
            weight = assignment / mass[..., None]  # Seed ownership guarantees nonempty regions.
            pooled = weight @ tokens.float()
            centers = weight @ points
            # Centered moments avoid cancellation in E[xx^T] - E[x]E[x]^T.
            delta = points[:, None] - centers[:, :, None]
            covariance = (weight[..., None] * delta).transpose(-2, -1) @ delta
            normalized_covariance = covariance / self.geometry_scale**2
            upper = torch.stack(
                tuple(
                    normalized_covariance[..., i, j]
                    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
                ),
                -1,
            )
            descriptor = torch.cat((centers / self.position_scale, upper), -1)

        geometry = self.geometry_embed(descriptor.to(self.geometry_embed[0].weight.dtype))
        # Keep pooled features and small learned residuals in FP32 under autocast.
        base_tokens = pooled + geometry.float()
        scene_tokens = base_tokens
        refinement_update = attention_bias = None
        if self.cross_refinement is not None:
            query = (
                base_tokens if self.learned_query is None
                else self.learned_query.float().expand_as(base_tokens)
            )
            if query_context is not None:
                query = query + query_context[:, None].float()
            attention_bias = self.cross_refinement.attention_bias(geometry_bias, weight, patch_mask)
            refinement_update = self.cross_refinement(query, tokens, attention_bias)
            scene_tokens = scene_tokens + cast(Tensor, self.refinement_gain).float() * refinement_update.float()
        normalized = self.ffn_norm(scene_tokens.to(self.ffn_norm.weight.dtype))
        scene_tokens = scene_tokens + self.ffn_gain.float() * self.ffn(normalized).float()

        output = {
            "scene_token": self.norm(scene_tokens.to(self.norm.weight.dtype)),
            "scene_center": centers,
            "scene_covariance": covariance,
            "scene_mass": mass,
            "scene_mask": torch.ones_like(mass, dtype=torch.bool),
        }
        if return_intermediate:
            output.update(
                assignment=assignment,
                pool_weight=weight,
                anchor_indices=seed_indices,
                region_anchors=anchors,
                pool_support=allowed,
                pooled_token=pooled,
                base_scene_token=base_tokens,
            )
            if refinement_update is not None:
                output["refinement_update"] = refinement_update
                output["attention_bias"] = attention_bias
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return self.num_scene_tokens, self.token_channels
