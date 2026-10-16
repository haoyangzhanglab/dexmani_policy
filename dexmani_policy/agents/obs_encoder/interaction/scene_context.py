"""GReST: Geometric Region-relation Scene Tokenizer.

Dense spatial patches -> local competitive regions -> one narrow relation block.
Only the scene branch is compressed; the hand reader keeps full GAPT memory.
Slots are observed regions, not guaranteed objects, contacts or tracked identities.
See docs/grest_scene_tokenizer.md for geometry semantics and validation boundaries.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _gather(x: Tensor, idx: Tensor) -> Tensor:
    return x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))


@torch.no_grad()
def coverage_seeds(xyz: Tensor, valid: Tensor, count: int) -> Tensor:
    """Unique-index FPS. Exact distance ties are resolved by input index.

    Valid count >= count is a caller precondition. Coincident points are allowed.
    This has a fixed Python loop over K, with device tensor operations, no CPU
    connected-components routine. It is not a fused CUDA FPS implementation.
    """
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        p = torch.where(valid[..., None], xyz.float(), 0.0)
        mean = p.sum(1, keepdim=True) / valid.sum(1)[:, None, None].float()
        dist = (p - mean).square().sum(-1)
        farthest = dist.masked_fill(~valid, -torch.inf).argmax(-1)
        batch = torch.arange(p.shape[0], device=p.device)
        used = torch.zeros_like(valid)
        nearest = torch.full_like(dist, torch.inf)
        result = []
        for j in range(count):
            result.append(farthest)
            used[batch, farthest] = True
            d2 = (p - p[batch, farthest, None]).square().sum(-1)
            nearest = torch.minimum(nearest, d2)
            if j + 1 < count:
                farthest = nearest.masked_fill(used | ~valid, -torch.inf).argmax(-1)
        return torch.stack(result, 1)


class RegionRelation(nn.Module):
    """One narrow slot-to-slot attention update with geometric bias AND values.

    Positions specify the originating regions, not complete final receptive
    fields. No distance cutoff: far-apart regions may be task related.
    Self edges are valid, including when K=1.
    """

    def __init__(self, channels: int, attention_dim: int, heads: int, scale: float):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (channels, attention_dim, heads)):
            raise ValueError("relation dimensions must be positive integers")
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
        self.edge = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads + attention_dim)
        )
        self.out = nn.Linear(attention_dim, channels)
        self.gain = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, slots: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
        b, k, _ = slots.shape
        qkv = self.qkv(self.norm(slots)).reshape(b, k, 3, self.heads, self.head_dim)
        q, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, key = self.q_norm(q), self.k_norm(key)
        with torch.autocast(device_type=slots.device.type, enabled=False):
            # edge k -> l is c_l - c_k.
            relative = (centers[:, None].float() - centers[:, :, None].float()) / self.scale
            edge_input = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
        edge = self.edge(edge_input.to(self.edge[0].weight.dtype))
        bias, edge_value = edge.split((self.heads, self.heads * self.head_dim), -1)
        edge_value = edge_value.reshape(b, k, k, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        with torch.autocast(device_type=slots.device.type, enabled=False):
            logits = q.float() @ key.float().transpose(-2, -1)
            logits = logits / math.sqrt(self.head_dim) + bias.float().permute(0, 3, 1, 2)
            attention = logits.softmax(-1)
            message = attention @ value.float()
            message = message + (attention[..., None] * edge_value.float()).sum(-2)
            message = message.transpose(1, 2).reshape(b, k, -1)
        update = self.out(message.to(self.out.weight.dtype))
        return slots + self.gain.to(slots.dtype) * update.to(slots.dtype), attention


class GeometricRegionRelationTokenizer(nn.Module):
    """Fixed-budget geometric grouping followed by a small relation encoder.

    Geometry scale is in meters. position_scale changes feature scaling only.
    Default K=12 is an initial budget, not an empirically optimal choice.
    Any positive K <= the valid input count is supported, including non-powers
    of two. No learned tensor has a K-sized axis; deployment should nevertheless
    use the trained budget until cross-budget performance has been evaluated.

    validate_inputs=True performs finite/count checks and may synchronize a GPU.
    Disable only after the upstream GAPT contract has been enforced. Metadata
    checks remain. No downstream padding mask is needed: every output is valid.
    """

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
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("token_channels", token_channels), ("num_scene_tokens", num_scene_tokens),
            ("route_dim", route_dim), ("routing_steps", routing_steps),
            ("nearest_slots", nearest_slots), ("relation_dim", relation_dim),
            ("relation_heads", relation_heads),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if relation_dim % relation_heads or relation_dim // relation_heads < 2:
            raise ValueError("relation_dim / relation_heads must be an integer >= 2")
        for name, value in (("geometry_scale", geometry_scale), ("position_scale", position_scale), ("mlp_ratio", mlp_ratio)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        hidden = int(token_channels * mlp_ratio)
        if hidden < 1:
            raise ValueError("mlp_ratio creates an empty hidden layer")
        self.token_channels = token_channels
        self.num_scene_tokens = num_scene_tokens
        self.route_dim = route_dim
        self.routing_steps = routing_steps
        self.nearest_slots = nearest_slots
        self.geometry_scale = float(geometry_scale)
        self.position_scale = float(position_scale)
        if type(validate_inputs) is not bool or type(use_relation) is not bool:
            raise TypeError("validate_inputs and use_relation must be bool")
        self.validate_inputs = validate_inputs
        self.route_norm = nn.LayerNorm(token_channels)
        self.route_projection = nn.Linear(token_channels, route_dim, bias=False)
        self.semantic_scale_logit = nn.Parameter(torch.zeros(()))
        # Small fixed-size descriptor: centroid XYZ + symmetric covariance (6).
        # Assignment count/mass is returned for diagnostics, NOT encoded as
        # object importance, occupied volume, physical mass or confidence.
        self.geometry_embed = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(), nn.Linear(64, token_channels)
        )
        self.relation = RegionRelation(
            token_channels, relation_dim, relation_heads, self.geometry_scale
        ) if use_relation else None
        self.ffn_norm = nn.LayerNorm(token_channels)
        self.ffn = nn.Sequential(
            nn.Linear(token_channels, hidden), nn.GELU(), nn.Linear(hidden, token_channels)
        )
        self.ffn_gain = nn.Parameter(torch.full((token_channels,), 0.1))
        self.norm = nn.LayerNorm(token_channels)

    def _validate(self, z: Tensor, xyz: Tensor, mask: Tensor | None) -> Tensor:
        if z.ndim != 3 or z.shape[-1] != self.token_channels:
            raise ValueError("patch_token must be [B,M,token_channels]")
        if not z.shape[0] or z.shape[1] < self.num_scene_tokens:
            raise ValueError("nonempty batch and M >= num_scene_tokens are required")
        if xyz.shape != (*z.shape[:2], 3):
            raise ValueError("patch_center must be [B,M,3]")
        if not z.is_floating_point() or not xyz.is_floating_point():
            raise TypeError("tokens and XYZ must be floating point")
        if xyz.device != z.device:
            raise ValueError("tokens and XYZ must share a device")
        if mask is None:
            mask = torch.ones(z.shape[:2], device=z.device, dtype=torch.bool)
        elif mask.shape != z.shape[:2] or mask.dtype != torch.bool or mask.device != z.device:
            raise ValueError("patch_mask must be bool [B,M] on the input device")
        if self.validate_inputs:
            if not bool((mask.sum(-1) >= self.num_scene_tokens).all()):
                raise ValueError("each sample needs at least K valid patches")
            finite = torch.isfinite(z).all(-1) & torch.isfinite(xyz).all(-1)
            if not bool((finite | ~mask).all()):
                raise ValueError("valid patches must be finite")
        return mask

    def forward(
        self,
        patch_token: Tensor,
        patch_center: Tensor,
        return_intermediate: bool = False,
        *,
        patch_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        valid = self._validate(patch_token, patch_center, patch_mask)
        z = torch.where(valid[..., None], patch_token, 0.0)
        p = torch.where(valid[..., None], patch_center.float(), 0.0)
        k = self.num_scene_tokens
        seed_idx = coverage_seeds(p, valid, k)
        anchors = _gather(p, seed_idx)
        with torch.autocast(device_type=z.device.type, enabled=False):
            distance2 = (anchors[:, :, None] - p[:, None]).square().sum(-1)
            # A Cauchy-shaped locality prior has a slower tail than a Gaussian.
            # Sparse budgets / larger workspaces need not make routing one-hot
            # before learned feature scores can contribute. Nearest-seed support
            # still restricts cross-region mixing. This is not contact probability.
            geometry_bias = -torch.log1p(distance2 / self.geometry_scale ** 2)
        with torch.no_grad():
            near = distance2.topk(min(self.nearest_slots, k), dim=1, largest=False).indices
            allowed = torch.zeros_like(distance2, dtype=torch.bool).scatter_(1, near, True)
            seed_ownership = F.one_hot(seed_idx, z.shape[1]).bool()
            is_seed = seed_ownership.any(dim=1, keepdim=True)
            allowed = torch.where(is_seed, seed_ownership, allowed) & valid[:, None]
        # Project dense features ONCE. Refinement occurs only in route_dim.
        keys = self.route_projection(self.route_norm(z.to(self.route_projection.weight.dtype)))
        with torch.autocast(device_type=z.device.type, enabled=False):
            keys = F.normalize(keys.float(), dim=-1, eps=1e-6)
            prototypes = _gather(keys, seed_idx)
            scale = 4.0 * self.semantic_scale_logit.float().sigmoid()
            for step in range(self.routing_steps):
                logits = scale * (prototypes @ keys.transpose(-2, -1)) + geometry_bias
                logits = logits.masked_fill(~allowed, -torch.inf)
                # No softmax(-inf,...,-inf) for padded columns.
                logits = torch.where(valid[:, None], logits, 0.0)
                assignment = logits.softmax(dim=1) * valid[:, None].float()
                mass = assignment.sum(-1)
                weight = assignment / mass.clamp_min(1e-12)[..., None]
                if step + 1 < self.routing_steps:
                    prototypes = F.normalize(weight @ keys, dim=-1, eps=1e-6)
            # Full-width dense features are pooled only at the final iteration.
            pooled = weight @ z.float()
            centers = weight @ p
            # Direct centered moments avoid cancellation in E[xx^T]-E[x]E[x]^T.
            delta = p[:, None] - centers[:, :, None]
            covariance = torch.einsum("bkn,bkni,bknj->bkij", weight, delta, delta)
            normalized_cov = covariance / self.geometry_scale ** 2
            upper = torch.stack(tuple(normalized_cov[..., i, j] for i, j in
                                      ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))), -1)
            descriptor = torch.cat((centers / self.position_scale, upper), -1)
        geom = self.geometry_embed(descriptor.to(self.geometry_embed[0].weight.dtype))
        base = pooled.to(geom.dtype) + geom
        slots = base
        relation_attention = None
        if self.relation is not None:
            slots, relation_attention = self.relation(slots, centers)
        slots = slots + self.ffn_gain.to(slots.dtype) * self.ffn(self.ffn_norm(slots))
        slots = self.norm(slots)
        output = {
            "scene_token": slots,
            "scene_center": centers,
            "scene_covariance": covariance,
            "scene_mass": mass,
            "scene_mask": torch.ones_like(mass, dtype=torch.bool),
        }
        if return_intermediate:
            output.update(
                assignment=assignment, pool_weight=weight, anchor_indices=seed_idx,
                routing_anchors=anchors, routing_allowed=allowed,
                pooled_token=pooled, base_scene_token=base,
            )
            if relation_attention is not None:
                output["relation_attention"] = relation_attention
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return self.num_scene_tokens, self.token_channels
