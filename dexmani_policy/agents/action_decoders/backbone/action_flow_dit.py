from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP


WEIGHT_INIT_STD = 0.02


def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # eps=1e-5 (not 1e-6) also caps the backward gain on a near-zero row: the
    # Jacobian scales as 1/sqrt(rms^2 + eps), so a QK-Norm'd q/k row whose RMS
    # collapses is amplified 316x rather than 1000x. Matters under bf16.
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized.to(dtype=x.dtype)


def modulate_rms(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """RMSNorm followed by adaptive scale/shift modulation (standalone, no module overhead)."""
    return _rms_norm(x, eps) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _prepare_scalar(value, batch_size: int, device: torch.device) -> torch.Tensor:
    """Normalize a python float / 0-d tensor / [B] tensor into a [B] float tensor."""
    if not torch.is_tensor(value):
        value = torch.tensor([value], dtype=torch.float32, device=device)
    elif value.ndim == 0:
        value = value[None].to(device)
    else:
        value = value.to(device)
    return value.expand(batch_size)


class RMSNorm(nn.Module):
    """Weight-only RMSNorm.

    ``eps`` is 1e-5 (not 1e-6) for the same reason as in the GeoFormer: it caps
    the backward gain on a near-zero row at ``1/sqrt(eps)`` = 316x instead of
    1000x. ``fusion_norm`` runs here over the state+timestep+step sum, which can
    cancel toward zero on some dims once conditioning weights grow.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.eps) * self.weight.to(dtype=x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: ``out_proj(silu(gate_proj(x)) * value_proj(x))``."""

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.out_proj = nn.Linear(ffn_hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm:
            q = _rms_norm(q)
            k = _rms_norm(k)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        return self.out_proj(output)


class CrossAttentionWithCache(nn.Module):
    """Action-to-observation full cross-attention with a static per-generation KV cache.

    Q, K and V all use ``num_heads`` heads (no GQA in v1 — refactor spec §24).
    The cache retains the K/V projections of the geometry memory so the context
    is encoded once and reused across every NFE solver iteration.

    ``head_dim`` is derived from ``context_dim // num_heads`` so the K/V
    projection lives in the geometry-token width.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if context_dim % num_heads != 0:
            raise ValueError(
                f"context_dim ({context_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = context_dim // num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop

        self.q_proj = nn.Linear(query_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(context_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, query_dim)

        self._cached_k: torch.Tensor | None = None
        self._cached_v: torch.Tensor | None = None

    def _project_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_len, _ = context.shape
        k = self.k_proj(context).reshape(
            batch_size,
            context_len,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(context).reshape(
            batch_size,
            context_len,
            self.num_heads,
            self.head_dim,
        )
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.qk_norm:
            k = _rms_norm(k)
        return k, v

    def setup_kv_cache(self, context: torch.Tensor) -> None:
        self._cached_k, self._cached_v = self._project_kv(context)

    def clear_kv_cache(self) -> None:
        self._cached_k = None
        self._cached_v = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        q = q.permute(0, 2, 1, 3)
        if self.qk_norm:
            q = _rms_norm(q)

        if self._cached_k is None:
            k, v = self._project_kv(context)
        else:
            k, v = self._cached_k, self._cached_v

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        )
        return self.out_proj(output)


class ActionFlowDiTXBlock(nn.Module):
    """DiT-X block: Self-Attn -> Cross-Attn -> FFN, each zero-gated.

    Modulation order (9 x hidden_dim)::

        scale_sa, shift_sa, gate_sa,
        scale_ca, shift_ca, gate_ca,
        scale_ffn, shift_ffn, gate_ffn

    The shared modulation head lives on the parent; each block only applies a
    per-layer affine calibration to the shared 384-D conditioner latent.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int = 12,
        ffn_hidden_dim: int = 2048,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        cond_bottleneck_dim: int = 384,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
        )
        self.cross_attn = CrossAttentionWithCache(
            query_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
        )
        self.ffn = SwiGLU(hidden_dim, ffn_hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(cond_bottleneck_dim))
        self.beta = nn.Parameter(torch.zeros(cond_bottleneck_dim))

    def forward(
        self,
        x: torch.Tensor,
        cond_latent: torch.Tensor,
        shared_modulation: nn.Linear,
        context: torch.Tensor,
    ) -> torch.Tensor:
        calibrated = cond_latent * (1 + self.gamma) + self.beta
        mod = shared_modulation(calibrated)
        (
            scale_sa, shift_sa, gate_sa,
            scale_ca, shift_ca, gate_ca,
            scale_ffn, shift_ffn, gate_ffn,
        ) = mod.chunk(9, dim=-1)

        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate_rms(x, scale_sa, shift_sa)
        )

        x = x + gate_ca.unsqueeze(1) * self.cross_attn(
            modulate_rms(x, scale_ca, shift_ca), context
        )

        x = x + gate_ffn.unsqueeze(1) * self.ffn(
            modulate_rms(x, scale_ffn, shift_ffn)
        )

        return x


class ActionFlowDiT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 768,
        context_dim: int | None = None,
        depth: int = 8,
        num_heads: int = 12,
        ffn_hidden_dim: int = 2048,
        timestep_embed_dim: int = 128,
        step_embed_dim: int = 64,
        state_embed_hidden_dim: int = 256,
        cond_bottleneck_dim: int = 384,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.context_dim = hidden_dim if context_dim is None else context_dim

        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.state_mlp = nn.Sequential(
            nn.Linear(2 * state_dim, state_embed_hidden_dim),
            nn.SiLU(),
            nn.Linear(state_embed_hidden_dim, hidden_dim),
        )
        self.timestep_embedder = TimestepMLP(
            pos_emb_dim=timestep_embed_dim,
            output_dim=hidden_dim,
        )
        self.step_embedder = TimestepMLP(
            pos_emb_dim=step_embed_dim,
            output_dim=hidden_dim,
        )
        self.fusion_norm = RMSNorm(hidden_dim)
        self.compact = nn.Sequential(
            nn.Linear(hidden_dim, cond_bottleneck_dim),
            nn.SiLU(),
        )
        self.shared_modulation = nn.Linear(cond_bottleneck_dim, 9 * hidden_dim)
        self.final_modulation = nn.Linear(cond_bottleneck_dim, 2 * hidden_dim)

        self.layers = nn.ModuleList(
            [
                ActionFlowDiTXBlock(
                    hidden_dim=hidden_dim,
                    context_dim=self.context_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    cond_bottleneck_dim=cond_bottleneck_dim,
                )
                for _ in range(depth)
            ]
        )
        self.action_out = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        nn.init.normal_(self.action_in.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.action_in.bias)
        nn.init.normal_(self.action_pos, std=WEIGHT_INIT_STD)

        nn.init.zeros_(self.shared_modulation.weight)
        nn.init.zeros_(self.shared_modulation.bias)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.zeros_(self.step_embedder.net[-1].weight)
        nn.init.zeros_(self.step_embedder.net[-1].bias)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["action_pos"],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | float,
        context: torch.Tensor,
        state: torch.Tensor,
        step_size: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.horizon or x.shape[2] != self.action_dim:
            raise ValueError(
                f"x must have shape [B, {self.horizon}, {self.action_dim}], got {tuple(x.shape)}"
            )
        if (
            context.ndim != 3
            or context.shape[0] != x.shape[0]
            or context.shape[2] != self.context_dim
        ):
            raise ValueError(
                f"context must have shape [B, L, {self.context_dim}] with B={x.shape[0]}, "
                f"got {tuple(context.shape)}"
            )
        if (
            state.ndim != 2
            or state.shape[0] != x.shape[0]
            or state.shape[1] != 2 * self.state_dim
        ):
            raise ValueError(
                f"state must have shape [B, {2 * self.state_dim}] with B={x.shape[0]}, "
                f"got {tuple(state.shape)}"
            )

        batch_size = x.shape[0]
        hidden = self.action_in(x) + self.action_pos.to(dtype=x.dtype)
        t = _prepare_scalar(timestep, batch_size, x.device)
        d = _prepare_scalar(step_size, batch_size, x.device)
        e = self.fusion_norm(
            self.state_mlp(state) + self.timestep_embedder(t) + self.step_embedder(d)
        )
        cond_latent = self.compact(e)

        for block in self.layers:
            hidden = block(hidden, cond_latent, self.shared_modulation, context)

        scale_f, shift_f = self.final_modulation(cond_latent).chunk(2, dim=-1)
        return self.action_out(modulate_rms(hidden, scale_f, shift_f))

    def setup_kv_cache(self, context: torch.Tensor) -> None:
        for block in self.layers:
            block.cross_attn.setup_kv_cache(context)

    def clear_kv_cache(self) -> None:
        for block in self.layers:
            block.cross_attn.clear_kv_cache()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActionFlowDiT(
        horizon=16,
        action_dim=19,
        state_dim=19,
        hidden_dim=768,
        context_dim=768,
        depth=8,
        num_heads=12,
        ffn_hidden_dim=2048,
        timestep_embed_dim=128,
        step_embed_dim=64,
        state_embed_hidden_dim=256,
        cond_bottleneck_dim=384,
    ).to(device)

    x = torch.randn(2, 16, 19, device=device)
    context = torch.randn(2, 385, 768, device=device)
    state = torch.randn(2, 38, device=device)
    timestep = torch.rand(2, device=device)

    # 2. Zero-init: every gate is 0 and action_out is 0, so output is exactly 0.
    with torch.no_grad():
        out_zero = model(x, timestep, context, state, 0.0)
    assert torch.count_nonzero(out_zero) == 0
    print("[PASS] zero-init: output is exactly zero at init")

    # Break the zero-inits so the model produces a non-trivial output for parity.
    nn.init.normal_(model.shared_modulation.weight, std=WEIGHT_INIT_STD)
    nn.init.normal_(model.shared_modulation.bias, std=WEIGHT_INIT_STD)
    nn.init.normal_(model.final_modulation.weight, std=WEIGHT_INIT_STD)
    nn.init.normal_(model.final_modulation.bias, std=WEIGHT_INIT_STD)
    nn.init.normal_(model.action_out.weight, std=WEIGHT_INIT_STD)
    nn.init.normal_(model.action_out.bias, std=WEIGHT_INIT_STD)
    nn.init.normal_(model.step_embedder.net[-1].weight, std=WEIGHT_INIT_STD)

    # 1. Forward shape + finite backward.
    out = model(x, 0.5, context, state, torch.tensor(0.0, device=device))
    assert tuple(out.shape) == (2, 16, 19), tuple(out.shape)
    out.float().square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.grad is not None
    )
    print("[PASS] forward shape [2,16,19] + finite backward")
    model.zero_grad()

    # 3. Parameter count.
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] total parameters: {total:,}")
    assert 79e6 <= total <= 82e6, total
    print("[PASS] parameter count within [79M, 82M]")
    breakdown = [
        ("action_in", model.action_in),
        ("action_pos", model.action_pos),
        ("state_mlp", model.state_mlp),
        ("timestep_embedder", model.timestep_embedder),
        ("step_embedder", model.step_embedder),
        ("fusion_norm", model.fusion_norm),
        ("compact", model.compact),
        ("shared_modulation", model.shared_modulation),
        ("final_modulation", model.final_modulation),
        ("layers (x8)", model.layers),
        ("action_out", model.action_out),
    ]
    for name, sub in breakdown:
        n = sub.numel() if isinstance(sub, torch.Tensor) else sum(
            p.numel() for p in sub.parameters()
        )
        print(f"    {name}: {n:,}")

    # 4. KV-cache parity.
    model.eval()
    with torch.no_grad():
        out_uncached = model(x, timestep, context, state, 0.0)
        assert torch.count_nonzero(out_uncached) > 0, "output must be non-trivial"
        model.setup_kv_cache(context)
        out_cached = model(x, timestep, context, state, 0.0)
        model.clear_kv_cache()
    torch.testing.assert_close(out_cached, out_uncached, rtol=1e-5, atol=1e-5)
    print("[PASS] KV-cache parity (fp32, 1e-5)")

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out_uncached_bf16 = model(x, timestep, context, state, 0.0)
            model.setup_kv_cache(context)
            out_cached_bf16 = model(x, timestep, context, state, 0.0)
            model.clear_kv_cache()
    torch.testing.assert_close(out_cached_bf16, out_uncached_bf16, rtol=1e-2, atol=1e-2)
    print("[PASS] KV-cache parity (bf16, 1e-2)")

    # 5. Cache must never enter state_dict (plain python attribute).
    model.setup_kv_cache(context)
    state_dict_keys = list(model.state_dict().keys())
    model.clear_kv_cache()
    assert not any("_cached_k" in k or "_cached_v" in k for k in state_dict_keys)
    print("[PASS] KV cache not present in state_dict")

    print("ALL CHECKS PASSED")
