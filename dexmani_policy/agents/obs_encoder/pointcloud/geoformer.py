"""GeoFormer: a two-frame 3D geometric self-attention encoder.

Relational reasoning over point-cloud patch tokens conditioned on their 3D
coordinates via a metric-wavelength 3D RoPE. Fully self-contained (torch only);
knows nothing about joint_state, actions, flow timesteps, NFE, or robot FK.

Public contract:
    tokens: [B, N, hidden_dim]  +  xyz: [B, N, 3]  ->  tokens: [B, N, hidden_dim]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Parameter-free root-mean-square normalization (computed in fp32).

    ``eps`` is 1e-5 rather than 1e-6 because it also caps the backward gain on a
    near-zero row: the Jacobian scales as ``1/sqrt(rms^2 + eps)``, so a q/k row
    whose RMS collapses toward 0 is amplified by ``1/sqrt(eps)`` — 1000x at 1e-6
    versus 316x at 1e-5. Under bf16 that difference is the margin between a large
    gradient and an overflow to Inf.
    """
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized.to(dtype=x.dtype)


def _apply_3d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x along three axis blocks, pairing even/odd dims within each block.

    x:   [B, H, N, head_dim] with head_dim = 3 * 2 * half_axis
    cos: [B, N, 3, half_axis]
    sin: [B, N, 3, half_axis]
    """
    B, H, N, D = x.shape
    half_axis = D // 6
    x = x.reshape(B, H, N, 3, half_axis, 2)
    cos = cos.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 3, half_axis, 1]
    sin = sin.unsqueeze(1).unsqueeze(-1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    rot0 = x0 * cos - x1 * sin
    rot1 = x0 * sin + x1 * cos
    return torch.cat((rot0, rot1), dim=-1).reshape(B, H, N, D)


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch with probability ``drop_prob``.

    Applied per-sample (batch index), not per-token, and scales the kept branch by
    ``1/keep_prob`` so the expected magnitude is preserved (timm convention).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with a single weight vector (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, dim] -> [B, N, dim]
        return _rms_norm(x, self.eps) * self.weight.to(dtype=x.dtype)


class RotaryPositionEmbedding3D(nn.Module):
    """Metric-wavelength 3D rotary embedding over (x, y, z) patch centers.

    head_dim is split into three axis blocks of head_dim // 3, each containing
    head_dim // 6 even/odd frequency pairs. Rotation angle for axis a is
    ``2 * pi * xyz[..., a] / wavelength``, so the same physical distance maps to
    the same rotary phase across episodes (no per-cloud normalization).

    Design note: base=10000 sinusoidal frequencies give wavelengths ~1..10000 in
    normalized units, far too coarse to resolve 0.04-0.08 patch radii, which
    would leave RoPE effectively inert. Hence the metric wavelength range
    [0.02, 2.0], exposed as kwargs so it is tunable without a code change.
    """

    def __init__(
        self,
        head_dim: int,
        min_wavelength: float = 0.02,
        max_wavelength: float = 2.0,
    ):
        super().__init__()
        assert head_dim % 6 == 0, "head_dim must be divisible by 6 (3 axes x even/odd pairs)"
        self.head_dim = head_dim
        self.half_axis = head_dim // 6

        # Log-spaced wavelengths; inv_freq = 2*pi / wavelength (radians per unit distance).
        wavelengths = torch.logspace(
            math.log10(min_wavelength), math.log10(max_wavelength), self.half_axis
        )
        inv_freq = 2.0 * math.pi / wavelengths  # [half_axis], fp32
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Angles computed in fp32, then cast back to xyz.dtype.

        xyz: [B, N, 3] -> (cos, sin) each [B, N, 3, half_axis]
        """
        angles = xyz.float().unsqueeze(-1) * self.inv_freq  # [B, N, 3, half_axis]
        cos = torch.cos(angles).to(dtype=xyz.dtype)
        sin = torch.sin(angles).to(dtype=xyz.dtype)
        return cos, sin


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: W_o(SiLU(W_g x) * W_v x)."""

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.out_proj = nn.Linear(ffn_hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class GeoFormerBlock(nn.Module):
    """RMSNorm -> full self-attention (QK-Norm + 3D RoPE + SDPA) -> SwiGLU, with residuals."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        qk_norm: bool,
        use_3d_rope: bool,
        attn_drop: float,
        rope: RotaryPositionEmbedding3D,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm
        self.use_3d_rope = use_3d_rope
        self.attn_drop = attn_drop
        self.rope = rope

        self.norm1 = RMSNorm(hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_hidden_dim)
        self.drop_path_attn = DropPath(drop_path)
        self.drop_path_ffn = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, N, d], xyz: [B, N, 3] -> [B, N, d]
        B, N, D = x.shape
        H = self.num_heads

        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, head_dim]

        if self.qk_norm:
            q = _rms_norm(q)
            k = _rms_norm(k)
        if self.use_3d_rope:
            if cos is None or sin is None:
                cos, sin = self.rope(xyz)
            q = _apply_3d_rope(q, cos, sin)
            k = _apply_3d_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )  # [B, H, N, head_dim]
        attn = attn.transpose(1, 2).reshape(B, N, D)
        x = x + self.drop_path_attn(self.out_proj(attn))
        x = x + self.drop_path_ffn(self.ffn(self.norm2(x)))
        return x


class GeoFormer(nn.Module):
    """Stack of GeoFormerBlocks sharing a single 3D RoPE.

    Args:
        tokens: [B, N, 576]
        xyz:    [B, N, 3]
    Returns:
        tokens: [B, N, 576]
    """

    def __init__(
        self,
        hidden_dim: int = 576,
        depth: int = 4,
        num_heads: int = 12,
        ffn_hidden_dim: int = 1536,
        qk_norm: bool = True,
        use_3d_rope: bool = True,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        min_wavelength: float = 0.02,
        max_wavelength: float = 2.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_3d_rope = use_3d_rope

        # Hoisted to GeoFormer so the rotary trig can be computed once per forward
        # (xyz is identical across blocks) and shared, instead of per block.
        self.rope = RotaryPositionEmbedding3D(hidden_dim // num_heads, min_wavelength, max_wavelength)
        # Stochastic depth ramps linearly from 0 (first block) to drop_path_rate
        # (last block); a 4-block transformer cannot tolerate a flat high rate.
        drop_path_rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                GeoFormerBlock(
                    hidden_dim,
                    num_heads,
                    ffn_hidden_dim,
                    qk_norm,
                    use_3d_rope,
                    attn_drop,
                    self.rope,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )
        # Final norm on the residual stream, as in every pre-norm transformer.
        # Pre-norm bounds each sublayer's INPUT but never its output, so without
        # this the stream grows with weight magnitude and the backward Jacobian
        # grows far faster than the forward (measured: ~wscale^11 vs ~wscale^3),
        # producing Inf gradients while the loss still looks healthy. With it,
        # both output and gradients are weight-scale invariant.
        self.norm_out = RMSNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        assert tokens.dim() == 3 and tokens.shape[-1] == self.hidden_dim, tokens.shape
        assert xyz.dim() == 3 and xyz.shape[-1] == 3, xyz.shape
        # RoPE trig depends only on xyz, which is shared across blocks — compute once.
        cos = sin = None
        if self.use_3d_rope:
            cos, sin = self.rope(xyz)
        for block in self.blocks:
            tokens = block(tokens, xyz, cos=cos, sin=sin)
        return self.norm_out(tokens)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, D = 2, 385, 576
    torch.manual_seed(0)
    tokens = torch.randn(B, N, D, device=device)
    xyz = torch.randn(B, N, 3, device=device)

    model = GeoFormer().to(device)

    # 1. Shape
    out = model(tokens, xyz)
    assert out.shape == (B, N, D), out.shape
    print("PASS: shape [2,385,576] -> [2,385,576]")

    # 2. Finite backward
    model.zero_grad()
    loss = out.square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    print("PASS: finite backward (all grads finite)")

    # 3. bf16 forward under autocast, no NaN/Inf
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_bf16 = model(tokens, xyz)
    assert torch.isfinite(out_bf16).all()
    print("PASS: bf16 forward under autocast has no NaN/Inf")

    # 4. torch.compile matches eager (both fp32)
    with torch.no_grad():
        out_eager = model(tokens, xyz)
    compiled = torch.compile(model)
    _ = compiled(tokens, xyz)  # warmup / compile
    out_compiled = compiled(tokens, xyz)
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-4, rtol=1e-4)
    print("PASS: torch.compile matches eager within tolerance")

    # 5. Permutation test (RoPE reads position from xyz, not sequence index)
    model_fp32 = GeoFormer().to(device).eval()
    torch.manual_seed(1234)
    tok = torch.randn(B, N, D, device=device)
    xyz2 = torch.randn(B, N, 3, device=device)
    xyz2[:, 0] = 0.0  # CLS at origin

    perm = torch.randperm(N - 1, device=device)
    inv = torch.argsort(perm)
    idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device), perm + 1])
    inv_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device), inv + 1])

    tok_perm = tok[:, idx]
    xyz_perm = xyz2[:, idx]

    with torch.no_grad():
        out_orig = model_fp32(tok, xyz2)
        out_perm = model_fp32(tok_perm, xyz_perm)
    out_perm_unperm = out_perm[:, inv_idx]
    torch.testing.assert_close(out_perm_unperm, out_orig, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_perm_unperm[:, 0], out_orig[:, 0], atol=1e-5, rtol=1e-5)
    print("PASS: permutation test (patches + CLS invariant)")

    # 6. Param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    # 4 blocks x 3,988,416 + 576 (norm_out weight)
    assert n_params == 15_954_240, n_params
    print("PASS: param count == 15,954,240")
