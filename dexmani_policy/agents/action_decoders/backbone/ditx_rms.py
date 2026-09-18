"""RMS-normalized DiT-X backbone for standard rectified flow."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import RmsNorm

from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP

from .attention import CrossAttention
from .dit import Attention

WEIGHT_INIT_STD = 0.02


def _ada_rms(norm: nn.Module, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return norm(x) * (1.0 + scale.unsqueeze(1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(torch.nn.functional.silu(gate) * value)


class DiTXRMSBlock(nn.Module):
    """AdaRMS-Zero block: self-attn -> cross-attn -> SwiGLU."""

    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        ffn_dim: int,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm_sa = RmsNorm(hidden_dim, eps=1e-6)
        self.norm_ca = RmsNorm(hidden_dim, eps=1e-6)
        self.norm_ff = RmsNorm(hidden_dim, eps=1e-6)

        self.self_attn = Attention(
            dim=hidden_dim,
            num_heads=n_head,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=p_drop_attn,
            norm_layer=RmsNorm,
        )
        self.cross_attn = CrossAttention(
            dim=hidden_dim,
            num_heads=n_head,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=p_drop_attn,
            norm_layer=RmsNorm,
        )
        self.ffn = SwiGLU(hidden_dim, ffn_dim)

        # scale + gate for each residual branch. Zero-init makes each block an
        # identity map at initialization while keeping RMSNorm shift-free.
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            scale_sa,
            gate_sa,
            scale_ca,
            gate_ca,
            scale_ff,
            gate_ff,
        ) = self.modulation(time_cond).chunk(6, dim=-1)

        sa = self.self_attn(_ada_rms(self.norm_sa, x, scale_sa))
        x = x + gate_sa.unsqueeze(1) * sa

        ca = self.cross_attn(
            _ada_rms(self.norm_ca, x, scale_ca),
            context,
            mask=context_mask,
        )
        x = x + gate_ca.unsqueeze(1) * ca

        ff = self.ffn(_ada_rms(self.norm_ff, x, scale_ff))
        x = x + gate_ff.unsqueeze(1) * ff
        return x


class DiTXRMS(nn.Module):
    """Observation-token-conditioned action backbone for standard flow matching.

    The backbone intentionally does not impose a positional embedding on
    observation tokens. Spatial, temporal and modality identity belong to the
    observation encoder. Action tokens retain a learned temporal position
    embedding because their sequence order is semantic.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        obs_token_dim: int,
        timestep_embed_dim: int = 256,
        n_layers: int = 8,
        hidden_dim: int = 512,
        n_head: int = 8,
        ffn_dim: int = 1536,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if horizon <= 0 or action_dim <= 0 or obs_token_dim <= 0:
            raise ValueError("horizon, action_dim and obs_token_dim must be positive")

        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.timestep_embedder = TimestepMLP(
            pos_emb_dim=timestep_embed_dim,
            output_dim=hidden_dim,
        )

        self.blocks = nn.ModuleList(
            [
                DiTXRMSBlock(
                    hidden_dim=hidden_dim,
                    n_head=n_head,
                    ffn_dim=ffn_dim,
                    p_drop_attn=p_drop_attn,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.final_proj = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)

        nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
        if self.input_embedder.bias is not None:
            nn.init.zeros_(self.input_embedder.bias)
        nn.init.normal_(self.input_pos_embed, std=WEIGHT_INIT_STD)

        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        if self.context_embedder.bias is not None:
            nn.init.zeros_(self.context_embedder.bias)

        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Re-apply AdaRMS-Zero after the global linear initialization.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)

        nn.init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            nn.init.zeros_(self.final_proj.bias)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["input_pos_embed"],
            extra_blacklist=(RmsNorm,),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B,H,A), got {tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"context must have shape (B,L,D), got {tuple(context.shape)}"
            )
        if x.shape[1] != self.horizon or x.shape[2] != self.action_dim:
            raise ValueError(
                f"x shape {tuple(x.shape)} is incompatible with "
                f"horizon={self.horizon}, action_dim={self.action_dim}"
            )
        if context.shape[0] != x.shape[0]:
            raise ValueError("x and context batch sizes must match")

        action_tokens = self.input_embedder(x)
        action_tokens = action_tokens + self.input_pos_embed.to(
            device=x.device,
            dtype=action_tokens.dtype,
        )

        context_tokens = self.context_embedder(context)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
                [timestep],
                device=x.device,
                dtype=torch.float32,
            )
        elif timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=x.device).expand(x.shape[0])
        time_cond = self.timestep_embedder(timestep)

        for block in self.blocks:
            action_tokens = block(
                action_tokens,
                time_cond,
                context_tokens,
                context_mask=context_mask,
            )

        return self.final_proj(self.final_norm(action_tokens))
