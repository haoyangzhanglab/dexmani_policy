from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, use_fused_attn
from torch.jit import Final

from dexmani_policy.agents.action_decoders.backbone.ditx import AdaLNZero
from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP

from .dit import Attention, _approx_gelu

WEIGHT_INIT_STD = 0.02


class CrossAttentionWithCache(nn.Module):
    """Cross-attention with KV cache for fixed observation tokens during inference."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._cached_k: torch.Tensor | None = None
        self._cached_v: torch.Tensor | None = None

    def setup_kv_cache(self, obs_tokens: torch.Tensor) -> None:
        B, L, _ = obs_tokens.shape
        kv = (
            self.kv(obs_tokens)
            .reshape(B, L, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        self._cached_k = self.k_norm(k)
        self._cached_v = v

    def clear_kv_cache(self) -> None:
        self._cached_k = None
        self._cached_v = None

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        if self._cached_k is not None:
            k, v = self._cached_k, self._cached_v
        else:
            L = c.shape[1]
            kv = (
                self.kv(c)
                .reshape(B, L, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv.unbind(0)
            k = self.k_norm(k)

        if mask is not None:
            L_kv = k.shape[-2]
            mask = mask.reshape(B, 1, 1, L_kv).expand(-1, -1, N, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ActionFlowDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        temporal_heads: int,
        obs_heads: int,
        mlp_ratio: float = 2.5,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.adaLN_attn = AdaLNZero(hidden_size, hidden_size)
        self.adaLN_cross = AdaLNZero(hidden_size, hidden_size)
        self.adaLN_mlp = AdaLNZero(hidden_size, hidden_size)

        self.temporal_attn = Attention(
            hidden_size, num_heads=temporal_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop,
        )
        self.cross_attn = CrossAttentionWithCache(
            hidden_size, num_heads=obs_heads, qkv_bias=False,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim,
            act_layer=_approx_gelu, drop=0,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal_attn(self.adaLN_attn(x, t_emb))
        x = x + self.cross_attn(self.adaLN_cross(x, t_emb), obs_tokens)
        x = x + self.mlp(self.adaLN_mlp(x, t_emb))
        return x


class ActionFlowDiT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 8,
        temporal_heads: int = 2,
        obs_heads: int = 8,
        mlp_ratio: float = 2.5,
        timestep_embed_dim: int = 128,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.timestep_embedder = TimestepMLP(pos_emb_dim=timestep_embed_dim, output_dim=hidden_dim)

        self.blocks = nn.ModuleList([
            ActionFlowDiTBlock(hidden_dim, temporal_heads, obs_heads, mlp_ratio, attn_drop=attn_drop)
            for _ in range(depth)
        ])

        self.final_adaLN = AdaLNZero(hidden_dim, hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # AdaLN-Zero: override with zero-init
        for block in self.blocks:
            block.adaLN_attn.initialize_weights()
            block.adaLN_cross.initialize_weights()
            block.adaLN_mlp.initialize_weights()
        self.final_adaLN.initialize_weights()

        nn.init.normal_(self.pos_embed, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.constant_(self.input_embedder.bias, 0)

        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["pos_embed"],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_embedder(x) + self.pos_embed.to(dtype=x.dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        t_emb = self.timestep_embedder(timestep)

        for block in self.blocks:
            x = block(x, t_emb, context)

        x = self.final_adaLN(x, t_emb)
        x = self.final_linear(x)
        return x

    def setup_kv_cache(self, obs_tokens: torch.Tensor) -> None:
        for block in self.blocks:
            block.cross_attn.setup_kv_cache(obs_tokens)

    def clear_kv_cache(self) -> None:
        for block in self.blocks:
            block.cross_attn.clear_kv_cache()