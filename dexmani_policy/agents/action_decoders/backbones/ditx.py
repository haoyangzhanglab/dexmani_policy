"""DiT-X backbones for single-time and consistency flow matching."""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp, RmsNorm

from dexmani_policy.common.config import positive_int

from .modules import (
    TimestepMLP,
    WEIGHT_INIT_STD,
    add_temporal_embedding,
    approximate_gelu,
    batch_timesteps,
    initialize_linear,
    modulate,
)


class ContextAdaLN(nn.Module):
    """ManiFlow applies SiLU before identity-initialized affine conditioning."""

    def __init__(self, dim, cond_dim):
        super().__init__()

        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_linear = nn.Linear(cond_dim, dim * 2)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[: self.dim], 1.0)
        nn.init.zeros_(self.cond_linear.bias[self.dim :])

    def forward(self, x, cond):
        x = self.norm(x)
        gamma, beta = self.cond_linear(F.silu(cond).unsqueeze(1)).chunk(2, dim=-1)
        x = x * gamma + beta
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def prepare_kv(self, context):
        batch, length, _ = context.shape
        kv = (
            self.kv(context)
            .reshape(batch, length, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        return self.k_norm(k), v

    def forward(self, x, context, *, context_kv=None):
        batch, length, channels = x.shape
        q = self.q(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.prepare_kv(context) if context_kv is None else context_kv
        x = F.scaled_dot_product_attention(self.q_norm(q), k, v)
        return self.proj(x.transpose(1, 2).reshape(batch, length, channels))


class DiTXBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        p_drop_attn=0.0,
        qkv_bias=False,
        qk_norm=False,
    ):
        super().__init__()

        # ManiFlow self-attention keeps biased QKV; QK options apply only to cross-attention.
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=p_drop_attn,
        )

        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approximate_gelu,
            drop=0.0,
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        modulation_size = 9 * hidden_size
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, modulation_size, bias=True)
        )

    def forward(self, x, time_c, context_c, *, context_kv=None):
        modulation = self.adaln_modulation(time_c)

        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output = self.self_attn(normed_x, normed_x, normed_x, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, context_kv=context_kv)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output

        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class DiTXOutput(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()

        self.norm_final = RmsNorm(hidden_size, eps=1e-6)

        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approximate_gelu,
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class DiTX(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        timestep_embed_dim: int = 128,
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        positive_int(n_obs_steps, "n_obs_steps")

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.obs_token_dim = obs_token_dim
        self.hidden_dim = hidden_dim

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.timestep_embedder = TimestepMLP(timestep_embed_dim, hidden_dim)
        self.ditx_blocks = nn.ModuleList(
            DiTXBlock(
                hidden_size=hidden_dim,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                p_drop_attn=p_drop_attn,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            )
            for _ in range(n_layers)
        )
        self.final_layer = DiTXOutput(hidden_dim, action_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.apply(initialize_linear)
        for block in self.ditx_blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)

        nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.input_pos_embed, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.context_frame_pos_embed)
        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.weight)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.bias)

    def embed_context(self, context):
        tokens = self.context_embedder(context)
        return add_temporal_embedding(tokens, self.context_frame_pos_embed.to(tokens.dtype))

    def prepare_context_kv(self, context):
        if self.training:
            return None
        context_c = self.embed_context(context)
        return tuple(block.cross_attn.prepare_kv(context_c) for block in self.ditx_blocks)

    def forward(
        self, x: torch.Tensor, timestep, context: torch.Tensor, *, context_kv=None
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.horizon:
            raise ValueError("actions must have shape [B, horizon, action_dim]")
        if context.shape[0] != x.shape[0]:
            raise ValueError("actions and context must have the same batch size")
        context_c = self.embed_context(context) if context_kv is None else None
        timestep = batch_timesteps(timestep, x, dtype=torch.float32)
        time_c = self.timestep_embedder(timestep)

        x = self.input_embedder(x)
        x = x + self.input_pos_embed.to(dtype=x.dtype)
        for index, block in enumerate(self.ditx_blocks):
            x = block(
                x,
                time_c,
                context_c,
                context_kv=None if context_kv is None else context_kv[index],
            )
        return self.final_layer(x)


class ConsistencyDiTX(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        timestep_embed_dim: int = 128,
        target_t_embed_dim: int = 128,
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        pre_norm_modality: bool = False,
    ):
        super().__init__()
        positive_int(n_obs_steps, "n_obs_steps")

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.pre_norm_modality = pre_norm_modality

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        if self.pre_norm_modality:
            self.context_norm = ContextAdaLN(dim=hidden_dim, cond_dim=hidden_dim)

        self.timestep_embedder = TimestepMLP(pos_emb_dim=timestep_embed_dim, output_dim=hidden_dim)
        self.target_t_embedder = TimestepMLP(pos_emb_dim=target_t_embed_dim, output_dim=hidden_dim)

        self.timestep_and_target_t_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.ditx_blocks = nn.ModuleList(
            [
                DiTXBlock(
                    hidden_size=hidden_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    p_drop_attn=p_drop_attn,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer = DiTXOutput(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.ditx_blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

        self.apply(initialize_linear)

        for block in self.ditx_blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)

        if self.pre_norm_modality:
            self.context_norm.initialize_weights()

        nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.input_pos_embed, std=WEIGHT_INIT_STD)

        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.context_frame_pos_embed)

        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in self.target_t_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.normal_(self.timestep_and_target_t_fusion.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.timestep_and_target_t_fusion.bias)

        nn.init.zeros_(self.final_layer.ffn_final.fc2.weight)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.bias)

    def embed_context(self, context):
        tokens = self.context_embedder(context)
        return add_temporal_embedding(tokens, self.context_frame_pos_embed.to(tokens.dtype))

    def prepare_context_kv(self, context):
        # The caller owns this cache for one eval sample with time-independent context.
        if self.training or self.pre_norm_modality:
            return None
        context_c = self.embed_context(context)
        return tuple(block.cross_attn.prepare_kv(context_c) for block in self.ditx_blocks)

    def forward(self, x, timestep, target_t, context, *, context_kv=None):
        x = self.input_embedder(x) + self.input_pos_embed.to(dtype=x.dtype)

        timestep = batch_timesteps(timestep, x)
        timestep_embed = self.timestep_embedder(timestep)

        target_t = batch_timesteps(target_t, x)
        target_t_embed = self.target_t_embedder(target_t)

        time_c = self.timestep_and_target_t_fusion(
            torch.cat([timestep_embed, target_t_embed], dim=-1)
        )

        context_c = self.embed_context(context) if context_kv is None else None
        if self.pre_norm_modality:
            context_c = self.context_norm(context_c, time_c)

        for index, block in enumerate(self.ditx_blocks):
            x = block(
                x,
                time_c,
                context_c,
                context_kv=None if context_kv is None else context_kv[index],
            )

        x = self.final_layer(x)

        x = x[:, -self.horizon :]
        return x
