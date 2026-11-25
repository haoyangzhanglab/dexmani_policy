"""DiT-X backbones for single-time and consistency flow matching."""

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import GELUTanh, Mlp, RmsNorm

from .modules import (
    TimestepMLP,
    WEIGHT_INIT_STD,
    add_temporal_embedding,
    batch_timesteps,
    initialize_linear,
    modulate,
)


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


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, n_layers * 9 * hidden_dim)
        self.layer_shape = (n_layers, 9, hidden_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.projection(F.silu(condition)).unflatten(-1, self.layer_shape)


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

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=GELUTanh,
            drop=0.0,
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, modulation, context_c, *, context_kv=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.unsqueeze(1).unbind(dim=2)

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output = self.self_attn(
            normed_x, normed_x, normed_x, need_weights=False,
        )[0]
        x = x + gate_msa * self_attn_output

        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, context_kv=context_kv)
        x = x + gate_cross * cross_attn_output

        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp * mlp_output

        return x


class DiTXOutput(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()

        self.norm_final = RmsNorm(hidden_size, eps=1e-6)

        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=output_dim,
            act_layer=GELUTanh,
            drop=0,
        )

    def forward(self, x):
        return self.ffn_final(self.norm_final(x))


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
        action_io: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.action_io = action_io
        if self.action_io is None:
            self.input_embedder = nn.Linear(action_dim, hidden_dim)
            self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.timestep_embedder = TimestepMLP(timestep_embed_dim, hidden_dim)
        self.adaln_modulation = AdaLNModulation(hidden_dim, n_layers)
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
        if self.action_io is None:
            self.final_layer = DiTXOutput(hidden_dim, action_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Action I/O owns its initialization, including the zero velocity head.
        for module in self.children():
            if module is not self.action_io:
                module.apply(initialize_linear)
        for block in self.ditx_blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
        nn.init.zeros_(self.adaln_modulation.projection.weight)

        if self.action_io is None:
            nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
            nn.init.normal_(self.input_pos_embed, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.context_frame_pos_embed)
        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
        if self.action_io is None:
            nn.init.zeros_(self.final_layer.ffn_final.fc2.weight)

    def embed_context(self, context):
        tokens = self.context_embedder(context)
        return add_temporal_embedding(tokens, self.context_frame_pos_embed.to(tokens.dtype))

    def _encode_condition(self, context):
        return self.embed_context(context), None

    def prepare_context_kv(self, context):
        if self.training:
            return None
        memory, state = self._encode_condition(context)
        return state, tuple(block.cross_attn.prepare_kv(memory) for block in self.ditx_blocks)

    def _forward_hidden(
        self, x: torch.Tensor, timestep, context: torch.Tensor, *, context_kv=None, modulations=None,
    ) -> torch.Tensor:
        if context.shape[0] != x.shape[0]:
            raise ValueError("actions and context must have the same batch size")
        if context_kv is None:
            memory, state = self._encode_condition(context)
            layer_kv = None
        else:
            state, layer_kv = context_kv
            memory = None
        if modulations is None:
            time = self.timestep_embedder(batch_timesteps(timestep, x, dtype=torch.float32))
            modulations = self.adaln_modulation(time if state is None else time + state)

        if self.action_io is None:
            x = self.input_embedder(x)
            x = x + self.input_pos_embed[:, :x.shape[1]].to(dtype=x.dtype)
        else:
            x = self.action_io.encode(x)
        for index, (block, modulation) in enumerate(zip(self.ditx_blocks, modulations.unbind(1))):
            x = block(
                x, modulation, memory,
                context_kv=None if layer_kv is None else layer_kv[index],
            )
        return x  # [B, horizon, hidden_dim]

    def forward(
        self, x: torch.Tensor, timestep, context: torch.Tensor, *, context_kv=None, modulations=None,
    ) -> torch.Tensor:
        x = self._forward_hidden(x, timestep, context, context_kv=context_kv, modulations=modulations)
        if self.action_io is None:
            return self.final_layer(x)
        return self.action_io.decode_velocity(x)


class FutureInteractionHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.action_projection = nn.Linear(hidden_dim, latent_dim)
        self.queries = nn.Parameter(torch.empty(1, 6, latent_dim))
        self.attention = CrossAttention(latent_dim, num_heads=4)
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim), nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.apply(initialize_linear)
        nn.init.normal_(self.queries, std=WEIGHT_INIT_STD)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        memory = self.action_projection(hidden)  # [B, 16, 192]
        queries = self.queries.expand(hidden.shape[0], -1, -1)  # [B, 6, 192], wrist first
        return self.mlp(self.norm(self.attention(queries, memory)))


class InteractionDiTX(DiTX):
    """DAI-Flow action expert with unified scene/interaction memory."""

    def __init__(
        self, *, n_obs_steps: int, obs_token_dim: int, num_scene_tokens: int,
        hidden_dim: int = 768, **kwargs,
    ) -> None:
        super().__init__(
            n_obs_steps=n_obs_steps, obs_token_dim=obs_token_dim, hidden_dim=hidden_dim, **kwargs,
        )
        self.condition_counts = (n_obs_steps * num_scene_tokens, 1, 6)
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.interaction_projection = nn.Linear(obs_token_dim, hidden_dim)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        self.role_embeddings = nn.Parameter(torch.empty(1, 6, hidden_dim))
        self.state_mlp = nn.Sequential(
            nn.Linear(obs_token_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.interaction_projection.apply(initialize_linear)
        self.state_mlp.apply(initialize_linear)
        nn.init.normal_(self.role_embeddings, std=WEIGHT_INIT_STD)
        self.future_head = FutureInteractionHead(hidden_dim, obs_token_dim)

    def _encode_condition(self, context):
        scene, state, interaction = context.split(self.condition_counts, dim=1)
        scene = self.scene_norm(self.embed_context(scene))  # [B, 256, 768]
        interaction = self.interaction_projection(interaction)
        interaction = self.interaction_norm(interaction + self.role_embeddings.to(interaction.dtype))
        return torch.cat((scene, interaction), dim=1), self.state_mlp(state[:, 0])

    def forward(
        self, x: torch.Tensor, timestep, context: torch.Tensor, *,
        context_kv=None, modulations=None, return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self._forward_hidden(
            x, timestep, context, context_kv=context_kv, modulations=modulations,
        )
        velocity = self.final_layer(hidden)
        if return_aux:
            return velocity, self.future_head(hidden)
        return velocity

    def sampling_modulations(self, timesteps, context_kv, chunk_size):
        state, _ = context_kv
        # Bound the per-step modulation tensor when the caller requests a longer Euler grid.
        for chunk in timesteps.split(chunk_size):
            time = self.timestep_embedder(chunk).unsqueeze(1)
            yield from self.adaln_modulation(time + state).unbind(0)


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
    ):
        super().__init__()

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.timestep_embedder = TimestepMLP(pos_emb_dim=timestep_embed_dim, output_dim=hidden_dim)
        self.target_t_embedder = TimestepMLP(pos_emb_dim=target_t_embed_dim, output_dim=hidden_dim)

        self.timestep_and_target_t_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.adaln_modulation = AdaLNModulation(hidden_dim, n_layers)

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

        nn.init.zeros_(self.adaln_modulation.projection.weight)

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

    def embed_context(self, context):
        tokens = self.context_embedder(context)
        return add_temporal_embedding(tokens, self.context_frame_pos_embed.to(tokens.dtype))

    def prepare_context_kv(self, context):
        # The caller owns this cache for one eval sample with time-independent context.
        if self.training:
            return None
        context_c = self.embed_context(context)
        return tuple(block.cross_attn.prepare_kv(context_c) for block in self.ditx_blocks)

    def forward(self, x, timestep, target_t, context, *, context_kv=None):
        x = self.input_embedder(x)
        x = x + self.input_pos_embed.to(dtype=x.dtype)

        timestep = batch_timesteps(timestep, x)
        timestep_embed = self.timestep_embedder(timestep)

        target_t = batch_timesteps(target_t, x)
        target_t_embed = self.target_t_embedder(target_t)

        time_c = self.timestep_and_target_t_fusion(
            torch.cat([timestep_embed, target_t_embed], dim=-1)
        )

        context_c = self.embed_context(context) if context_kv is None else None
        modulations = self.adaln_modulation(time_c)
        for index, (block, modulation) in enumerate(zip(self.ditx_blocks, modulations.unbind(1))):
            x = block(
                x,
                modulation,
                context_c,
                context_kv=None if context_kv is None else context_kv[index],
            )

        return self.final_layer(x)
