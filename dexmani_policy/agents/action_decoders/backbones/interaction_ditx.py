"""Interaction-conditioned action expert."""

import torch
from torch import nn
from timm.layers import GELUTanh, Mlp

from .ditx import CrossAttention, DiTX
from .modules import WEIGHT_INIT_STD, initialize_linear, modulate


class InteractionDiTXBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, p_drop_attn=0.0,
        qkv_bias=False, qk_norm=True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=p_drop_attn,
        )
        self.scene_cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias, qk_norm)
        self.interaction_cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias, qk_norm)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
            act_layer=GELUTanh, drop=0.0,
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def prepare_context_kv(self, context):
        scene, interaction = context
        return self.scene_cross_attn.prepare_kv(scene), self.interaction_cross_attn.prepare_kv(interaction)

    def forward(self, x, modulation, context_c, *, context_kv=None):
        (shift_sa, scale_sa, gate_sa, shift_scene, scale_scene, gate_scene,
         shift_inter, scale_inter, gate_inter, shift_ffn, scale_ffn, gate_ffn) = (
            modulation.unsqueeze(1).unbind(dim=2)
        )
        scene, interaction = (None, None) if context_c is None else context_c
        scene_kv, interaction_kv = (None, None) if context_kv is None else context_kv
        query = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa * self.self_attn(query, query, query, need_weights=False)[0]
        query = modulate(self.norm2(x), shift_scene, scale_scene)
        x = x + gate_scene * self.scene_cross_attn(query, scene, context_kv=scene_kv)
        query = modulate(self.norm3(x), shift_inter, scale_inter)
        x = x + gate_inter * self.interaction_cross_attn(query, interaction, context_kv=interaction_kv)
        return x + gate_ffn * self.mlp(modulate(self.norm4(x), shift_ffn, scale_ffn))


class InteractionDiTX(DiTX):
    """InteractionFlow action expert with global-first serial conditioning and time-only AdaLN."""

    block_cls = InteractionDiTXBlock
    num_modulations = 12

    def __init__(
        self, *, n_obs_steps: int, obs_token_dim: int, num_scene_tokens: int,
        hidden_dim: int = 768, **kwargs,
    ) -> None:
        super().__init__(
            n_obs_steps=n_obs_steps, obs_token_dim=obs_token_dim, hidden_dim=hidden_dim, **kwargs,
        )
        self.condition_counts = (n_obs_steps * num_scene_tokens, n_obs_steps, n_obs_steps * 6)
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.interaction_projection = nn.Linear(obs_token_dim, hidden_dim)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        self.interaction_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.role_embeddings = nn.Parameter(torch.empty(1, 6, hidden_dim))
        self.state_projection = nn.Linear(obs_token_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.interaction_projection.apply(initialize_linear)
        self.state_projection.apply(initialize_linear)
        nn.init.normal_(self.role_embeddings, std=WEIGHT_INIT_STD)

    def _encode_condition(self, context):
        scene, state, interaction = context.split(self.condition_counts, dim=1)
        scene = self.scene_norm(self.embed_context(scene))
        state = self.state_projection(state)
        state = self.state_norm(state + self.context_frame_pos_embed.to(state.dtype))
        global_context = torch.cat((scene, state), dim=1)
        interaction = interaction.unflatten(1, (self.interaction_frame_pos_embed.shape[1], 6))
        interaction = self.interaction_projection(interaction)
        interaction = self.interaction_norm(
            interaction + self.interaction_frame_pos_embed.to(interaction.dtype).unsqueeze(2)
            + self.role_embeddings.to(interaction.dtype).unsqueeze(1)
        ).flatten(1, 2)
        return global_context, interaction

    def sampling_modulations(self, timesteps, chunk_size):
        # Bound the per-step modulation tensor when the caller requests a longer Euler grid.
        for chunk in timesteps.split(chunk_size):
            time = self.timestep_embedder(chunk)
            yield from self.adaln_modulation(time).unsqueeze(1).unbind(0)
