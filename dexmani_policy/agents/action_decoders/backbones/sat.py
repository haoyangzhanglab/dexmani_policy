"""Structural Action Transformer over joint trajectories, with flow-matching I/O."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp

from .modules import (
    DiTTimestepEmbedding,
    add_temporal_embedding,
    approximate_gelu,
    batch_timesteps,
    initialize_linear,
    modulate,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class StructuralAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, p_drop_attn: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = 64
        self.inner_dim = num_heads * self.head_dim
        self.p_drop_attn = p_drop_attn
        self.rmsnorm = nn.RMSNorm(dim)
        self.to_kv = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.to_actions_qkvg = nn.Linear(dim, 4 * self.inner_dim, bias=False)
        self.to_actions_out = nn.Linear(self.inner_dim, dim, bias=False)

    def prepare_kv(self, context):
        batch, length, _ = context.shape
        kv = self.to_kv(self.rmsnorm(context))
        return kv.reshape(batch, length, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

    def forward(self, actions, context, *, context_kv=None):
        batch, length, _ = actions.shape
        qkvg = self.to_actions_qkvg(actions)
        q, k, v, gate = qkvg.reshape(
            batch, length, 4, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        context_k, context_v = self.prepare_kv(context) if context_kv is None else context_kv
        k = torch.cat((context_k, k), dim=2)
        v = torch.cat((context_v, v), dim=2)

        scores = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)
        attention = (50.0 * torch.tanh(scores / 50.0)).softmax(dim=-1)
        attention = F.dropout(attention, self.p_drop_attn, training=self.training)
        update = (attention @ v) * gate.sigmoid()
        update = update.transpose(1, 2).reshape(batch, length, self.inner_dim)
        return self.to_actions_out(update)


class SATBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, p_drop_attn: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = StructuralAttention(dim, num_heads, p_drop_attn)
        self.norm2 = RMSNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=approximate_gelu, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, actions, context, time_c, *, context_kv=None):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(time_c).chunk(6, dim=-1)
        )
        query = modulate(self.norm1(actions), shift_attn, scale_attn)
        actions = actions + gate_attn[:, None] * self.attn(query, context, context_kv=context_kv)
        return actions + gate_mlp[:, None] * self.mlp(modulate(self.norm2(actions), shift_mlp, scale_mlp))


class SATOutput(nn.Module):
    def __init__(self, dim: int, horizon: int):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, horizon)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, actions, time_c):
        # The official final layer uses chunk 0 as scale and chunk 1 as shift.
        scale, shift = self.adaLN_modulation(time_c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(actions), shift, scale))


class SAT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        timestep_embed_dim: int = 256,
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim != 19 or hidden_dim % 4:
            raise ValueError("SAT requires 19 xArm7/XHand joints and hidden_dim divisible by 4")
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.hidden_dim = hidden_dim

        self.x_embedder = nn.Sequential(nn.Linear(horizon, 64), nn.Mish(), nn.Linear(64, hidden_dim // 2))
        self.robot_embed = nn.Embedding(2, hidden_dim // 4)
        self.joint_embed = nn.Embedding(12, hidden_dim // 4)
        self.register_buffer("robot_ids", torch.tensor([0] * 7 + [1] * 12), persistent=False)
        self.register_buffer("joint_ids", torch.tensor([*range(7), *range(12)]), persistent=False)

        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.t_embedder = DiTTimestepEmbedding(hidden_dim, timestep_embed_dim)
        self.blocks = nn.ModuleList(
            SATBlock(hidden_dim, n_head, mlp_ratio, p_drop_attn) for _ in range(n_layers)
        )
        self.final_layer = SATOutput(hidden_dim, horizon)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(initialize_linear)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def embed_context(self, context):
        tokens = self.context_embedder(context)
        return add_temporal_embedding(tokens, self.context_frame_pos_embed.to(tokens.dtype))

    def prepare_context_kv(self, context):
        if self.training:
            return None
        tokens = self.embed_context(context)
        return tuple(block.attn.prepare_kv(tokens) for block in self.blocks)

    def forward(self, x, timestep, context, *, context_kv=None):
        trajectory = self.x_embedder(x.transpose(1, 2))
        descriptor = torch.cat((self.robot_embed(self.robot_ids), self.joint_embed(self.joint_ids)), dim=-1)
        actions = torch.cat((trajectory, descriptor.to(trajectory.dtype).expand(x.shape[0], -1, -1)), dim=-1)
        time_c = self.t_embedder(batch_timesteps(timestep, x, dtype=torch.float32))
        tokens = self.embed_context(context) if context_kv is None else None
        for i, block in enumerate(self.blocks):
            actions = block(actions, tokens, time_c, context_kv=None if context_kv is None else context_kv[i])
        return self.final_layer(actions, time_c).transpose(1, 2)
