"""R3D one-way attention keeps geometric position separate from observation features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TimestepMLP, add_temporal_embedding, batch_timesteps


class ProjectedAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if num_heads <= 0 or self.internal_dim % num_heads:
            raise ValueError("internal_dim must be divisible by positive num_heads")

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, C // self.num_heads)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N_heads, N_tokens, C_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, N_tokens, N_heads * C_per_head)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.split_heads(self.q_proj(q))
        k = self.split_heads(self.k_proj(k))
        v = self.split_heads(self.v_proj(v))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = self.merge_heads(out)
        return self.out_proj(out)


class R3DBlock(nn.Module):
    """SAM-style action self-attention and one-way observation cross-attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.self_attn = ProjectedAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn = ProjectedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, queries, keys, query_pe, key_pe, self_attn_mask=None):
        q = queries + query_pe
        queries = queries + self.self_attn(q=q, k=q, v=queries, attn_mask=self_attn_mask)
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn(q=q, k=k, v=keys)
        queries = self.norm2(queries)

        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)
        return queries


class R3DTransformer(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        pc_pe_dim: int,
        timestep_embed_dim: int = 128,
        embedding_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        use_aux_ee: bool = False,
        joint_dim: int | None = None,
        ee_dim: int | None = None,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.embedding_dim = embedding_dim
        self.pc_pe_dim = pc_pe_dim
        self.obs_feature_dim = obs_token_dim - pc_pe_dim
        self.use_aux_ee = use_aux_ee

        if pc_pe_dim != embedding_dim:
            raise ValueError("pc_pe_dim must equal embedding_dim for key position encoding")

        self.timestep_encoder = TimestepMLP(
            pos_emb_dim=timestep_embed_dim, output_dim=timestep_embed_dim
        )

        if use_aux_ee:
            if joint_dim is None or ee_dim is None:
                raise ValueError("Auxiliary EE prediction requires joint_dim and ee_dim")
            self.joint_action_proj = nn.Linear(joint_dim, embedding_dim)
            self.ee_action_proj = nn.Linear(ee_dim, embedding_dim)
            self.joint_output_proj = nn.Linear(embedding_dim, joint_dim)
            self.ee_output_proj = nn.Linear(embedding_dim, ee_dim)

            mask = torch.zeros(2 * horizon, 2 * horizon)
            # Joint queries must not attend to auxiliary EE tokens.
            mask[:horizon, horizon:] = float("-inf")
            self.register_buffer("self_attn_mask", mask, persistent=False)
        else:
            self.action_proj = nn.Linear(action_dim, embedding_dim)
            self.output_proj = nn.Linear(embedding_dim, action_dim)

        self.global_cond_proj = nn.Linear(timestep_embed_dim + self.obs_feature_dim, embedding_dim)

        self.temporal_pe_horizon = nn.Parameter(torch.zeros(1, horizon, embedding_dim))
        self.temporal_pe_obs = nn.Parameter(torch.zeros(1, n_obs_steps, embedding_dim))
        nn.init.normal_(self.temporal_pe_horizon, std=0.02)
        nn.init.normal_(self.temporal_pe_obs, std=0.02)

        self.layers = nn.ModuleList(
            R3DBlock(
                embedding_dim,
                num_heads,
                mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
            )
            for _ in range(depth)
        )

    def forward(self, x, timestep, context):
        B, H, _ = x.shape
        obs_feat, pc_pe = context.split((self.obs_feature_dim, self.pc_pe_dim), dim=-1)

        if self.use_aux_ee:
            joint_dim = self.joint_action_proj.in_features
            ee_dim = self.ee_action_proj.in_features
            joint = self.joint_action_proj(x[..., :joint_dim])
            ee = self.ee_action_proj(x[..., joint_dim : joint_dim + ee_dim])
            queries = torch.cat([joint, ee], dim=1)
            query_pe = self.temporal_pe_horizon.repeat(1, 2, 1).expand(B, -1, -1)
            self_attn_mask = self.self_attn_mask
        else:
            queries = self.action_proj(x)
            query_pe = self.temporal_pe_horizon[:, :H, :].expand(B, -1, -1)
            self_attn_mask = None

        timestep = batch_timesteps(timestep, x)
        t_emb = self.timestep_encoder(timestep)
        projection = self.global_cond_proj
        time_dim = t_emb.shape[-1]
        keys = F.linear(obs_feat, projection.weight[:, time_dim:]) + F.linear(
            t_emb, projection.weight[:, :time_dim], projection.bias,
        ).unsqueeze(1)

        context_pe = add_temporal_embedding(pc_pe, self.temporal_pe_obs)

        output = queries
        for layer in self.layers:
            output = layer(output, keys, query_pe, context_pe, self_attn_mask)

        if self.use_aux_ee:
            joint = self.joint_output_proj(output[:, :H])
            ee = self.ee_output_proj(output[:, H : 2 * H])
            return torch.cat([joint, ee], dim=-1)
        return self.output_proj(output)
