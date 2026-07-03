"""OneWayTransformer backbone for R3D action decoder.

Cross-attention between noisy action tokens (queries) and dense geometric
observation tokens (keys). pc_pe is split from context before global_cond_proj
and added to key positional encoding after projection, exactly matching R3D.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from dexmani_policy.agents.optim_util import OptimGroupMixin
from dexmani_policy.agents.position_encodings import TimestepMLP

def _is_cuda_available() -> bool:
    return torch.cuda.is_available()

class MLPBlock(nn.Module):
    """Linear -> GELU -> Linear."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Attention(nn.Module):
    """Multi-head attention with SDPA (Flash/Mem-Efficient) + manual fallback."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1,
                 attn_drop: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, \
            f"num_heads ({num_heads}) must divide internal_dim ({self.internal_dim})"

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.fused_attn = _is_cuda_available()
        self.attn_drop = attn_drop

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, C // self.num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N_heads, N_tokens, C_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, N_tokens, N_heads * C_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)
            attn = attn / math.sqrt(c_per_head)
            if attn_mask is not None:
                attn = attn + attn_mask  # additive float mask (-inf = masked)
            attn = torch.softmax(attn, dim=-1)
            out = attn @ v

        out = self._recombine_heads(out)
        return self.out_proj(out)

class OneWayAttentionBlock(nn.Module):
    """Self-Attn → Cross-Attn → MLP, with residual + LayerNorm (adapted from SAM)."""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, queries, keys, query_pe, key_pe,
                self_attn_mask=None):
        # Self-attention on queries
        q = queries + query_pe
        queries = queries + self.self_attn(
            q=q, k=q, v=queries, attn_mask=self_attn_mask)
        queries = self.norm1(queries)

        # Cross-attention: queries attend to keys (no mask — all queries can see all obs tokens)
        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm2(queries)

        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)
        return queries, keys

class OneWayTransformer(nn.Module):
    """Stack of OneWayAttentionBlock layers."""

    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            OneWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
            )
            for _ in range(depth)
        ])
    def forward(self, global_feature_embeded: torch.Tensor,
                global_pe: torch.Tensor,
                sample_embedded: torch.Tensor,
                sample_pe: torch.Tensor,
                self_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        queries = sample_embedded
        keys = global_feature_embeded

        for layer in self.layers:
            queries, keys = layer(
                queries=queries, keys=keys,
                query_pe=sample_pe, key_pe=global_pe,
                self_attn_mask=self_attn_mask,
            )

        return queries

class OneWayTransformerBackbone(OptimGroupMixin, nn.Module):
    """R3D action decoder: OneWayTransformer with timestep + temporal PE + pc_pe routing.

    forward(x, timestep, context) -> (B, horizon, action_dim)

    Token layout (H = horizon):
      standard:    H tokens  (joint only)
      use_aux_ee: 2H tokens  (joint + EE wrist pose)

    Cascading self-attention mask: joint sees only itself; EE sees
    joint + EE (hierarchical, prevents information leakage).
    """

    def __init__(self, horizon: int, action_dim: int, n_obs_steps: int,
                 num_obs_tokens: int, obs_token_dim: int,
                 pc_pe_dim: int,
                 timestep_embed_dim: int = 128,
                 embedding_dim: int = 256,
                 depth: int = 4,
                 num_heads: int = 8,
                 mlp_dim: int = 2048,
                 attention_downsample_rate: int = 2,
                 use_aux_ee: bool = False,
                 joint_dim: int | None = None,
                 ee_dim: int | None = None):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.embedding_dim = embedding_dim
        self.pc_pe_dim = pc_pe_dim
        self._obs_feat_dim = obs_token_dim - pc_pe_dim
        self.use_aux_ee = use_aux_ee

        # Shape contract: pc_pe_dim must equal embedding_dim for key_pe addition
        assert pc_pe_dim == embedding_dim, (
            f"OneWayTransformerBackbone: pc_pe_dim ({pc_pe_dim}) must equal "
            f"embedding_dim ({embedding_dim}) for key_pe addition."
        )
        assert num_obs_tokens % n_obs_steps == 0, (
            f"OneWayTransformerBackbone: num_obs_tokens ({num_obs_tokens}) must be "
            f"divisible by n_obs_steps ({n_obs_steps})."
        )

        self.timestep_encoder = TimestepMLP(
            pos_emb_dim=timestep_embed_dim, output_dim=timestep_embed_dim)

        # ── Build per-head projections (1 or 2 heads) ──
        self.joint_dim = joint_dim
        self.ee_dim = ee_dim if use_aux_ee else None

        has_aux = use_aux_ee
        n_heads = 1 + int(use_aux_ee)
        n_tokens = n_heads * horizon

        # Input projections
        self.joint_action_proj = nn.Linear(joint_dim, embedding_dim) if has_aux else None
        self.ee_action_proj = nn.Linear(ee_dim, embedding_dim) if use_aux_ee else None

        # Output projections
        self.joint_output_proj = nn.Linear(embedding_dim, joint_dim) if has_aux else None
        self.ee_output_proj = nn.Linear(embedding_dim, ee_dim) if use_aux_ee else None

        # Single-projection layers: only used in standard (no-aux) mode
        self.action_proj = None if has_aux else nn.Linear(action_dim, embedding_dim)
        self.output_proj = None if has_aux else nn.Linear(embedding_dim, action_dim)

        # ── Cascading self-attention mask ──
        # Joint (primary) sees only itself; EE sees joint + itself.
        if use_aux_ee:
            mask = torch.zeros(n_tokens, n_tokens)
            # EE (tokens H:2H) blocked from attending to joint (tokens 0:H)? No —
            # joint blocked from EE; EE CAN see joint. So block joint from EE:
            mask[:horizon, horizon:] = float('-inf')
            self.register_buffer('self_attn_mask', mask, persistent=False)
        else:
            self.register_buffer('self_attn_mask', torch.empty(0), persistent=False)

        self.global_cond_proj = nn.Linear(
            timestep_embed_dim + self._obs_feat_dim, embedding_dim
        )

        self.temporal_pe_horizon = nn.Parameter(
            torch.zeros(1, horizon, embedding_dim)
        )
        self.temporal_pe_obs = nn.Parameter(
            torch.zeros(1, n_obs_steps, embedding_dim)
        )
        nn.init.normal_(self.temporal_pe_horizon, std=0.02)
        nn.init.normal_(self.temporal_pe_obs, std=0.02)

        self.transformer = OneWayTransformer(
            depth=depth,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attention_downsample_rate=attention_downsample_rate,
        )

    def forward(self, x, timestep, context):
        B, H, _ = x.shape
        N = context.shape[1]
        T = self.n_obs_steps
        K = N // T

        # Defensive: catch token misalignment early with a clear message.
        assert K * T == N, (
            f"OneWayTransformerBackbone: context tokens N={N} not divisible by "
            f"n_obs_steps T={T} (K={K})."
        )
        assert context.shape[-1] == self._obs_feat_dim + self.pc_pe_dim, (
            f"OneWayTransformerBackbone: context last dim {context.shape[-1]} != "
            f"_obs_feat_dim ({self._obs_feat_dim}) + pc_pe_dim ({self.pc_pe_dim})."
        )

        obs_feat = context[..., :self._obs_feat_dim]
        pc_pe    = context[..., self._obs_feat_dim:]

        # ── Action query construction (composable: 1 or 2 heads) ──
        if self.action_proj is not None:
            # Standard mode: single projection
            queries = self.action_proj(x)                  # (B, H, E)
            query_pe = self.temporal_pe_horizon[:, :H, :].expand(B, -1, -1)
            self_attn_mask = None
        else:
            # Aux mode: project each head separately
            offset = 0
            query_parts = []
            # Head 0: joint
            query_parts.append(self.joint_action_proj(x[..., offset:offset + self.joint_dim]))
            offset += self.joint_dim
            # Head 1: EE wrist pose
            if self.ee_action_proj is not None:
                query_parts.append(self.ee_action_proj(x[..., offset:offset + self.ee_dim]))
                offset += self.ee_dim

            n_heads = len(query_parts)
            queries = torch.cat(query_parts, dim=1)         # (B, n_heads*H, E)
            query_pe = self.temporal_pe_horizon.repeat(1, n_heads, 1).expand(B, -1, -1)
            self_attn_mask = self.self_attn_mask             # (n_heads*H, n_heads*H)

        # ── Timestep encoding ──
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B)

        t_emb = self.timestep_encoder(timestep)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)
        global_feat = torch.cat([t_emb, obs_feat], dim=-1)
        keys = self.global_cond_proj(global_feat)

        # ── Key positional encoding (unchanged) ──
        temporal_pe = self.temporal_pe_obs[:, :T, :]
        temporal_pe = temporal_pe.repeat_interleave(K, dim=1)
        key_pe = temporal_pe + pc_pe

        # ── Transformer ──
        output = self.transformer(
            keys, key_pe, queries, query_pe,
            self_attn_mask=self_attn_mask,
        )

        # ── Output projection (composable: 1 or 2 heads) ──
        if self.output_proj is not None:
            return self.output_proj(output)
        else:
            out_parts = []
            # Head 0: joint
            out_parts.append(self.joint_output_proj(output[:, :H, :]))
            # Head 1: EE wrist pose
            if self.ee_output_proj is not None:
                out_parts.append(self.ee_output_proj(output[:, H:2*H, :]))
            return torch.cat(out_parts, dim=-1)

    _optim_no_decay_names = ("temporal_pe_horizon", "temporal_pe_obs")
