"""SAT (Structural Action Transformer) backbone.

Structural-centric action representation: each Transformer token represents one
joint's full future trajectory (Da as sequence length, T as per-token feature).

Implements:
- ``EmbodiedJointCodebook``: 3-field summed embedding for joint identity (EJC)
- ``MultiModalAttention``: single concatenated attention with obs-as-KV-prefix mask
- ``SATBlock``: AdaLN-modulated block with MultiModalAttention + MLP
- ``SATBackbone``: full backbone with axis transposition and shuffle support

Reference: "Structural Action Transformer for 3D Dexterous Manipulation", CVPR 2026.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm

from dexmani_policy.agents.action_decoders.backbone.dit import (
    WEIGHT_INIT_STD,
    _approx_gelu,
    modulate,
)
from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP

# ---------------------------------------------------------------------------
# Embodied Joint Codebook (EJC)
# ---------------------------------------------------------------------------


class EmbodiedJointCodebook(nn.Module):
    """3-field summed embedding providing per-joint structural identity.

    Paper spec (Sec 3.2):
      C_j = E_emb(embodiment_j) + E_func(function_j) + E_axis(axis_j)

    Each joint's identity is the sum of three separately-projected
    embedding vectors, encoding *which* robot part, *what* functional
    role, and *which* axis of motion it represents.

    Defaults assign a unique function ID per joint and a single
    embodiment/axis type, making this a learned per-joint positional
    encoding that can be extended to cross-embodiment settings.
    """

    def __init__(
        self,
        num_joints: int,
        hidden_dim: int,
        num_embodiments: int = 1,
        num_functions: int | None = None,
        num_axes: int = 1,
        embodiment_dim: int = 8,
        function_dim: int = 32,
        axis_dim: int = 8,
    ):
        super().__init__()
        if num_functions is None:
            num_functions = num_joints

        self.hidden_dim = hidden_dim

        # Three embedding tables (one per field)
        self.emb_emb = nn.Embedding(num_embodiments, embodiment_dim)
        self.func_emb = nn.Embedding(num_functions, function_dim)
        self.axis_emb = nn.Embedding(num_axes, axis_dim)

        # Project each field to hidden_dim
        self.proj_emb = nn.Linear(embodiment_dim, hidden_dim)
        self.proj_func = nn.Linear(function_dim, hidden_dim)
        self.proj_axis = nn.Linear(axis_dim, hidden_dim)

        # Per-joint type assignments (configurable via state_dict / manual set)
        self.register_buffer("joint_embodiment", torch.zeros(num_joints, dtype=torch.long))
        self.register_buffer("joint_function", torch.arange(num_joints, dtype=torch.long))
        self.register_buffer("joint_axis", torch.zeros(num_joints, dtype=torch.long))

    def forward(self) -> torch.Tensor:
        """Return ``(num_joints, hidden_dim)`` — one embedding per action token."""
        emb = self.proj_emb(self.emb_emb(self.joint_embodiment))
        func = self.proj_func(self.func_emb(self.joint_function))
        axis = self.proj_axis(self.axis_emb(self.joint_axis))
        return emb + func + axis  # sum, NOT concat (paper spec)


# ---------------------------------------------------------------------------
# MultiModalAttention — obs-as-prefix, bidirectional action
# ---------------------------------------------------------------------------


class MultiModalAttention(nn.Module):
    """Single concatenated attention with observation-as-KV-prefix masking.

    Obs tokens and action tokens are concatenated into one sequence.
    The attention mask enforces:
      - Obs tokens: attend only to other obs tokens (unidirectional prefix)
      - Action tokens: attend to ALL tokens (bidirectional, both obs and self)

    This replaces the DiTX pattern of separate self-attention + cross-attention
    with a single attention pass.  Both obs and action tokens are updated.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        c_obs: torch.Tensor,
        x_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            c_obs: ``(B, N_obs, dim)`` — observation prefix tokens
            x_action: ``(B, Da, dim)`` — action tokens

        Returns:
            ``(x_action_out, c_obs_out)`` — both updated
        """
        B, N_obs, _ = c_obs.shape
        _, Da, C = x_action.shape

        # Concatenate: [obs | action]
        combined = torch.cat([c_obs, x_action], dim=1)  # (B, N_obs+Da, C)

        # Single QKV projection
        qkv = (
            self.qkv(combined).reshape(B, N_obs + Da, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N_obs+Da, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        # Build observation-prefix attention mask as float (additive).
        #   obs rows [0:N_obs, :N_obs]   = 0.0  (allow obs→obs)
        #   obs rows [0:N_obs, N_obs:]   = -inf (mask obs→action)
        #   action rows [N_obs:, :]      = 0.0  (allow action→all)
        total = N_obs + Da
        mask = torch.zeros(total, total, dtype=combined.dtype, device=combined.device)
        mask[:N_obs, N_obs:] = float("-inf")

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        # Merge heads
        x = x.transpose(1, 2).reshape(B, total, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Split back
        c_obs_out = x[:, :N_obs, :]
        x_action_out = x[:, N_obs:, :]

        return x_action_out, c_obs_out


# ---------------------------------------------------------------------------
# SATBlock — AdaLN-modulated block
# ---------------------------------------------------------------------------


class SATBlock(nn.Module):
    """One SAT Transformer block with AdaLN modulation.

    MultiModalAttention updates *both* obs and action tokens via the
    prefix attention mask.  The MLP only updates action tokens (obs
    passes through unchanged from attention output).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.0,
    ):
        super().__init__()

        self.attn = MultiModalAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=p_drop_attn,
            proj_drop=p_drop_attn,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=_approx_gelu,
            drop=0.0,
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.obs_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 6-chunk AdaLN: shift/scale/gate × (attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x_action: torch.Tensor,
        c_obs: torch.Tensor,
        time_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_action: ``(B, Da, hidden_size)``
            c_obs: ``(B, N_obs, hidden_size)``
            time_c: ``(B, hidden_size)`` — time conditioning

        Returns:
            ``(x_action, c_obs)`` — action updated via attn+MLP,
            obs updated via attn only
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_c).chunk(
            6, dim=-1
        )

        # Attention: updates BOTH action and obs
        x_norm = modulate(self.norm1(x_action), shift_msa, scale_msa)
        c_obs_norm = self.obs_norm(c_obs)
        x_attn_out, c_obs_out = self.attn(c_obs_norm, x_norm)
        x_action = x_action + gate_msa.unsqueeze(1) * x_attn_out
        c_obs = c_obs + c_obs_out  # ungated residual (standard Transformer)

        # MLP: updates ONLY action (obs passes through from attention)
        x_norm = modulate(self.norm2(x_action), shift_mlp, scale_mlp)
        x_action = x_action + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x_action, c_obs


# ---------------------------------------------------------------------------
# SATBackbone — full backbone
# ---------------------------------------------------------------------------


class SATBackbone(nn.Module):
    """SAT backbone: structural-centric action Transformer.

    Forward signature matches ``FlowMatch`` protocol:
    ``forward(x, timestep, context) -> (B, Da, T)``

    Actions are processed as ``(B, Da, T)`` internally — each token
    along the Da axis represents one joint's full future trajectory.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_obs_tokens: int,
        obs_token_dim: int,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.1,
        # EJC args
        ejc_num_embodiments: int = 1,
        ejc_num_functions: int | None = None,
        ejc_num_axes: int = 1,
        ejc_embodiment_dim: int = 8,
        ejc_function_dim: int = 32,
        ejc_axis_dim: int = 8,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # ---- Trajectory embedder: T -> 64 -> hidden_dim ----
        self.x_embedder = nn.Sequential(
            nn.Linear(horizon, 64),
            nn.Mish(),
            nn.Linear(64, hidden_dim),
        )

        # ---- Joint identity (EJC) ----
        self.joint_codebook = EmbodiedJointCodebook(
            num_joints=action_dim,
            hidden_dim=hidden_dim,
            num_embodiments=ejc_num_embodiments,
            num_functions=ejc_num_functions,
            num_axes=ejc_num_axes,
            embodiment_dim=ejc_embodiment_dim,
            function_dim=ejc_function_dim,
            axis_dim=ejc_axis_dim,
        )

        # ---- Observation context projection ----
        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)

        # ---- Timestep embedding ----
        self.timestep_embedder = TimestepMLP(
            pos_emb_dim=128,
            output_dim=hidden_dim,
        )

        # ---- Obs pre-norm (AdaLNZero, time-conditioned) ----
        self.obs_pre_norm = _AdaLNZeroObs(dim=hidden_dim, cond_dim=hidden_dim)

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList(
            [
                SATBlock(
                    hidden_size=hidden_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    p_drop_attn=p_drop_attn,
                )
                for _ in range(n_layers)
            ]
        )

        # ---- Final projection: hidden_dim -> T ----
        self.final_layer = _FinalLayer(hidden_dim, horizon)

        self.initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def initialize_weights(self):
        # Linear layers: Xavier uniform
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Embedders: normal
        for layer in self.x_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        if self.context_embedder.bias is not None:
            nn.init.constant_(self.context_embedder.bias, 0)

        # Timestep MLP: normal on Linear layers
        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # AdaLN: zero init all modulation outputs
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Final layer: zero init (AdaLN + linear)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Re-apply AdaLN-Zero init destroyed by _basic_init sweep above
        self.obs_pre_norm.initialize_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        shuffle: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, Da, T)`` — action tokens (Da = number of joints)
            timestep: ``(B,)`` — flow time in [0, 1]
            context: ``(B, N_obs, obs_token_dim)`` — observation tokens
            shuffle: if True, randomly permute the Da axis (joint
                     tokens) together with their EJC identities.
                     Each sample gets its own independent permutation
                     (paper §2.4: "为每个样本生成随机排列π").

        Returns:
            ``(B, Da, T)`` — predicted velocity field per joint
        """
        B, Da, T_in = x.shape
        assert T_in == self.horizon, f"horizon mismatch: {T_in} vs {self.horizon}"

        # 1. Embed per-joint trajectories
        x = self.x_embedder(x)  # (B, Da, T) -> (B, Da, hidden_dim)

        # 2. Joint identity
        ejc = self.joint_codebook()  # (Da, hidden_dim)

        # 3. Per-sample random shuffle (paper §2.4, §6.3)
        perm = None
        if shuffle and self.training:
            perm = torch.stack([torch.randperm(Da, device=x.device) for _ in range(B)], dim=0)
            # Shuffle action tokens: (B, Da, hidden_dim)
            x = torch.gather(x, dim=1, index=perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            # Shuffle EJC synchronously
            ejc = ejc.unsqueeze(0).expand(B, -1, -1)
            ejc = torch.gather(ejc, dim=1, index=perm.unsqueeze(-1).expand(-1, -1, ejc.shape[-1]))

        # 4. Token = trajectory feature + joint identity (ADD, paper §2.3)
        x = x + ejc  # (B, Da, hidden_dim)

        # 5. Embed observation context
        c_obs = self.context_embedder(context)  # (B, N_obs, hidden_dim)

        # 6. Timestep conditioning
        time_c = self.timestep_embedder(timestep)  # (B, hidden_dim)

        # 7. Time-conditioned pre-norm on obs tokens
        c_obs = self.obs_pre_norm(c_obs, time_c)

        # 8. Transformer blocks
        for block in self.blocks:
            x, c_obs = block(x, c_obs, time_c)

        # 9. Final projection with AdaLN: hidden_dim -> horizon (T)
        x = self.final_layer(x, time_c)  # (B, Da, T)

        # 10. Unshuffle if needed (paper §2.4)
        if perm is not None:
            inv_perm = torch.argsort(perm, dim=1)  # (B, Da)
            x = torch.gather(x, dim=1, index=inv_perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        return x

    # ------------------------------------------------------------------
    # Optimizer groups
    # ------------------------------------------------------------------

    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=[],
            extra_blacklist=(RmsNorm, nn.LayerNorm),
        )


# ---------------------------------------------------------------------------
# Internal helpers (pattern-matched from ditx.py)
# ---------------------------------------------------------------------------


class _AdaLNZeroObs(nn.Module):
    """AdaLN-Zero pre-normalisation for observation tokens.

    Zero-initialised so obs tokens start at identity contribution
    and gradually participate in conditioning the action tokens.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cond_linear = nn.Linear(cond_dim, dim * 2)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[: self.dim], 1.0)
        nn.init.zeros_(self.cond_linear.bias[self.dim :])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma, beta = self.cond_linear(cond).chunk(2, dim=-1)
        # Broadcast: (B, hidden_dim) -> (B, 1, hidden_dim)
        return x * gamma.unsqueeze(1) + beta.unsqueeze(1)


class _FinalLayer(nn.Module):
    """Final projection with AdaLN time modulation (DiT standard).

    Time-conditioned shift/scale → RMSNorm → Linear.
    Zero-initialised so the model starts from identity output.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
