"""Structural Action Transformer: each [B, Da, T] token is one joint trajectory."""

import torch
from torch import nn
from timm.models.vision_transformer import Mlp, RmsNorm

from .modules import (
    AdaLNOutput,
    SelfAttention,
    TimestepMLP,
    WEIGHT_INIT_STD,
    approximate_gelu,
    batch_timesteps,
    initialize_linear,
    modulate,
)


class EmbodiedJointCodebook(nn.Module):
    """Sum projected embodiment, function and rotation-axis embeddings."""

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
        if num_functions < num_joints:
            raise ValueError(
                "Default joint_function=arange(num_joints) requires num_functions >= num_joints"
            )

        self.hidden_dim = hidden_dim

        self.emb_emb = nn.Embedding(num_embodiments, embodiment_dim)
        self.func_emb = nn.Embedding(num_functions, function_dim)
        self.axis_emb = nn.Embedding(num_axes, axis_dim)

        self.proj_emb = nn.Linear(embodiment_dim, hidden_dim)
        self.proj_func = nn.Linear(function_dim, hidden_dim)
        self.proj_axis = nn.Linear(axis_dim, hidden_dim)

        # The local default identifies joints; it does not assign anatomical roles or axes.
        self.register_buffer("joint_embodiment", torch.zeros(num_joints, dtype=torch.long))
        self.register_buffer("joint_function", torch.arange(num_joints, dtype=torch.long))
        self.register_buffer("joint_axis", torch.zeros(num_joints, dtype=torch.long))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name, size in (
            ("joint_embodiment", self.emb_emb.num_embeddings),
            ("joint_function", self.func_emb.num_embeddings),
            ("joint_axis", self.axis_emb.num_embeddings),
        ):
            value = state_dict.get(prefix + name)
            if value is not None and (
                value.dtype
                not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or bool(((value < 0) | (value >= size)).any())
            ):
                raise ValueError(
                    f"Invalid EJC mapping {prefix + name}: expected integer joint indices in [0, {size})"
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self) -> torch.Tensor:
        emb = self.proj_emb(self.emb_emb(self.joint_embodiment))
        func = self.proj_func(self.func_emb(self.joint_function))
        axis = self.proj_axis(self.axis_emb(self.joint_axis))
        return emb + func + axis


class SATBlock(nn.Module):
    """Attention updates both streams; the gated MLP updates only actions."""

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

        self.attn = SelfAttention(
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
            act_layer=approximate_gelu,
            drop=0.0,
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.obs_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x_action: torch.Tensor,
        c_obs: torch.Tensor,
        time_c: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(
            time_c
        ).chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x_action), shift_msa, scale_msa)
        c_obs_norm = self.obs_norm(c_obs)
        combined = self.attn(torch.cat([c_obs_norm, x_norm], dim=1), attn_mask=attn_mask)
        c_obs_out, x_attn_out = combined.split([c_obs.shape[1], x_action.shape[1]], dim=1)
        x_action = x_action + gate_msa.unsqueeze(1) * x_attn_out
        c_obs = c_obs + c_obs_out

        x_norm = modulate(self.norm2(x_action), shift_mlp, scale_mlp)
        x_action = x_action + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x_action, c_obs


class SAT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        obs_token_dim: int,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.1,
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

        self.x_embedder = nn.Sequential(
            nn.Linear(horizon, 64),
            nn.Mish(),
            nn.Linear(64, hidden_dim),
        )

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

        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)

        self.timestep_embedder = TimestepMLP(
            pos_emb_dim=128,
            output_dim=hidden_dim,
        )

        self.obs_pre_norm = ObservationAdaLN(dim=hidden_dim, cond_dim=hidden_dim)

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

        self.final_layer = AdaLNOutput(hidden_dim, horizon, RmsNorm(hidden_dim, eps=1e-6))

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(initialize_linear)

        for layer in self.x_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        if self.context_embedder.bias is not None:
            nn.init.zeros_(self.context_embedder.bias)

        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        self.obs_pre_norm.initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        num_joints = x.shape[1]
        x = self.x_embedder(x) + self.joint_codebook()
        c_obs = self.context_embedder(context)
        time_c = self.timestep_embedder(batch_timesteps(timestep, x))
        c_obs = self.obs_pre_norm(c_obs, time_c)

        # Observations attend only to observations; actions attend to both streams.
        num_obs = c_obs.shape[1]
        attn_mask = x.new_zeros(num_obs + num_joints, num_obs + num_joints)
        attn_mask[:num_obs, num_obs:] = float("-inf")
        for block in self.blocks:
            x, c_obs = block(x, c_obs, time_c, attn_mask)

        return self.final_layer(x, time_c)


class ObservationAdaLN(nn.Module):
    """SAT uses direct conditioning with identity affine initialization, without SiLU."""

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

        return x * gamma.unsqueeze(1) + beta.unsqueeze(1)
