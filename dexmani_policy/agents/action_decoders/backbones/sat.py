"""Structural Action Transformer: [B,T,Da] I/O, internal joint-trajectory tokens."""

import torch
from torch import nn
from timm.models.vision_transformer import Mlp, RmsNorm

from dexmani_policy.common.config import positive_int

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


class XHandJointEmbedding(nn.Module):
    """Canonical xArm7/XHand control order; axes are in local URDF frames."""

    # Group: arm/thumb/index/middle/ring/pinky; role: arm/bend/joint1/joint2.
    # Axis: +Z/+Y/+X/-X/-Z in local URDF frames, not world directions.
    metadata = {
        "joint_ids": tuple(range(19)),
        "group_ids": (0,) * 7 + (1,) * 3 + (2,) * 3 + (3,) * 2 + (4,) * 2 + (5,) * 2,
        "role_ids": (0,) * 7 + (1, 2, 3) + (1, 2, 3) + (2, 3) * 3,
        "axis_ids": (0,) * 7 + (4, 2, 2) + (3, 1, 1) + (1, 1) * 3,
    }

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.joint = nn.Embedding(19, hidden_dim)
        self.group = nn.Embedding(6, hidden_dim)
        self.role = nn.Embedding(4, hidden_dim)
        self.axis = nn.Embedding(5, hidden_dim)
        for name, values in self.metadata.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.long))
        self.initialize_weights()

    def initialize_weights(self):
        for embedding in (self.joint, self.group, self.role, self.axis):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.005)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name, values in self.metadata.items():
            value = state_dict.get(prefix + name)
            if value is not None and (
                value.dtype != torch.long
                or not torch.equal(value.cpu(), torch.tensor(values, dtype=torch.long))
            ):
                error_msgs.append(f"{prefix + name} must match canonical xArm7/XHand metadata")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self) -> torch.Tensor:
        return (
            self.joint(self.joint_ids)
            + self.group(self.group_ids)
            + self.role(self.role_ids)
            + self.axis(self.axis_ids)
        )


class SATBlock(nn.Module):
    """Read a fixed observation token and update only action trajectories."""

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
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(
            time_c
        ).chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x_action), shift_msa, scale_msa)
        c_obs_norm = self.obs_norm(c_obs)
        combined = self.attn(torch.cat([c_obs_norm, x_norm], dim=1), attn_mask=attn_mask)
        x_attn_out = combined[:, c_obs.shape[1] :]
        x_action = x_action + gate_msa.unsqueeze(1) * x_attn_out

        x_norm = modulate(self.norm2(x_action), shift_mlp, scale_mlp)
        x_action = x_action + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x_action


class SAT(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 8,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.0,
    ):
        super().__init__()

        if (horizon, action_dim, hidden_dim) != (16, 19, 256):
            raise ValueError("SAT requires horizon=16, action_dim=19, hidden_dim=256")

        positive_int(n_layers, "n_layers")

        self.x_embedder = nn.Sequential(
            nn.Linear(horizon, 64),
            nn.Mish(),
            nn.Linear(64, hidden_dim),
        )

        self.joint_embedding = XHandJointEmbedding(hidden_dim)

        self.timestep_embedder = TimestepMLP(
            pos_emb_dim=128,
            output_dim=hidden_dim,
        )

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

        for layer in (*self.x_embedder, *self.timestep_embedder.net):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)

        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (16, 19):
            raise ValueError("SAT action must be [B,16,19]")
        if context.shape != (x.shape[0], 256):
            raise ValueError("SAT context must be [B,256]")
        x = self.x_embedder(x.transpose(1, 2)) + self.joint_embedding()
        context = context.unsqueeze(1)
        time_c = self.timestep_embedder(batch_timesteps(timestep, x))

        # Observations attend only to observations; actions attend to both streams.
        attn_mask = x.new_zeros(20, 20)
        attn_mask[0, 1:] = float("-inf")
        for block in self.blocks:
            x = block(x, context, time_c, attn_mask)

        return self.final_layer(x, time_c).transpose(1, 2)
