from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction import HandObjectInteractionEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import Policy


class InterFlowEncoder(nn.Module):
    def __init__(
        self,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        n_obs_steps: int = 2,
        patch_config: dict | None = None,
        spatial_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dinov3",
        rgb_backbone_config: dict | None = None,
        rgb_patch_sampling: Literal["center", "group_mean"] = "center",
        *,
        geometry_scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.out_dim = 192
        self.interaction_encoder = HandObjectInteractionEncoder(
            pc_dim=pc_dim,
            num_points=num_points,
            patch_config=patch_config,
            spatial_config=spatial_config,
            fps_random_config=fps_random_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_backbone_config=rgb_backbone_config,
            rgb_patch_sampling=rgb_patch_sampling,
            geometry_scales=geometry_scales,
        )
        self.state_encoder = StateEncoder(state_dim, self.out_dim)
        self.token_type_embedding = nn.Parameter(torch.empty(1, 2, self.out_dim))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        self.token_norm = nn.LayerNorm(self.out_dim)

    @property
    def rgb_encoder(self) -> nn.Module:
        return self.interaction_encoder.rgb_encoder

    @property
    def effective_config(self) -> dict:
        return self.interaction_encoder.effective_config

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        interaction = self.interaction_encoder(obs)
        state = self.state_encoder(obs["joint_state"]).unsqueeze(1)
        interaction = interaction + self.token_type_embedding[:, :1].to(interaction.dtype)
        state = state + self.token_type_embedding[:, 1:].to(state.dtype)
        frames = self.token_norm(torch.cat((interaction, state), dim=1))
        return frames.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)


class InterFlow(Policy):
    supported_action_keys = ("action",)
    required_identity_fields = frozenset({
        "point_cloud", "camera_intrinsic", "camera_extrinsic", "eef_pose", "fingertip_xyz",
    })

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        patch_config: dict | None = None,
        spatial_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dinov3",
        rgb_backbone_config: dict | None = None,
        rgb_patch_sampling: Literal["center", "group_mean"] = "center",
        n_layers: int = 6,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 8,
        time_shift_alpha: float = 1.0,
        *,
        geometry_scales: dict[str, float] | None = None,
    ) -> None:
        obs_encoder = InterFlowEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            patch_config=patch_config,
            spatial_config=spatial_config,
            fps_random_config=fps_random_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_backbone_config=rgb_backbone_config,
            rgb_patch_sampling=rgb_patch_sampling,
            geometry_scales=geometry_scales,
        )
        backbone = DiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            timestep_embed_dim=timestep_embed_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
            time_shift_alpha=time_shift_alpha,
        )

        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
        )
