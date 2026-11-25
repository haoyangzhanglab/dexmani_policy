from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.ditx import InteractionDiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction import HandObjectInteractionEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import Policy


class InteractionFlowEncoder(nn.Module):
    def __init__(
        self, interaction_encoder: HandObjectInteractionEncoder, state_dim: int, n_obs_steps: int,
    ) -> None:
        super().__init__()
        self.interaction_encoder = interaction_encoder
        self.state_encoder = StateEncoder(state_dim, interaction_encoder.out_dim)
        self.n_obs_steps = n_obs_steps

    @property
    def rgb_encoder(self):
        return self.interaction_encoder.rgb_encoder

    @property
    def effective_config(self) -> dict:
        return self.interaction_encoder.effective_config

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.interaction_encoder.forward_with_aux(obs)
        state = self.state_encoder(obs["joint_state"]).unsqueeze(1)
        frames = torch.cat((encoded["scene_tokens"], state, encoded["tokens"]), dim=1)
        return frames.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)


class InteractionFlow(Policy):
    required_identity_fields = frozenset({
        "point_cloud", "camera_intrinsic", "camera_extrinsic", "eef_pose", "fingertip_xyz",
    })

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        patch_config: dict | None = None,
        spatial_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_backbone_config: dict | None = None,
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 5,
        time_shift_alpha: float = 1.0,
    ) -> None:
        interaction_encoder = HandObjectInteractionEncoder(
            pc_dim=pc_dim, num_points=num_points, patch_config=patch_config,
            spatial_config=spatial_config, fps_random_config=fps_random_config,
            rgb_backbone_name=rgb_backbone_name, rgb_backbone_config=rgb_backbone_config,
            rgb_semantic_sampling=rgb_semantic_sampling,
        )
        obs_encoder = InteractionFlowEncoder(interaction_encoder, state_dim, n_obs_steps)
        backbone = InteractionDiTX(
            horizon=horizon, action_dim=action_dim, n_obs_steps=n_obs_steps,
            obs_token_dim=interaction_encoder.out_dim,
            num_scene_tokens=interaction_encoder.patch_encoder.num_patches,
            n_layers=n_layers, hidden_dim=hidden_dim, n_head=n_head, mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn, qkv_bias=qkv_bias, qk_norm=qk_norm,
            timestep_embed_dim=timestep_embed_dim,
        )
        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=RectifiedFlow(backbone, num_inference_steps, time_shift_alpha),
            horizon=horizon, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
            action_dim=action_dim,
        )
