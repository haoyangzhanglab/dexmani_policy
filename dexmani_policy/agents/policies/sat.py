from __future__ import annotations

from functools import partial
from typing import Literal

import torch

from dexmani_policy.agents.action_decoders.backbones.sat_action_io import SATActionIO
from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction import HandObjectInteractionEncoder
from dexmani_policy.agents.policies.base import Policy


class SAT(Policy):
    required_identity_fields = frozenset({
        "point_cloud", "camera_intrinsic", "camera_extrinsic", "eef_pose", "fingertip_xyz",
    })
    supports_rtc = True
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        pc_dim: int = 3,
        num_points: int = 1024,
        patch_config: dict | None = None,
        spatial_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_patch_sampling: Literal["center", "group_mean"] = "center",
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 5,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        time_shift_alpha: float = 1.0,
        backbone_type: Literal["ditx", "dit"] = "ditx",
    ) -> None:
        if action_dim != 19 or hidden_dim % 4:
            raise ValueError("SAT requires joint action_dim=19 and hidden_dim divisible by 4")
        if backbone_type not in {"ditx", "dit"}:
            raise ValueError("SAT backbone_type must be 'ditx' or 'dit'")

        obs_encoder = HandObjectInteractionEncoder(
            pc_dim=pc_dim,
            num_points=num_points,
            patch_config=patch_config,
            spatial_config=spatial_config,
            fps_random_config=fps_random_config,
            rgb_backbone_config=rgb_backbone_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_patch_sampling=rgb_patch_sampling,
        )

        backbone_class = DiT if backbone_type == "dit" else partial(DiTX, horizon=horizon, action_dim=action_dim)
        backbone = backbone_class(
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            timestep_embed_dim=timestep_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            action_io=SATActionIO(horizon, hidden_dim),
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

    def _encode_obs(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = obs_dict["point_cloud"].shape[0]
        tokens = super()._encode_obs(obs_dict)
        return tokens.unflatten(0, (batch_size, self.n_obs_steps)).flatten(1, 2)
