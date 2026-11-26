from __future__ import annotations

from functools import partial
from typing import Literal

import torch

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.backbones.sat_action_io import SATActionIO
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder
from dexmani_policy.agents.policies.base import Policy


class SAT(Policy):
    required_identity_fields = frozenset({"point_cloud", "camera_intrinsic", "camera_extrinsic"})
    supports_rtc = True
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 10,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        time_shift_alpha: float = 1.0,
        backbone_type: Literal["ditx", "dit"] = "ditx",
    ) -> None:
        if action_dim != 19 or hidden_dim % 4:
            raise ValueError("SAT requires joint action_dim=19 and hidden_dim divisible by 4")
        if backbone_type not in {"ditx", "dit"}:
            raise ValueError("SAT backbone_type must be 'ditx' or 'dit'")

        obs_encoder = RGBPointObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            rgb_backbone_config=rgb_backbone_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_semantic_sampling=rgb_semantic_sampling,
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
        obs_steps = obs_dict["point_cloud"].shape[1]
        if obs_steps != self.n_obs_steps:
            raise ValueError(f"SAT requires {self.n_obs_steps} observation frames, got {obs_steps}")
        return super()._encode_obs(obs_dict)
