from __future__ import annotations

from typing import Literal

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.backbones.factorized_action_io import FactorizedActionIO
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder
from dexmani_policy.agents.policies.base import Policy


class FlowMatch(Policy):
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
        n_layers: int = 6,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 10,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        time_shift_alpha: float = 3.0,
        factorized: bool = False,
    ) -> None:
        if action_dim != 19:
            raise ValueError("FlowMatch requires joint action=7+12")

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

        backbone = DiTX(
            horizon=horizon,
            action_dim=action_dim,
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
            action_io=FactorizedActionIO(horizon, hidden_dim) if factorized else None,
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
