from __future__ import annotations

from typing import Literal

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.action_io import SATActionIO
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder
from dexmani_policy.agents.policies.base import Policy


class SAT(Policy):
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
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dinov3",
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
        obs_encoder = RGBPointObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            rgb_backbone_config=rgb_backbone_config,
            rgb_backbone_name=rgb_backbone_name,
        )
        backbone_cls = DiT if backbone_type == "dit" else DiTX
        backbone = backbone_cls(
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
