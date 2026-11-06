from __future__ import annotations

from typing import Any, Literal

import torch

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.codebook import (
    HandStateCodebook, validate_hand_coordinates,
)
from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
from dexmani_policy.agents.policies.base import Policy


class DQRISE(Policy):
    supported_action_keys = ("action",)
    arm_dim = 7
    hand_dim = 12
    flow_action_dim = 8

    def __init__(
        self,
        *,
        num_codes: int,
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
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.0,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 5,
        time_shift_alpha: float = 1.0,
    ) -> None:
        if (n_obs_steps, action_dim) != (2, 19):
            raise ValueError("DQ-RISE requires n_obs_steps=2 and joint action=7+12")
        obs_encoder = SpatialObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            rgb_backbone_config=rgb_backbone_config,
            rgb_backbone_name=rgb_backbone_name,
        )
        backbone = DiTX(
            horizon=horizon,
            action_dim=self.flow_action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            timestep_embed_dim=timestep_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            p_drop_attn=p_drop_attn,
        )
        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
            t_sample_mode="uniform",
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
        self.hand_codebook = HandStateCodebook(
            hand_dim=self.hand_dim,
            num_codes=num_codes,
        )

    @property
    def effective_config(self) -> dict:
        return {
            **super().effective_config,
            "num_codes": self.hand_codebook.num_codes,
            "time_shift_alpha": self.action_decoder.time_shift_alpha,
        }

    def initialize_training(self, data_description, codebook) -> None:
        validate_hand_coordinates(data_description["hand_coordinates"])
        if data_description["action_key"] != "action":
            raise ValueError("DQ-RISE requires joint policy data with action_key=action")
        self.hand_codebook = codebook

    def compute_loss(
        self, batch: dict[str, Any], *, log_metrics: bool = True, **kwargs: Any,
    ) -> tuple[torch.Tensor, dict]:
        condition = self._encode_obs(batch["obs"])

        normed = self.normalizer["action"].normalize(batch["action"])
        arm_part = normed[..., : self.arm_dim]
        hand_part = normed[..., self.arm_dim :]
        index = self.hand_codebook.encode(hand_part)

        flow_target = torch.cat([arm_part, index], dim=-1)
        return self.action_decoder.compute_loss(
            condition, flow_target, log_metrics=log_metrics, **kwargs,
        )

    @torch.no_grad()
    def _predict_from_condition(
        self, condition: torch.Tensor, inference_steps: int | None = None,
        *, generator: torch.Generator | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if rtc_prefix is not None or delay_steps or rtc_guidance_cap:
            raise NotImplementedError("DQ-RISE does not support RTC")
        reduced_action = self.action_decoder.predict_action(
            condition, (condition.shape[0], self.horizon, self.flow_action_dim),
            inference_steps=inference_steps, generator=generator,
        )

        arm_pred = reduced_action[..., : self.arm_dim]
        idx_pred = reduced_action[..., -1]
        hand_pred = self.hand_codebook.decode(idx_pred)

        normalized_full_action = torch.cat([arm_pred, hand_pred], dim=-1)
        prediction = self.normalizer["action"].unnormalize(normalized_full_action)
        return self._action_output(prediction)

    def extra_repr(self) -> str:
        return (
            f"action_dim={self.action_dim}, arm_dim={self.arm_dim}, "
            f"hand_dim={self.hand_dim}, "
            f"flow_action_dim={self.flow_action_dim}, "
            f"codes={self.hand_codebook.num_codes}"
        )
