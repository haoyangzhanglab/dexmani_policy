from __future__ import annotations

from typing import Literal

import torch

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior
from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
from dexmani_policy.agents.policies.base import Policy


class DexLatent(Policy):
    supported_action_keys = ("action",)
    hand_dim = 12
    flow_action_dim = 39

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        pc_dim: int = 3,
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
            raise ValueError("DexLatent requires two observations and 19D joint actions/state")
        obs_encoder = SpatialObservationEncoder(
            pc_dim=pc_dim,
            state_dim=19,
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
        self.hand_prior = DexLatentHandPrior()

    @property
    def effective_config(self) -> dict:
        return {**super().effective_config, "time_shift_alpha": self.action_decoder.time_shift_alpha}

    @torch.no_grad()
    def _encode_actions(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalizer["action"].normalize(actions)
        return normalized[..., :7], self.hand_prior.encode(normalized[..., 7:])

    def fit_latent_normalizer(self, dataset) -> None:
        latents = (self._encode_actions(chunk)[1] for chunk in dataset.iter_normalization_data("action"))
        self.normalizer.fit_field("hand_latent", latents, mode="limits")

    def compute_loss(self, batch: dict, *, log_metrics: bool = True, **kwargs) -> tuple[torch.Tensor, dict]:
        condition = self._encode_obs(batch["obs"])
        arm, latent = self._encode_actions(batch["action"])
        flow_target = torch.cat([arm, self.normalizer["hand_latent"].normalize(latent)], dim=-1)
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
            raise NotImplementedError("DexLatent does not support RTC")
        prediction = self.action_decoder.predict_action(
            condition, (condition.shape[0], self.horizon, self.flow_action_dim),
            inference_steps=inference_steps, generator=generator,
        )
        latent = self.normalizer["hand_latent"].unnormalize(prediction[..., 7:])
        hand = self.hand_prior.decode(latent)
        prediction = self.normalizer["action"].unnormalize(torch.cat([prediction[..., :7], hand], dim=-1))
        return self._action_output(prediction)
