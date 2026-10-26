from __future__ import annotations

import inspect

from dexmani_policy.agents.action_decoders.backbones.unet1d import (
    ConditionalUNet1D,
    validate_unet_horizon,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion, validate_action_clipping
from dexmani_policy.agents.obs_encoder.rgb_state import RGBStateObservationEncoder
from dexmani_policy.agents.policies.base import BasePolicy


class DPPolicy(BasePolicy):
    supported_execution_modes = ("sync", "async", "rtc")

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        RGBStateObservationEncoder.prepare_restore_config(config)

    @classmethod
    def validate_model_config(cls, cfg, spec, *, dataset, inference_steps) -> None:
        validate_action_clipping(spec, cfg.get("clip_sample", True))
        defaults = inspect.signature(cls).parameters["down_dims"].default
        validate_unet_horizon(cfg.horizon, cfg.get("down_dims", defaults))

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        rgb_backbone_name: str,
        state_dim: int,
        state_out_dim: int = 64,
        rgb_backbone_config: dict | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        modality_dropout_probs: dict[str, float] | None = None,
        cond_predict_scale: bool = True,
        clip_sample: bool = True,
    ) -> None:
        obs_encoder = RGBStateObservationEncoder(
            rgb_backbone_name,
            state_dim,
            n_obs_steps,
            state_out_dim,
            rgb_backbone_config=rgb_backbone_config,
        )
        backbone = ConditionalUNet1D(
            input_dim=action_dim,
            context_dim=obs_encoder.out_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        action_decoder = Diffusion(
            backbone,
            num_training_steps,
            num_inference_steps,
            prediction_type,
            clip_sample=clip_sample,
        )
        super().__init__(
            obs_encoder,
            action_decoder,
            horizon,
            n_obs_steps,
            n_action_steps,
            action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )
