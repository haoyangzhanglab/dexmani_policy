from __future__ import annotations

from typing import Literal

import torch
from omegaconf import DictConfig
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import (
    ConditionalUNet1D,
    validate_unet_horizon,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.obs_encoder.rgb.build import build_rgb_encoder
from dexmani_policy.agents.policies.base import BasePolicy


class DPObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "rgb")

    def __init__(
        self,
        rgb_backbone_name: Literal["resnet", "dino", "siglip"],
        state_dim: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        rgb_backbone_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone, self.image_processor = build_rgb_encoder(
            rgb_backbone_name, config=rgb_backbone_config
        )
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.backbone.out_dim + self.state_mlp.out_dim

    @property
    def effective_config(self):
        return {"rgb_backbone_config": self.backbone.effective_config}

    def forward(self, obs: dict) -> torch.Tensor:
        rgb = self.image_processor.process_images(obs["rgb"])

        features = torch.cat(
            [
                self.backbone(rgb)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class DPPolicy(BasePolicy):
    supported_execution_modes = ("sync", "async", "rtc")

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        rgb = dict(config.rgb_backbone_config)
        if config.rgb_backbone_name == "resnet":
            if not {"tune_mode", "norm_mode"} <= rgb.keys():
                raise ValueError("Saved ResNet requires explicit tune_mode and norm_mode")
            rgb["weights"] = None
        if config.rgb_backbone_name in {"dino", "siglip"} and not isinstance(
            rgb.get("architecture"), (dict, DictConfig)
        ):
            raise ValueError("Saved HF backbone requires an architecture mapping")
        rgb["load_pretrained"] = False
        config.rgb_backbone_config = rgb

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        rgb_backbone_name: Literal["resnet", "dino", "siglip"],
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
        validate_unet_horizon(horizon, down_dims)
        obs_encoder = DPObservationEncoder(
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
