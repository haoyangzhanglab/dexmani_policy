from __future__ import annotations

import inspect

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.obs_encoder.rgb.build import build_rgb_encoder
from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNetEncoder
from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import validate_action_clipping


class DPObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "rgb")

    def __init__(
        self,
        rgb_backbone_name: str,
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

    def forward(self, obs: dict) -> torch.Tensor:
        rgb = self.image_processor.process_images(obs["rgb"])

        if isinstance(self.backbone, ResNetEncoder):
            rgb = rgb.to(memory_format=torch.channels_last)

        features = torch.cat(
            [
                self.backbone(rgb)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class DPPolicy(BasePolicy):
    @classmethod
    def validate_model_config(cls, cfg, spec) -> None:
        validate_action_clipping(spec, cfg.agent.get("clip_sample", True))
        defaults = inspect.signature(cls).parameters["down_dims"].default
        down_dims = cfg.agent.get("down_dims", defaults)
        if not down_dims:
            raise ValueError("UNet down_dims must be nonempty")
        divisor = 2 ** (len(down_dims) - 1)
        if cfg.horizon % divisor:
            raise ValueError(f"UNet horizon must be divisible by {divisor}, got {cfg.horizon}")

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
            down_dims=list(down_dims),
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
