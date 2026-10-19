from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.core.base_policy import UNetDiffusionPolicy
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.obs_encoder.rgb.registry import build_rgb_encoder
from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNetEncoder


class RGBObservationEncoder(nn.Module):
    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return ("joint_state", "rgb")

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

        # Use channels-last memory format for CNN backbones; shape stays NCHW.
        if isinstance(self.backbone, ResNetEncoder):
            rgb = rgb.to(memory_format=torch.channels_last)

        feat = torch.cat(
            [
                self.backbone(rgb)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        return feat.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class DiffusionPolicy(UNetDiffusionPolicy):
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
        **kwargs,
    ) -> None:
        obs_encoder = RGBObservationEncoder(
            rgb_backbone_name,
            state_dim,
            n_obs_steps,
            state_out_dim,
            rgb_backbone_config=rgb_backbone_config,
        )
        super().__init__(obs_encoder, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs)
