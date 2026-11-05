from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import (
    ConditionalUNet1D,
    validate_unet_horizon,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.interaction.correspondence import (
    ContactCoordinationEncoder,
)
from dexmani_policy.agents.policies.base import Policy, module_parameter_group


class CordViPEncoder(nn.Module):

    def __init__(self, n_obs_steps):
        super().__init__()
        self.interaction_encoder = ContactCoordinationEncoder()
        self.n_obs_steps = n_obs_steps

    @property
    def out_dim(self) -> int:
        return self.n_obs_steps * self.interaction_encoder.out_dim

    def forward(self, obs):
        sequence = {
            key: value.unflatten(0, (-1, self.n_obs_steps))
            for key, value in obs.items()
        }
        return torch.cat(self.interaction_encoder(sequence), dim=-1).flatten(1)


class CordViP(Policy):
    supports_rtc = True
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 3,
        n_action_steps: int = 8,
        action_dim: int = 19,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        diffusion_step_embed_dim: int = 64,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        clip_sample: bool = True,
    ) -> None:
        if action_dim != 19:
            raise ValueError("CordViP uses XYZ and arm7+hand12 state/actions")
        validate_unet_horizon(horizon, down_dims)
        obs_encoder = CordViPEncoder(n_obs_steps)
        backbone = ConditionalUNet1D(
            action_dim, obs_encoder.out_dim, diffusion_step_embed_dim,
            down_dims, kernel_size, n_groups, cond_predict_scale,
        )
        decoder = Diffusion(backbone, num_training_steps, num_inference_steps, prediction_type, clip_sample)
        super().__init__(obs_encoder, decoder, horizon, n_obs_steps, n_action_steps, action_dim)

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay):
        return [
            module_parameter_group(self.action_decoder, lr=lr, weight_decay=weight_decay),
            module_parameter_group(self.obs_encoder, lr=obs_lr, weight_decay=obs_weight_decay),
        ]
