import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D, validate_unet_horizon
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.interaction.correspondence import (
    ContactCoordinationEncoder,
)
from dexmani_policy.agents.policies.base import BasePolicy, module_parameter_group


class CordViPObservationEncoder(nn.Module):
    consumed_observation_fields = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, n_obs_steps):
        super().__init__()
        self.encoder = ContactCoordinationEncoder()
        self.n_obs_steps = n_obs_steps

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def forward(self, obs):
        sequence = {
            key: value.unflatten(0, (-1, self.n_obs_steps))
            for key, value in obs.items()
        }
        return torch.cat(self.encoder(sequence), dim=-1).flatten(1)


class CordViPPolicy(BasePolicy):
    supported_action_keys = ("action",)

    def __init__(self, horizon=12, n_obs_steps=4, n_action_steps=6, action_dim=19,
                 state_dim=19, pc_dim=3,
                 down_dims=(512, 1024, 2048), diffusion_step_embed_dim=64,
                 kernel_size=5, n_groups=8, cond_predict_scale=True,
                 num_training_steps=100, num_inference_steps=10,
                 prediction_type="sample", clip_sample=True):
        if (action_dim, state_dim, pc_dim) != (19, 19, 3):
            raise ValueError("CordViP uses XYZ and arm7+hand12 state/actions")
        validate_unet_horizon(horizon, down_dims)
        obs_encoder = CordViPObservationEncoder(n_obs_steps)
        backbone = ConditionalUNet1D(
            action_dim, n_obs_steps * obs_encoder.out_dim, diffusion_step_embed_dim,
            down_dims, kernel_size, n_groups, cond_predict_scale,
        )
        decoder = Diffusion(backbone, num_training_steps, num_inference_steps, prediction_type, clip_sample)
        super().__init__(obs_encoder, decoder, horizon, n_obs_steps, n_action_steps, action_dim)

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay):
        return [
            module_parameter_group(self.action_decoder, lr=lr, weight_decay=weight_decay),
            module_parameter_group(self.obs_encoder, lr=obs_lr, weight_decay=obs_weight_decay),
        ]
