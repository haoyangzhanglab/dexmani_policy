import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D, validate_unet_horizon
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet
from dexmani_policy.agents.obs_encoder.pretrained import AFRO_ENCODER_CONFIG, load_observation_encoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy


class AFROObservationEncoder(nn.Module):
    consumed_observation_fields = ("point_cloud", "joint_state")

    def __init__(self, n_obs_steps):
        super().__init__()
        self.visual = PointNet(3, 64, pooling="max")
        self.state = StateEncoder(19, 64, (64,))
        self.n_obs_steps = n_obs_steps

    def forward(self, obs):
        features = torch.cat((self.visual(obs["point_cloud"])["global_token"], self.state(obs["joint_state"])), -1)
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class AFROPolicy(BasePolicy):
    supported_action_keys = ("action",)

    def __init__(self, horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19,
                 state_dim=19, pc_dim=3, pretrained_path=None, freeze_visual=True,
                 down_dims=(512, 1024, 2048), diffusion_step_embed_dim=128,
                 kernel_size=5, n_groups=8, cond_predict_scale=True,
                 num_training_steps=100, num_inference_steps=10,
                 prediction_type="sample", clip_sample=True):
        if (action_dim, state_dim, pc_dim) != (19, 19, 3):
            raise ValueError("AFRO uses XYZ and arm7+hand12 state/actions")
        validate_unet_horizon(horizon, down_dims)
        obs_encoder = AFROObservationEncoder(n_obs_steps)
        obs_encoder.visual.requires_grad_(not freeze_visual)
        backbone = ConditionalUNet1D(
            action_dim, n_obs_steps * 128, diffusion_step_embed_dim,
            down_dims, kernel_size, n_groups, cond_predict_scale,
        )
        decoder = Diffusion(backbone, num_training_steps, num_inference_steps, prediction_type, clip_sample)
        super().__init__(obs_encoder, decoder, horizon, n_obs_steps, n_action_steps, action_dim)
        self.pretrained_path, self.freeze_visual = pretrained_path, freeze_visual

    @classmethod
    def prepare_restore_config(cls, config):
        config.pretrained_path = None

    def initialize_training(self, data_description):
        if data_description["point_counts"]["point_cloud"] != 1024:
            raise ValueError("AFRO requires 1024 scene points")
        if self.pretrained_path is None:
            if self.freeze_visual:
                raise ValueError("AFRO frozen visual requires pretrained_path; scratch requires freeze_visual=false")
            print("[AFRO] Scratch training with trainable visual encoder")
            return
        state = load_observation_encoder(
            self.pretrained_path, method="afro", encoder_config=AFRO_ENCODER_CONFIG,
            data_description=data_description,
        )
        self.obs_encoder.visual.load_state_dict(state, strict=True)

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay):
        return [
            {"params": [p for p in self.action_decoder.parameters() if p.requires_grad],
             "lr": lr, "weight_decay": weight_decay},
            {"params": [p for p in self.obs_encoder.parameters() if p.requires_grad],
             "lr": obs_lr, "weight_decay": obs_weight_decay},
        ]
