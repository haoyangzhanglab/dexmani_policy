import torch
import torch.nn as nn
from typing import List

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.agents.base_agent import BaseAgent
from dexmani_policy.agents.obs_encoder.dp import DPEncoder
from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion


class DPAgent(BaseAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        # encoder params
        rgb_backbone_name: str,
        state_dim: int,
        # backbone params
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        # expert params
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
    ):
        super().__init__(horizon, n_obs_steps, n_action_steps, action_dim)

        self.obs_encoder = DPEncoder(
            rgb_backbone_name=rgb_backbone_name,
            state_dim=state_dim,
        )

        if condition_type == "film":
            self.obs_cond_dim = self.obs_encoder.out_shape * n_obs_steps
        elif condition_type == "cross_attention_film":
            self.obs_cond_dim = self.obs_encoder.out_shape
        else:
            raise ValueError(f"{condition_type} is not implemented")

        self.backbone = ConditionalUnet1D(
            input_dim=action_dim,
            context_dim=self.obs_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
        )

        self.action_expert = Diffusion(
            model=self.backbone,
            num_training_steps=num_training_steps,
            num_inference_steps=num_inference_steps,
            prediction_type=prediction_type,
        )

    def preprocess(self, obs_dict):
        this_obs = self.normalizer.normalize(obs_dict)
        this_obs = dict_apply(
            this_obs,
            lambda x: x[:, :self.n_obs_steps, ...] if torch.is_tensor(x) else x,
        )
        this_obs = dict_apply(this_obs, lambda x: x.flatten(0, 1) if torch.is_tensor(x) else x)
        return this_obs

    def encode_obs_as_condition(self, obs_dict):
        batch_size = obs_dict["rgb"].shape[0]
        this_obs_dict = self.preprocess(obs_dict)

        feat = self.obs_encoder(this_obs_dict)
        if self.backbone.condition_type == "film":
            feat = feat.reshape(batch_size, -1)
        elif self.backbone.condition_type == "cross_attention_film":
            feat = feat.reshape(batch_size, self.n_obs_steps, -1)
        else:
            raise ValueError(f"{self.backbone.condition_type} is not implemented")

        return feat
