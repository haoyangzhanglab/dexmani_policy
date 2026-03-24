import torch
import torch.nn as nn
from typing import List

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.agents.base_agent import BaseAgent
from dexmani_policy.agents.obs_encoder.dp3 import DP3Encoder
from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion


class DP3Agent(BaseAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        # encoder params
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        state_dim: int,
        num_points: int,
        # backbone params
        diffusion_step_embed_dim:int = 256,
        down_dims:List[int] = [256,512,1024],
        kernel_size: int =5,
        n_groups: int = 8,
        condition_type='film',
        # expert params
        num_training_steps = 100,
        num_inference_steps = 10,
        prediction_type = "sample",       
    ):
        super().__init__(horizon, n_obs_steps, n_action_steps, action_dim)

        self.obs_encoder = DP3Encoder(
            type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            point_wise=False,
            state_dim=state_dim,
        )
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)

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
            use_up_condition=True
        )

        self.action_expert = Diffusion(
            model = self.backbone,
            num_training_steps=num_training_steps,
            num_inference_steps=num_inference_steps,
            prediction_type=prediction_type
        )


    def encode_obs_as_condition(self, obs_dict):
        B = obs_dict["point_cloud"].shape[0]

        this_obs_dict = self.normalize_and_slice_obs(obs_dict)
        this_obs_dict = self.preprocess_point_cloud(this_obs_dict, self.num_points, self.use_coord_only)
        this_obs_dict = dict_apply(this_obs_dict, lambda x: x.flatten(0, 1) if torch.is_tensor(x) else x)
        
        feat = self.obs_encoder(this_obs_dict)
        if self.backbone.condition_type == "film":
            feat = feat.reshape(B, -1)
        elif self.backbone.condition_type == "cross_attention_film":
            feat = feat.reshape(B, self.n_obs_steps, -1)
        else:
            raise ValueError(f"{self.backbone.condition_type} is not implemented") 

        return feat

