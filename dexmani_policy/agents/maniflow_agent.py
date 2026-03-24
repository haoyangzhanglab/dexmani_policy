import torch
import torch.nn as nn
from typing import List

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.agents.base_agent import BaseAgent
from dexmani_policy.agents.obs_encoder.dp3 import DP3Encoder
from dexmani_policy.agents.action_decoders.backbone.ditx import DiTX_FlowMatch
from dexmani_policy.agents.action_decoders.flowmatch import FlowMatch_With_Consistency


class ManiFlowAgent(BaseAgent):
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
        timestep_embed_dim: int = 128,
        target_t_embed_dim: int = 128,
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        pre_norm_modality: bool = False,
        # expert params
        denoise_timesteps:int = 10,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",      
    ):
        super().__init__(horizon, n_obs_steps, n_action_steps, action_dim)

        self.obs_encoder = DP3Encoder(
            type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            point_wise=True,
            state_dim=state_dim,
        )
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        self.obs_cond_dim = self.obs_encoder.out_shape

        self.backbone = DiTX_FlowMatch(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_seq_len=self.num_points,
            obs_feat_dim=self.obs_cond_dim,
            timestep_embed_dim=timestep_embed_dim,
            target_t_embed_dim=target_t_embed_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            pre_norm_modality=pre_norm_modality
        )

        self.action_expert = FlowMatch_With_Consistency(
            model=self.backbone,
            denoise_timesteps=denoise_timesteps,
            flow_batch_ratio=flow_batch_ratio,
            consistency_batch_ratio=consistency_batch_ratio,
            t_sample_mode_for_flow=t_sample_mode_for_flow,
            t_sample_mode_for_consistency=t_sample_mode_for_consistency,
            dt_sample_mode_for_consistency=dt_sample_mode_for_consistency,
            target_t_sample_mode=target_t_sample_mode
        )


    def encode_obs_as_condition(self, obs_dict):
        B = obs_dict["point_cloud"].shape[0]

        this_obs_dict = self.normalize_and_slice_obs(obs_dict)
        this_obs_dict = self.preprocess_point_cloud(this_obs_dict, self.num_points, self.use_coord_only)
        this_obs_dict = dict_apply(this_obs_dict, lambda x: x.flatten(0, 1) if torch.is_tensor(x) else x)
        
        feat = self.obs_encoder(this_obs_dict)
        feat = feat.reshape(B, -1, self.obs_cond_dim)
        return feat

