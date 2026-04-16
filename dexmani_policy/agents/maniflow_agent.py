import inspect
import torch
import torch.nn as nn
from typing import Dict, Optional

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.agents.base_agent import BaseAgent
from dexmani_policy.agents.obs_encoder.pointcloud.registry import (
    build_pc_global_encoder,
    build_pc_patch_tokenizer,
)
from dexmani_policy.agents.common.mlp import create_mlp
from dexmani_policy.agents.action_decoders.backbone.ditx import DiTX_FlowMatch
from dexmani_policy.agents.action_decoders.flowmatch import FlowMatch_With_Consistency
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import farthest_point_sample


class ManiFlowAgent(BaseAgent):
    STATE_OUT_DIM = 64

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        # encoder params
        encoder_type: str,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        pc_encoder_config: Optional[Dict] = None,
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
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",
    ):
        super().__init__(horizon, n_obs_steps, n_action_steps, action_dim)

        if encoder_type == "idp3":
            self._pc_encoder, self._pc_seq_len, self._pc_out_dim = build_pc_global_encoder(encoder_type, pc_dim, pc_encoder_config)
        else:
            self._pc_encoder, self._pc_seq_len, self._pc_out_dim = build_pc_patch_tokenizer(encoder_type, pc_dim, pc_encoder_config)

        self.state_mlp = create_mlp(state_dim, [64, self.STATE_OUT_DIM])

        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        # idp3 需要外部 FPS；pointpn/tokenizer 编码器内部处理，跳过
        self.use_external_fps = (encoder_type == "idp3")

        self.obs_cond_dim = self._pc_out_dim + self.STATE_OUT_DIM
        obs_seq_len = self._pc_seq_len

        self.backbone = DiTX_FlowMatch(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_seq_len=obs_seq_len,
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
            pre_norm_modality=pre_norm_modality,
        )

        self.action_expert = FlowMatch_With_Consistency(
            model=self.backbone,
            denoise_timesteps=denoise_timesteps,
            flow_batch_ratio=flow_batch_ratio,
            consistency_batch_ratio=consistency_batch_ratio,
            t_sample_mode_for_flow=t_sample_mode_for_flow,
            t_sample_mode_for_consistency=t_sample_mode_for_consistency,
            dt_sample_mode_for_consistency=dt_sample_mode_for_consistency,
            target_t_sample_mode=target_t_sample_mode,
        )

    def preprocess(self, obs_dict):
        this_obs = self.normalizer.normalize(obs_dict)
        this_obs = dict_apply(
            this_obs,
            lambda x: x[:, :self.n_obs_steps, ...] if torch.is_tensor(x) else x,
        )

        if self.use_coord_only:
            this_obs["point_cloud"] = this_obs["point_cloud"][..., :3]

        if self.use_external_fps:
            current_num_points = this_obs["point_cloud"].shape[2]
            if current_num_points < self.num_points:
                raise ValueError(
                    f"Point cloud shape wrong, required {self.num_points} points, got {current_num_points} points"
                )
            if current_num_points > self.num_points:
                downsample_point_cloud, _ = farthest_point_sample(
                    this_obs["point_cloud"].flatten(0, 1),
                    num_samples=self.num_points,
                )
                batch_size, time_steps = this_obs["point_cloud"].shape[:2]
                this_obs["point_cloud"] = downsample_point_cloud.reshape(
                    batch_size, time_steps, self.num_points, -1
                )

        this_obs = dict_apply(this_obs, lambda x: x.flatten(0, 1) if torch.is_tensor(x) else x)
        return this_obs

    def compute_loss(self, batch, **kwargs):
        ema_model = kwargs.get("ema_model")
        cond = self.encode_obs_as_condition(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        loss, loss_dict = self.action_expert.compute_loss(cond, nactions, ema_model=ema_model.backbone)

        return loss, loss_dict

    def encode_obs_as_condition(self, obs_dict):
        B = obs_dict["point_cloud"].shape[0]
        this_obs_dict = self.preprocess(obs_dict)

        pc = this_obs_dict["point_cloud"]
        result = self._pc_encoder(pc)

        if isinstance(result, dict):
            global_token = result["global_token"].unsqueeze(1)
            pc_feat = global_token
        elif len(result) == 3:
            patch_token, patch_center, global_token = result
            pc_feat = torch.cat([global_token, patch_token], dim=1)
        else:
            patch_token, patch_center = result
            fn = self._pc_encoder.get_global_token
            sig = inspect.signature(fn)
            if len(sig.parameters) == 2:
                global_token = fn(patch_token, patch_center)
            else:
                global_token = fn(patch_token)
            if global_token.ndim == 2:
                global_token = global_token.unsqueeze(1)
            pc_feat = torch.cat([global_token, patch_token], dim=1)

        state = this_obs_dict["joint_state"]
        state_feat = self.state_mlp(state).unsqueeze(1).expand(-1, pc_feat.size(1), -1)
        feat = torch.cat([pc_feat, state_feat], dim=-1)

        feat = feat.reshape(B, -1, self.obs_cond_dim)
        return feat
