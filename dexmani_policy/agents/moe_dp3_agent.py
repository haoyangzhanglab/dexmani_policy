import torch
from typing import Dict, List, Optional

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.agents.base_agent import BaseAgent
from dexmani_policy.agents.obs_encoder.moe_dp3 import MoEDP3Encoder
from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import farthest_point_sample


class MoEDP3Agent(BaseAgent):
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
        # moe params
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_out_dim: int | None = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        temperature: float = 1.0,
        residual: bool = True,
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

        self.obs_encoder = MoEDP3Encoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            point_wise=False,
            state_dim=state_dim,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            moe_out_dim=moe_out_dim,
            moe_num_layers=moe_num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
            temperature=temperature,
            residual=residual,
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

        if self.use_coord_only:
            this_obs["point_cloud"] = this_obs["point_cloud"][..., :3]

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
            this_obs["point_cloud"] = downsample_point_cloud.reshape(batch_size, time_steps, self.num_points, -1)

        this_obs = dict_apply(this_obs, lambda x: x.flatten(0, 1) if torch.is_tensor(x) else x)
        return this_obs

    def encode_obs_as_condition(
        self,
        obs_dict,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weight: Optional[torch.Tensor] = None,
    ):
        batch_size = obs_dict["point_cloud"].shape[0]
        this_obs_dict = self.preprocess(obs_dict)
        if topk_idx is not None and not torch.is_tensor(topk_idx):
            topk_idx = torch.as_tensor(topk_idx)
        if topk_weight is not None and not torch.is_tensor(topk_weight):
            topk_weight = torch.as_tensor(topk_weight)

        feat, aux = self.obs_encoder(
            this_obs_dict,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            num_groups=batch_size,
        )

        if self.backbone.condition_type == "film":
            feat = feat.reshape(batch_size, -1)
        elif self.backbone.condition_type == "cross_attention_film":
            feat = feat.reshape(batch_size, self.n_obs_steps, -1)
        else:
            raise ValueError(f"{self.backbone.condition_type} is not implemented")

        return feat, aux

    def compute_loss(self, batch, **kwargs):
        cond, moe_aux = self.encode_obs_as_condition(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        action_loss, action_loss_dict = self.action_expert.compute_loss(cond, nactions, **kwargs)
        moe_aux_loss = moe_aux.aux_loss
        total_loss = action_loss + moe_aux_loss

        loss_dict = dict(action_loss_dict)
        loss_dict["loss_action"] = action_loss
        loss_dict["loss_moe_aux"] = moe_aux_loss
        loss_dict["loss_moe_load_balance"] = moe_aux.load_balance_loss
        loss_dict["loss_moe_entropy"] = moe_aux.entropy_loss
        loss_dict["moe_active_expert_count"] = (moe_aux.expert_token_count > 0).sum().to(action_loss.dtype)
        loss_dict["moe_dominant_expert_rate"] = moe_aux.expert_activation_rate.max()
        loss_dict["loss"] = total_loss

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        denoise_timesteps=None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        cond, moe_aux = self.encode_obs_as_condition(
            obs_dict,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
        )
        action_template = torch.zeros(
            (cond.shape[0], self.horizon, self.action_dim),
            device=cond.device,
            dtype=cond.dtype,
        )
        pred_naction = self.action_expert.predict_action(cond, action_template, denoise_timesteps)
        pred_action = self.normalizer["action"].unnormalize(pred_naction)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = pred_action[:, start:end]
        active_expert_idx = torch.nonzero(moe_aux.expert_token_count > 0, as_tuple=False).flatten()
        dominant_expert_idx = torch.argmax(moe_aux.expert_token_count)
        dominant_expert_idx_by_sample = None
        if moe_aux.expert_activation_rate_by_group is not None:
            dominant_expert_idx_by_sample = torch.argmax(moe_aux.expert_activation_rate_by_group, dim=-1)

        return {
            "pred_action": pred_action,
            "control_action": action,
            "moe_activation": {
                "active_expert_idx": active_expert_idx,
                "dominant_expert_idx": dominant_expert_idx,
                "expert_token_count": moe_aux.expert_token_count,
                "expert_activation_rate": moe_aux.expert_activation_rate,
                "expert_activation_rate_by_sample": moe_aux.expert_activation_rate_by_group,
                "dominant_expert_idx_by_sample": dominant_expert_idx_by_sample,
            },
        }
