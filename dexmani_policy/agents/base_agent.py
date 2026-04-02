import torch
from typing import Dict, Tuple

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.agents.common.module_attr_mixin import ModuleAttrMixin

from dexmani_policy.agents.obs_encoder.pointcloud.utils import farthest_point_sample


class BaseAgent(ModuleAttrMixin):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int, 
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.normalizer = LinearNormalizer()


    def load_normalizer_from_dataset(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def configure_optimizer(self, lr: float, weight_decay: float,
                             obs_lr: float = None, obs_weight_decay: float = None,
                             betas: Tuple[float, float]=(0.9,0.95)):
        action_optim_groups = self.backbone.get_optim_groups(weight_decay)
        for group in action_optim_groups:
            group['lr'] = lr

        obs_lr = obs_lr if obs_lr is not None else lr
        obs_weight_decay = obs_weight_decay if obs_weight_decay is not None else weight_decay
        obs_optim_groups = self.obs_encoder.get_optim_groups(obs_weight_decay)
        for group in obs_optim_groups:
            group['lr'] = obs_lr

        optim_groups = action_optim_groups + obs_optim_groups
        optim_groups = [g for g in optim_groups if len(g["params"]) > 0]
        all_params = [p for g in optim_groups for p in g["params"]]
        assert len(all_params) == len(set(map(id, all_params))), \
            "Some parameters appear in more than one optimizer group"

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer


    def encode_obs_as_condition(self, obs_dict:Dict) -> torch.Tensor:
        raise NotImplementedError


    def compute_loss(self, batch, **kwargs):
        cond = self.encode_obs_as_condition(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        loss, loss_dict = self.action_expert.compute_loss(cond, nactions, **kwargs)
        return loss, loss_dict


    @torch.no_grad()
    def predict_action(self, obs_dict:Dict[str, torch.Tensor], denoise_timesteps=None) -> torch.Tensor:
        cond = self.encode_obs_as_condition(obs_dict)
        action_template = torch.zeros((cond.shape[0], self.horizon, self.action_dim), device=cond.device, dtype=cond.dtype)
        pred_naction = self.action_expert.predict_action(cond, action_template, denoise_timesteps)
        pred_action = self.normalizer['action'].unnormalize(pred_naction) 

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = pred_action[:,start:end]

        return {
            "pred_action": pred_action,
            "control_action": action,
        }


    #################################################################################
    #                   处理输入数据，在encode_obs_as_condition中调用
    #################################################################################
    def normalize_and_slice_obs(self, obs_dict:Dict) -> Dict:
        nobs = self.normalizer.normalize(obs_dict)
        this_obs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...] if torch.is_tensor(x) else x)
        return this_obs


    @staticmethod
    def preprocess_point_cloud(this_obs:Dict[str, torch.Tensor], num_points:int=1024, use_coord_only:bool=True):
        if use_coord_only:
            this_obs['point_cloud'] = this_obs['point_cloud'][..., :3]

        current_num_points = this_obs['point_cloud'].shape[2]
        if current_num_points < num_points:
            raise ValueError(f"Point cloud shape wrong, required {num_points} points, got {current_num_points} points")
        elif current_num_points == num_points:
            pass
        else:
            downsample_point_cloud, _ = farthest_point_sample(this_obs['point_cloud'].flatten(0, 1), num_points=num_points)
            B, T = this_obs['point_cloud'].shape[:2]
            this_obs['point_cloud'] = downsample_point_cloud.reshape(B, T, num_points, -1)

        return this_obs