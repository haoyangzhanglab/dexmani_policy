import torch
import torch.nn as nn
from typing import Dict, Any
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.agents.common.param_counter import print_param_count
from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.backbone.ditx import DiTXFlowMatch
from dexmani_policy.agents.action_decoders.flowmatch import FlowMatchWithConsistency


class BaseAgent(nn.Module):
    def __init__(
        self,
        obs_encoder: nn.Module,
        action_decoder: nn.Module,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        cond_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.cond_dropout_prob = cond_dropout_prob
        self.normalizer = LinearNormalizer()
        print_param_count(self)

    def load_normalizer_from_dataset(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def preprocess(self, obs_dict: Dict) -> Dict:
        obs = self.normalizer.normalize(obs_dict)
        result = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                v = v[:, :self.n_obs_steps].flatten(0, 1)
            result[k] = v
        return result

    @property
    def ema_backbone(self):
        return self.action_decoder.model

    def _apply_cond_dropout(self, cond):
        if self.training and self.cond_dropout_prob > 0:
            mask_shape = (cond.shape[0],) + (1,) * (cond.ndim - 1)
            keep_mask = (torch.rand(mask_shape, device=cond.device) > self.cond_dropout_prob).float()
            return cond * keep_mask
        return cond

    def _build_cond(self, obs_dict):
        obs = self.preprocess(obs_dict)
        cond, aux = self.obs_encoder(obs)
        return self._apply_cond_dropout(cond), aux

    def compute_loss(self, batch, **kwargs):
        cond, aux = self._build_cond(batch['obs'])

        nactions = self.normalizer['action'].normalize(batch['action'])

        ema_model = kwargs.pop('ema_model', None)
        if ema_model is not None:
            kwargs['ema_model'] = ema_model.ema_backbone

        return self.compute_loss_from_cond(cond, nactions, aux=aux, **kwargs)

    def compute_loss_from_cond(self, cond, nactions, aux=None, **kwargs):
        action_loss, loss_dict = self.action_decoder.compute_loss(cond, nactions, **kwargs)

        if aux and 'loss' in aux:
            total_loss = action_loss + aux['loss']
            loss_dict['loss'] = total_loss
            loss_dict['loss_action'] = action_loss
            for k, v in aux.items():
                if k != 'loss':
                    loss_dict[f'aux_{k}'] = v
            return total_loss, loss_dict

        return action_loss, loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict, denoise_timesteps=None) -> Dict:
        cond, _ = self.obs_encoder(self.preprocess(obs_dict))
        return self.predict_action_from_cond(cond, denoise_timesteps)

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None):
        template = torch.zeros(
            cond.shape[0], self.horizon, self.action_dim,
            device=cond.device, dtype=cond.dtype,
        )
        pred = self.action_decoder.predict_action(cond, template, denoise_timesteps)
        pred = self.normalizer['action'].unnormalize(pred)
        s = self.n_obs_steps - 1
        return {
            'pred_action': pred,
            'control_action': pred[:, s:s + self.n_action_steps],
        }

    @torch.no_grad()
    def compute_action_mse(self, batch: Dict[str, Any]) -> float:
        obs = batch["obs"]
        gt_action = batch["action"]
        pred_action = self.predict_action(obs)["pred_action"]
        return torch.nn.functional.mse_loss(pred_action, gt_action).item()

    def configure_optimizer(
        self, lr, weight_decay,
        obs_lr=None, obs_weight_decay=None,
        betas=(0.95, 0.999),
    ):
        obs_lr = obs_lr if obs_lr is not None else lr
        obs_wd = obs_weight_decay if obs_weight_decay is not None else weight_decay
        action_groups = self.action_decoder.model.get_optim_groups(weight_decay)
        for g in action_groups:
            g['lr'] = lr
        obs_params = [p for p in self.obs_encoder.parameters() if p.requires_grad]
        groups = action_groups + [{'params': obs_params, 'weight_decay': obs_wd, 'lr': obs_lr}]
        return torch.optim.AdamW(
            [g for g in groups if g['params']], lr=lr, betas=betas
        )


class UNetDiffusionAgent(BaseAgent):
    def __init__(
        self,
        obs_encoder: nn.Module,
        condition_type: str,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = 'sample',
        cond_dropout_prob: float = 0.0,
        cond_predict_scale: bool = True,
    ):
        if condition_type in ('film', 'mlp_film'):
            context_dim = obs_encoder.out_dim * n_obs_steps
        else:
            context_dim = obs_encoder.out_dim
        backbone = ConditionalUnet1D(
            input_dim=action_dim,
            context_dim=context_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            cond_predict_scale=cond_predict_scale,
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
        )
        action_decoder = Diffusion(backbone, num_training_steps, num_inference_steps, prediction_type)
        super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim,
                         cond_dropout_prob=cond_dropout_prob)
        self.condition_type = condition_type


class DiTXFlowMatchAgent(BaseAgent):
    def __init__(
        self,
        obs_encoder: nn.Module,
        num_obs_tokens: int,
        obs_token_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
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
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        t_sample_mode_for_flow: str = 'beta',
        t_sample_mode_for_consistency: str = 'discrete',
        dt_sample_mode_for_consistency: str = 'uniform',
        target_t_sample_mode: str = 'relative',
        cond_dropout_prob: float = 0.0,
    ):
        backbone = DiTXFlowMatch(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            num_obs_tokens=num_obs_tokens,
            obs_token_dim=obs_token_dim,
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
        action_decoder = FlowMatchWithConsistency(
            model=backbone,
            denoise_timesteps=denoise_timesteps,
            flow_batch_ratio=flow_batch_ratio,
            consistency_batch_ratio=consistency_batch_ratio,
            t_sample_mode_for_flow=t_sample_mode_for_flow,
            t_sample_mode_for_consistency=t_sample_mode_for_consistency,
            dt_sample_mode_for_consistency=dt_sample_mode_for_consistency,
            target_t_sample_mode=target_t_sample_mode,
        )
        super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim,
                         cond_dropout_prob=cond_dropout_prob)


