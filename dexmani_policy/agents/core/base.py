import torch
import torch.nn as nn
from typing import Dict
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
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.normalizer = LinearNormalizer()
        print_param_count(self)

    def load_normalizer_from_dataset(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def preprocess(self, obs_dict: Dict) -> Dict:
        # obs_dict: {key: (B, T, ...)}  →  output: {key: (B*T, ...)}, sliced to n_obs_steps
        obs = self.normalizer.normalize(obs_dict)
        obs = {k: v[:, :self.n_obs_steps] if torch.is_tensor(v) else v
               for k, v in obs.items()}
        obs = {k: v.flatten(0, 1) if torch.is_tensor(v) else v
               for k, v in obs.items()}
        return obs

    def compute_loss(self, batch, **kwargs):
        cond = self.obs_encoder(self.preprocess(batch['obs']))
        nactions = self.normalizer['action'].normalize(batch['action'])
        return self.action_decoder.compute_loss(cond, nactions, **kwargs)

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict, denoise_timesteps=None) -> Dict:
        cond = self.obs_encoder(self.preprocess(obs_dict))
        template = torch.zeros(
            cond.shape[0], self.horizon, self.action_dim,
            device=cond.device, dtype=cond.dtype,
        )
        pred = self.action_decoder.predict_action(cond, template, denoise_timesteps)
        pred = self.normalizer['action'].unnormalize(pred)
        # horizon 的第 0 步对应最旧的观测帧，第 n_obs_steps-1 步才是"下一步"起点。
        # 例如 n_obs_steps=2：pred[:,0] 对应 t-1 时刻，pred[:,1] 起才是待执行动作。
        s = self.n_obs_steps - 1
        return {
            'pred_action': pred,
            'control_action': pred[:, s:s + self.n_action_steps],
        }

    def configure_optimizer(
        self, lr, weight_decay,
        obs_lr=None, obs_weight_decay=None,
        betas=(0.9, 0.95),
    ):
        obs_lr = obs_lr or lr
        obs_wd = obs_weight_decay or weight_decay
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
    ):
        # film 模式：encoder 输出 (B, out_dim*T) 展平向量，context_dim 是总维度。
        # cross_attention 模式：encoder 输出 (B, T, out_dim) 序列，context_dim 是每步特征维度。
        if condition_type == 'film':
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
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
        )
        action_decoder = Diffusion(backbone, num_training_steps, num_inference_steps, prediction_type)
        super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim)
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
        super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim)

    def compute_loss(self, batch, **kwargs):
        cond = self.obs_encoder(self.preprocess(batch['obs']))
        nactions = self.normalizer['action'].normalize(batch['action'])
        ema_model = kwargs.get('ema_model')
        ema_backbone = ema_model.action_decoder.model if ema_model is not None else None
        return self.action_decoder.compute_loss(cond, nactions, ema_model=ema_backbone)
