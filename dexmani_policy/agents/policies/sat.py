"""Single-XHand joint trajectories conditioned on Sparse3D global observations."""

from __future__ import annotations

import torch

from dexmani_policy.agents.action_decoders.backbones.sat import SAT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.pointcloud.global_sparse3d import (
    Sparse3DGlobalObservationEncoder,
)
from dexmani_policy.agents.policies.base import BasePolicy


class SATPolicy(BasePolicy):
    """Structural Action Transformer with [B,Da,T] trajectory tokens."""

    @classmethod
    def validate_model_config(cls, cfg, spec, *, dataset, inference_steps) -> None:
        Sparse3DGlobalObservationEncoder.validate_config(cfg, spec, dataset=dataset)
        if cfg.n_obs_steps != 2:
            raise ValueError("SAT requires n_obs_steps=2")
        if dataset.action_key != "action" or (cfg.horizon, cfg.n_action_steps, cfg.action_dim) != (
            16,
            8,
            19,
        ):
            raise ValueError("SAT requires joint action=19, horizon=16, n_action_steps=8")
        if cfg.get("use_aux_ee", False) or cfg.get("hidden_dim", 256) != 256:
            raise ValueError("SAT requires hidden_dim=256 without auxiliary actions")

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.0,
        num_inference_steps: int = 10,
        t_sample_mode_for_flow: str = "uniform",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        modality_dropout_probs: dict | None = None,
        num_flow_train_timesteps: int = 10,
    ) -> None:
        if n_action_steps != 8 or n_obs_steps != 2:
            raise ValueError("SAT requires n_action_steps=8, n_obs_steps=2")
        obs_encoder = Sparse3DGlobalObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )

        backbone = SAT(
            horizon=horizon,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            p_drop_attn=p_drop_attn,
        )

        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
            num_flow_train_timesteps=num_flow_train_timesteps,
            t_sample_mode=t_sample_mode_for_flow,
            beta_s=beta_s,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )

        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

    def compute_loss(self, batch: dict, *, log_metrics: bool = True) -> tuple[torch.Tensor, dict]:
        self.validate_batch(batch)
        condition = self.encode_observation(batch["obs"])
        normalized_actions = self.normalizer["action"].normalize(batch["action"])
        return self.action_decoder.compute_loss(
            condition.unsqueeze(1), normalized_actions.transpose(1, 2), log_metrics=log_metrics
        )

    @torch.no_grad()
    def predict_action_from_condition(
        self, condition: torch.Tensor, inference_steps: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample joint trajectories [B, Da, T] and return physical actions [B, T, Da]."""
        template = condition.new_zeros((condition.shape[0], self.action_dim, self.horizon))

        prediction = self.action_decoder.predict_action(
            condition.unsqueeze(1), template, inference_steps=inference_steps
        )
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite SAT action prediction")
        prediction = prediction.transpose(1, 2)

        prediction = self.normalizer["action"].unnormalize(prediction)
        return self.format_action_output(prediction)
