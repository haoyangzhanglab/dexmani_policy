"""Local SAT policy with temporal feature fusion and (B, Da, T) joint trajectories."""

from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.sat import SAT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import positive_int, validate_sparse3d_config


class SATObservationEncoder(nn.Module):
    """Fuse frame features as [B,K+1,T*D]; patch slots need not match across frames."""

    consumed_observation_fields = ("joint_state", "point_cloud")

    @property
    def warmup_num_points(self) -> int:
        return self.num_points

    def __init__(
        self,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("fps_random_config", fps_random_config)

        positive_int(num_points, "num_points")
        positive_int(n_obs_steps, "n_obs_steps")
        pc_encoder_config.setdefault("num_patches", 96)
        self.pc_encoder = Sparse3DEncoder(input_channels=pc_dim, **pc_encoder_config)
        self.pc_encoder.effective_config = dict(pc_encoder_config)
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}

        self.obs_token_dim = n_obs_steps * (self.pc_encoder.out_dim + self.state_mlp.out_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )

        patch_tokens = self.pc_encoder(pc)["tokens"]
        global_token = patch_tokens.mean(dim=1, keepdim=True)
        pc_features = torch.cat([global_token, patch_tokens], dim=1)

        state_features = self.state_mlp(obs["joint_state"])
        state_features = state_features.unsqueeze(1).expand(-1, pc_features.size(1), -1)
        features = torch.cat([pc_features, state_features], dim=-1)

        return features.unflatten(0, (-1, self.n_obs_steps)).transpose(1, 2).flatten(2)


class SATPolicy(BasePolicy):
    """Structural Action Transformer with [B,Da,T] trajectory tokens."""

    @classmethod
    def validate_model_config(cls, cfg, spec) -> None:
        validate_sparse3d_config(cfg, spec)

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
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.1,
        ejc_num_embodiments: int = 1,
        ejc_num_functions: int | None = None,
        ejc_num_axes: int = 1,
        ejc_embodiment_dim: int = 8,
        ejc_function_dim: int = 32,
        ejc_axis_dim: int = 8,
        num_inference_steps: int = 10,
        t_sample_mode_for_flow: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        modality_dropout_probs: dict | None = None,
        num_flow_train_timesteps: int = 10,
    ) -> None:
        obs_encoder = SATObservationEncoder(
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
            obs_token_dim=obs_encoder.obs_token_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            p_drop_attn=p_drop_attn,
            ejc_num_embodiments=ejc_num_embodiments,
            ejc_num_functions=ejc_num_functions,
            ejc_num_axes=ejc_num_axes,
            ejc_embodiment_dim=ejc_embodiment_dim,
            ejc_function_dim=ejc_function_dim,
            ejc_axis_dim=ejc_axis_dim,
        )

        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
            **(
                {"num_flow_train_timesteps": num_flow_train_timesteps}
                if t_sample_mode_for_flow in ("discrete", "discrete_pow")
                else {}
            ),
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
            condition, normalized_actions.transpose(1, 2), log_metrics=log_metrics
        )

    @torch.no_grad()
    def predict_action_from_condition(
        self, condition: torch.Tensor, inference_steps: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample joint trajectories [B, Da, T] and return physical actions [B, T, Da]."""
        template = condition.new_zeros((condition.shape[0], self.action_dim, self.horizon))

        prediction = self.action_decoder.predict_action(
            condition, template, inference_steps=inference_steps
        )
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite SAT action prediction")
        prediction = prediction.transpose(1, 2)

        prediction = self.normalizer["action"].unnormalize(prediction)
        return self.format_action_output(prediction)
