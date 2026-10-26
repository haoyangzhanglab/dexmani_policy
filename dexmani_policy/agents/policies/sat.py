"""Single-XHand joint trajectories conditioned on Sparse3D global observations."""

from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.sat import SAT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import (
    Sparse3DEncoder,
    validate_sparse3d_config,
)
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import positive_int, validate_action_window


class SATObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "point_cloud")
    out_dim = 256
    use_coord_only = False
    validate_normalization = staticmethod(Sparse3DEncoder.validate_normalization)

    def __init__(
        self,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        n_obs_steps: int = 2,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        if (pc_dim, state_dim) != (6, 19) or n_obs_steps not in (1, 2):
            raise ValueError("Expected pc_dim=6, state_dim=19, n_obs_steps in (1, 2)")
        self.num_points = positive_int(num_points, "num_points")
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = dict(
            use_random=True,
            use_random_start=True,
            use_shuffle_output=False,
            random_noise_scale=0.0,
        )
        if fps_random_config is not None:
            self.fps_random_config.update(fps_random_config)

        config = {"num_patches": 64, "depth": 4, **(pc_encoder_config or {})}
        nested_fps = config.get("fps_random_config")
        if nested_fps is not None and dict(nested_fps) != self.fps_random_config:
            raise ValueError("Sparse3D patch FPS must match observation preprocessing FPS")
        config["fps_random_config"] = dict(self.fps_random_config)
        self.pc_encoder = Sparse3DEncoder(input_channels=pc_dim, **config)
        self.state_mlp = StateEncoder(state_dim, positive_int(state_out_dim, "state_out_dim"))
        fused_dim = n_obs_steps * (self.pc_encoder.out_dim + state_out_dim)
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.out_dim),
        )

    @property
    def effective_config(self):
        return {"pc_encoder_config": self.pc_encoder.effective_config}

    def forward(self, obs: dict) -> torch.Tensor:
        pc, state = obs["point_cloud"], obs["joint_state"]
        pc = preprocess_point_cloud(
            pc,
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )
        # Independent FPS has no cross-frame patch correspondence; pool before time fusion.
        pc_features = self.pc_encoder(pc)["tokens"].mean(dim=1)
        features = torch.cat((pc_features, self.state_mlp(state)), dim=-1)
        features = features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)
        return self.temporal_projection(features)


class SATPolicy(BasePolicy):
    """Structural Action Transformer with [B,Da,T] trajectory tokens."""

    @classmethod
    def validate_model_config(cls, cfg, spec, *, dataset, inference_steps) -> None:
        validate_sparse3d_config(cfg, spec, dataset=dataset, default_num_patches=64)
        if cfg.n_obs_steps not in (1, 2) or (
            cfg.get("pc_dim", 6), cfg.get("state_dim", 19)
        ) != (6, 19):
            raise ValueError("Sparse3D global observations require 1 or 2 XYZRGB/19D frames")
        if spec.get("joint_state") != "limits":
            raise ValueError(
                "Sparse3D global observations require normalization.joint_state: limits"
            )
        if cfg.n_obs_steps != 2:
            raise ValueError("SAT requires n_obs_steps=2")
        if dataset.action_key != "action" or (cfg.horizon, cfg.action_dim) != (16, 19):
            raise ValueError("SAT requires joint action=19, horizon=16")
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
        validate_action_window(horizon, n_obs_steps, n_action_steps)
        if n_obs_steps != 2:
            raise ValueError("SAT requires n_obs_steps=2")
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
        self, condition: torch.Tensor, inference_steps: int | None = None,
        *, generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample joint trajectories [B, Da, T] and return physical actions [B, T, Da]."""
        prediction = self.action_decoder.predict_action(
            condition.unsqueeze(1), (condition.shape[0], self.action_dim, self.horizon),
            inference_steps=inference_steps, generator=generator,
        )
        prediction = prediction.transpose(1, 2)

        prediction = self.normalizer["action"].unnormalize(prediction)
        return self.format_action_output(prediction)
