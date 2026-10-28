from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import (
    ConditionalUNet1D,
    validate_unet_horizon,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.interaction.future_tactile.diffusion_force_head import (
    DiffusionForceHead,
)
from dexmani_policy.agents.obs_encoder.interaction.future_tactile.force_guided_attention_fusion import (
    ForceGuidedAttentionFusion,
)
from dexmani_policy.agents.obs_encoder.interaction.future_tactile.force_prediction_transformer import (
    ForcePredictionTransformer,
)
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder
from dexmani_policy.agents.policies.base import BasePolicy, parameter_groups


class AdapTacObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "point_cloud", "finger_tactile_force")
    validate_normalization = staticmethod(Sparse3DEncoder.validate_normalization)

    def __init__(
        self,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        feature_dim: int,
        state_out_dim: int,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.pc_encoder = Sparse3DEncoder(
            input_channels=pc_dim, fps_random_config=fps_random_config, **(pc_encoder_config or {}),
        )
        self.visual_projection = nn.Linear(self.pc_encoder.out_dim, feature_dim)
        self.tactile_encoder = XHandForceEncoder()
        self.tactile_projection = nn.Linear(
            self.tactile_encoder.num_tokens * self.tactile_encoder.out_dim, feature_dim
        )
        self.state_mlp = StateEncoder(state_dim, state_out_dim)

    @property
    def effective_config(self) -> dict:
        return {
            "pc_encoder_config": self.pc_encoder.effective_config,
            "fps_random_config": self.pc_encoder.fps_random_config,
        }

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.pc_encoder.fps_random_config,
            training=self.training,
        )
        visual = self.visual_projection(self.pc_encoder(pc)["tokens"].mean(dim=1))
        tactile = self.tactile_projection(
            self.tactile_encoder(obs["finger_tactile_force"]).flatten(1)
        )
        state = self.state_mlp(obs["joint_state"])
        return visual, tactile, state


class AdapTacPolicy(BasePolicy):
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        feature_dim: int = 512,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        clip_sample: bool = True,
        num_training_steps: int = 100,
        num_inference_steps: int = 20,
        force_num_inference_steps: int = 20,
        force_scale: float = 1.0,
        force_loss_weight: float = 10.0,
        force_predictor_config: dict | None = None,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
    ) -> None:
        if (n_obs_steps, action_dim, state_dim) != (1, 19, 19):
            raise ValueError("AdapTac requires single-frame observations and 19D joint state/actions")
        if not math.isfinite(force_scale) or force_scale <= 0:
            raise ValueError("force_scale must be finite and positive")
        validate_unet_horizon(horizon, down_dims)
        obs_encoder = AdapTacObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            feature_dim=feature_dim,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        backbone = ConditionalUNet1D(
            input_dim=action_dim,
            context_dim=feature_dim + state_out_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        action_decoder = Diffusion(
            backbone,
            num_training_steps,
            num_inference_steps,
            prediction_type="epsilon",
            clip_sample=clip_sample,
        )
        super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim)
        self.force_predictor = DiffusionForceHead(
            ForcePredictionTransformer(
                horizon=self.future_steps,
                cond_dim=2 * feature_dim,
                **(force_predictor_config or {}),
            ),
            num_training_steps,
            force_num_inference_steps,
        )
        self.force_fusion = ForceGuidedAttentionFusion(
            visual_dim=feature_dim,
            tactile_dim=feature_dim,
            force_dim=3 * (n_obs_steps + self.future_steps),
            hidden_dim=feature_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )
        self.force_scale = float(force_scale)
        self.force_loss_weight = force_loss_weight

    def current_force(self, obs: dict) -> torch.Tensor:
        return obs["finger_tactile_force"].float().sum(dim=-2) / self.force_scale

    def action_condition(self, features, current_force, predicted_force) -> torch.Tensor:
        visual, tactile, state = features
        guide = torch.cat((current_force, predicted_force), dim=1).flatten(1)
        fused = self.force_fusion(visual, tactile, guide)
        return torch.cat((fused, state), dim=-1)

    def compute_loss(
        self, batch: dict, *, log_metrics: bool = True, **kwargs: Any,
    ) -> tuple[torch.Tensor, dict]:
        current_force = self.current_force(batch["obs"])
        features = self.obs_encoder(self.preprocess(batch["obs"]))
        force_loss, predicted_force = self.force_predictor.compute_loss_and_prediction(
            torch.cat(features[:2], dim=-1).unsqueeze(1),
            batch["future_force"].float() / self.force_scale,
        )
        # The action objective also trains the force predictor through this prediction.
        condition = self.action_condition(features, current_force, predicted_force)
        action_loss, metrics = self.action_decoder.compute_loss(
            condition,
            self.normalizer["action"].normalize(batch["action"]),
            log_metrics=log_metrics,
            **kwargs,
        )
        weighted_force_loss = self.force_loss_weight * force_loss
        loss = action_loss + weighted_force_loss
        if log_metrics:
            metrics.update(
                loss=loss.detach(),
                loss_force=force_loss.detach(),
                loss_force_weighted=weighted_force_loss.detach(),
            )
        return loss, metrics

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: dict,
        inference_steps: int | None = None,
        *,
        generator: torch.Generator | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        current_force = self.current_force(obs_dict)
        features = self.obs_encoder(self.preprocess(obs_dict))
        predicted_force = self.force_predictor.predict_action(
            torch.cat(features[:2], dim=-1).unsqueeze(1), (current_force.shape[0], self.future_steps, 3),
            generator=generator,
        )
        condition = self.action_condition(features, current_force, predicted_force)
        return self.predict_action_from_condition(
            condition,
            inference_steps=inference_steps,
            generator=generator,
            rtc_prefix=rtc_prefix,
            delay_steps=delay_steps,
            rtc_guidance_cap=rtc_guidance_cap,
        )

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay) -> list[dict]:
        groups = parameter_groups(self.obs_encoder, lr=obs_lr, weight_decay=obs_weight_decay)
        for module in (self.force_predictor, self.force_fusion, self.action_decoder):
            groups += parameter_groups(module, lr=lr, weight_decay=weight_decay)
        return groups

    def compile_backbone(self, **compile_kwargs: Any) -> None:
        super().compile_backbone(**compile_kwargs)
        self.force_predictor.model = cast(
            nn.Module, torch.compile(self.force_predictor.model, **compile_kwargs),
        )
