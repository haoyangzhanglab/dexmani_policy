from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import (
    ConditionalUNet1D,
    validate_unet_horizon,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import MultiStagePointNet, PointNet
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy


class DP3ObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "point_cloud")

    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        if encoder_type == "dp3":
            encoder_cls = PointNet
        elif encoder_type == "idp3":
            encoder_cls = MultiStagePointNet
        else:
            raise ValueError(f"encoder_type must be 'dp3' or 'idp3', got {encoder_type!r}")
        self.pc_encoder = encoder_cls(input_channels=pc_dim, output_channels=pc_out_dim)
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}
        self.out_dim = self.pc_encoder.out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.fps_random_config,
            training=self.training,
        )
        features = torch.cat(
            [
                self.pc_encoder(pc),
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class DP3Policy(BasePolicy):
    supported_execution_modes = ("sync", "async", "rtc")

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        state_dim: int,
        num_points: int,
        state_out_dim: int = 64,
        fps_random_config: dict | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        modality_dropout_probs: dict[str, float] | None = None,
        cond_predict_scale: bool = True,
        clip_sample: bool = True,
    ) -> None:
        validate_unet_horizon(horizon, down_dims)
        obs_encoder = DP3ObservationEncoder(
            encoder_type,
            pc_dim,
            pc_out_dim,
            state_dim,
            num_points,
            n_obs_steps,
            state_out_dim,
            fps_random_config=fps_random_config,
        )
        backbone = ConditionalUNet1D(
            input_dim=action_dim,
            context_dim=obs_encoder.out_dim * n_obs_steps,
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
            prediction_type,
            clip_sample=clip_sample,
        )
        super().__init__(
            obs_encoder,
            action_decoder,
            horizon,
            n_obs_steps,
            n_action_steps,
            action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )
