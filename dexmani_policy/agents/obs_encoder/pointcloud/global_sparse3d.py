"""Per-frame Sparse3D global pooling followed by ordered temporal fusion."""

from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.common.config import positive_int, validate_sparse3d_config

from .ops import preprocess_point_cloud
from .sparse3d_encoder import Sparse3DEncoder


class Sparse3DGlobalObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "point_cloud")
    out_dim = 256
    use_coord_only = False

    @staticmethod
    def validate_config(cfg, spec) -> None:
        validate_sparse3d_config(cfg, spec, default_num_patches=64)
        if (cfg.n_obs_steps, cfg.agent.get("pc_dim", 6), cfg.agent.get("state_dim", 19)) != (
            2,
            6,
            19,
        ):
            raise ValueError(
                "Sparse3D global observations require 2 frames of XYZRGB and 19D state"
            )
        if spec.get("joint_state") != "limits":
            raise ValueError(
                "Sparse3D global observations require normalization.joint_state: limits"
            )

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
        if (pc_dim, state_dim, n_obs_steps) != (6, 19, 2):
            raise ValueError("Expected pc_dim=6, state_dim=19, n_obs_steps=2")
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
        self.pc_encoder.effective_config = config
        self.state_mlp = StateEncoder(state_dim, positive_int(state_out_dim, "state_out_dim"))
        fused_dim = n_obs_steps * (self.pc_encoder.out_dim + state_out_dim)
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.out_dim),
        )

    @property
    def warmup_num_points(self) -> int:
        return self.num_points

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
