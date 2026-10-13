import torch
import torch.nn as nn

from dexmani_policy.agents.core.base import UNetDiffusionAgent
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_global_encoder
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp


class DP3ObsEncoder(nn.Module):
    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return ("joint_state", "point_cloud")

    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        fps_random_config: dict = None,
    ):
        super().__init__()
        self.pc_encoder = build_pc_global_encoder(
            encoder_type,
            pc_dim,
            config={
                "output_channels": pc_out_dim,
                **({"fps_random_config": fps_random_config} if encoder_type == "pointnext" else {}),
            },
        )
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}
        self.out_dim = self.pc_encoder.out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict):
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )
        feat = torch.cat(
            [
                self.pc_encoder(pc)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        B = feat.shape[0] // self.n_obs_steps
        return feat.reshape(B, -1), {}


class DP3Agent(UNetDiffusionAgent):
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
        fps_random_config: dict = None,
        **kwargs,
    ):
        obs_encoder = DP3ObsEncoder(
            encoder_type,
            pc_dim,
            pc_out_dim,
            state_dim,
            num_points,
            n_obs_steps,
            state_out_dim,
            fps_random_config=fps_random_config,
        )
        super().__init__(obs_encoder, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs)
