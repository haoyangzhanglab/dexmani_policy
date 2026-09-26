"""R3D observation encoder: Uni3D + StateMLP + spatial PE concatenation.

State features are broadcast to every spatial token and concatenated along
the feature dimension, matching the released R3D cat_on_token=false path.
pc_pe is appended separately; the backbone splits it from observation features
and adds it to key positional encoding after projection.
"""

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DPointcloudEncoder
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp


class R3DObsEncoder(nn.Module):
    """Uni3D point cloud encoder + StateMLP.

    forward(obs) -> (cond_tokens, {})
        cond_tokens: (B, T*K, D + D_s + D)  — feat + state + pc_pe
    """

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return ("joint_state", "point_cloud")

    def __init__(
        self,
        state_dim: int,
        n_obs_steps: int,
        pc_encoder_config: dict = None,
        state_out_dim: int = 256,
        fps_random_config: dict = None,
    ):
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("pc_in_channels", 6)
        if fps_random_config:
            pc_encoder_config.setdefault("fps_random_config", fps_random_config)

        self.pc_encoder = Uni3DPointcloudEncoder(**pc_encoder_config)
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.n_obs_steps = n_obs_steps

        K = pc_encoder_config.get("num_group", 512)
        D = pc_encoder_config.get("embed_dim", 256)
        D_s = state_out_dim

        self.num_pc_tokens = K
        self.num_obs_tokens = K * n_obs_steps
        self.obs_token_dim = D + D_s + D
        self.pc_pe_dim = D

    @property
    def out_dim(self) -> int:
        return self.obs_token_dim

    def forward(self, obs: dict):
        pc = obs["point_cloud"]
        state = obs["joint_state"]

        if pc.dtype != torch.float32:
            pc = pc.float()

        # The policy normalizer has already normalized this input. Official R3D
        # clamps the complete XYZRGB tensor; raw RGB in [0, 1] normally maps to
        # [-1, 1], where this is an identity. Keep the clamp local to R3D.
        pc = pc.clone()
        pc.clamp_(min=-1 - 1e-6, max=1 + 1e-6)

        patch_tokens, pc_pe = self.pc_encoder(pc, inference_mode=not self.training)

        state_emb = self.state_mlp(state)
        state_emb = state_emb.unsqueeze(1).expand(-1, patch_tokens.shape[1], -1)

        obs_feat = torch.cat([patch_tokens, state_emb], dim=-1)
        tokens = torch.cat([obs_feat, pc_pe], dim=-1)

        B = tokens.shape[0] // self.n_obs_steps
        return tokens.reshape(B, -1, self.obs_token_dim), {}
