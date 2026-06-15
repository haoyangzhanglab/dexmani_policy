"""R3D observation encoder: Uni3D + StateMLP + spatial PE concatenation.

pc_pe is concatenated along the feature dimension. The backbone splits it
from obs features and adds it to key positional encoding after projection,
exactly matching the R3D reference implementation.
"""

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DPointcloudEncoder
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import StateMLP


class R3DObsEncoder(nn.Module):
    """Uni3D point cloud encoder + StateMLP.

    forward(obs) -> (cond_tokens, {})
        cond_tokens: (B, T*K, D + D_s + D)  — feat + state + pc_pe
    """

    def __init__(
        self,
        pc_dim: int,
        num_points: int,
        state_dim: int,
        n_obs_steps: int,
        pc_encoder_config: dict = None,
        state_out_dim: int = 64,
        fps_random_config: dict = None,
    ):
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("pc_in_channels", 6)

        self.pc_encoder = Uni3DPointcloudEncoder(**pc_encoder_config)
        self.state_mlp = StateMLP(state_dim, state_out_dim)
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

        patch_tokens, pc_pe = self.pc_encoder(pc, eval=not self.training)

        state_emb = self.state_mlp(state)
        state_emb = state_emb.unsqueeze(1).expand(
            -1, patch_tokens.shape[1], -1
        )

        obs_feat = torch.cat([patch_tokens, state_emb], dim=-1)
        tokens = torch.cat([obs_feat, pc_pe], dim=-1)

        B = tokens.shape[0] // self.n_obs_steps
        return tokens.reshape(B, -1, self.obs_token_dim), {}
