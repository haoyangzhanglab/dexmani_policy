import torch
import torch.nn as nn
from typing import Dict, Optional

from dexmani_policy.agents.common.mlp import create_mlp
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.common.pc_state_encoder import (
    PointCloudEncoderAdapter,
    PointCloudStateEncoder,
)
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_encoder


class ManiFlowEncoder(nn.Module):
    """Point cloud + joint state encoder for ManiFlow.

    Internally delegates to PointCloudStateEncoder for pc+state fusion.
    """

    STATE_OUT_DIM = 64

    def __init__(
        self,
        type: str,
        pc_dim: int,
        state_dim: int,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        encoder, seq_len, pc_out_dim, extract_fn = build_pc_encoder(
            type, pc_dim, config
        )
        pc_adapter = PointCloudEncoderAdapter(encoder, seq_len, pc_out_dim, extract_fn)
        state_mlp = create_mlp(state_dim, [64, self.STATE_OUT_DIM])
        self.fused_encoder = PointCloudStateEncoder(pc_adapter, state_mlp, state_dim=self.STATE_OUT_DIM)

    @property
    def obs_seq_len(self) -> int:
        return self.fused_encoder.obs_seq_len

    @property
    def out_shape(self) -> int:
        return self.fused_encoder.out_shape

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict:
                point_cloud : (B*T, N, C)
                joint_state : (B*T, state_dim)
        Returns:
            feat: (B*T, seq_len, pc_out_dim + STATE_OUT_DIM)
        """
        return self.fused_encoder(obs_dict)

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(self, weight_decay)


def example():
    B, T, N = 2, 2, 1024
    pc = torch.randn(B * T, N, 3)
    state = torch.randn(B * T, 19)
    obs = {"point_cloud": pc, "joint_state": state}

    for enc_type, config in [
        ("idp3",      {"pc_out_dim": 128, "num_points": N}),
        ("pointpn",   {"num_points": N}),
        ("tokenizer", {"num_patches": 96}),
    ]:
        enc = ManiFlowEncoder(
            type=enc_type, pc_dim=3, state_dim=19, config=config
        )
        with torch.no_grad():
            feat = enc(obs)
        print(f"[{enc_type}] obs_seq_len={enc.obs_seq_len}, out_shape={enc.out_shape}, feat={tuple(feat.shape)}")


if __name__ == "__main__":
    example()
