import torch
import torch.nn as nn
from typing import Callable, Dict, Optional


class PointCloudEncoderAdapter(nn.Module):
    """Adapt a registry-built (encoder, extract_fn) pair to a standard interface.

    The registry returns (encoder, seq_len, out_dim, extract_fn) where extract_fn
    is a closure that takes (B, *, D_pc) → (B, seq_len, out_dim). This wrapper
    exposes those as properties and a forward method, making it composable with
    PointCloudStateEncoder.

    Tensor shapes:
        pc:      (B, N, C)
        output:  (B, seq_len, out_dim)
    """

    def __init__(
        self,
        encoder: nn.Module,
        seq_len: int,
        out_dim: int,
        extract_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.encoder = encoder
        self._seq_len = seq_len
        self._out_dim = out_dim
        self.extract_fn = extract_fn

    @property
    def obs_seq_len(self) -> int:
        return self._seq_len

    @property
    def out_shape(self) -> int:
        return self._out_dim

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.extract_fn(obs_dict["point_cloud"])


class PointCloudStateEncoder(nn.Module):
    """Fuse a point-cloud encoder's output with an optional state MLP.

    This wrapper unifies the common pattern of:
        pc_feat = pc_encoder(point_cloud)       # (B, seq_len, D_pc)
        state_feat = state_mlp(joint_state)     # (B, D_state)
        fused = cat(pc_feat, state broadcast)   # (B, seq_len, D_pc + D_state)

    Tensor shapes:
        point_cloud: (B, N, C)
        joint_state: (B, state_dim)
        output:      (B, seq_len, D_pc + D_state)

    If ``state_mlp`` is None, the encoder passes through the point-cloud
    features unchanged.
    """

    def __init__(
        self,
        pc_encoder: PointCloudEncoderAdapter,
        state_mlp: Optional[nn.Module] = None,
        state_dim: int = 0,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.state_mlp = state_mlp
        self._state_dim = state_dim

    @property
    def obs_seq_len(self) -> int:
        return self.pc_encoder.obs_seq_len

    @property
    def out_shape(self) -> int:
        return self.pc_encoder.out_shape + self._state_dim

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict:
                point_cloud: (B, N, C)
                joint_state: (B, state_dim) — only used if state_mlp is set
        Returns:
            feat: (B, seq_len, D_pc + D_state)
        """
        pc_feat = self.pc_encoder(obs_dict)  # (B, seq_len, D_pc)

        if self.state_mlp is not None:
            state = obs_dict["joint_state"]
            state_feat = self.state_mlp(state)  # (B, D_state)
            state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)
            return torch.cat([pc_feat, state_feat], dim=-1)

        return pc_feat
