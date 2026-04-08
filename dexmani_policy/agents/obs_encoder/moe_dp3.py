import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.obs_encoder.dp3 import DP3Encoder
from dexmani_policy.agents.obs_encoder.plugins.moe import MoEAux, MoEConditioner


class MoEDP3Encoder(nn.Module):
    def __init__(
        self,
        # dp3 encoder params
        encoder_type: str = "dp3",
        pc_dim: int = 3,
        pc_out_dim: int = 256,
        point_wise: bool = False,
        state_dim: int = 19,
        # moe params
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_out_dim: Optional[int] = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        temperature: float = 1.0,
        residual: bool = True,
    ):
        super().__init__()

        self.encoder = DP3Encoder(
            type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            point_wise=point_wise,
            state_dim=state_dim,
        )
        self.conditioner = MoEConditioner(
            encoder=self.encoder,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=moe_hidden_dim,
            out_dim=moe_out_dim,
            num_layers=moe_num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
            temperature=temperature,
            residual=residual,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weight: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
    ) -> Tuple[torch.Tensor, MoEAux]:
        return self.conditioner(
            obs=obs,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            return_aux=True,
            num_groups=num_groups,
        )

    @property
    def out_shape(self) -> int:
        return self.conditioner.out_shape

    def get_optim_groups(self, weight_decay: float):
        return get_optim_group_with_no_decay(self, weight_decay)


def example():
    batch_size = 2
    num_points = 256
    pc_dim = 3
    state_dim = 19

    obs = {
        "point_cloud": torch.randn(batch_size, num_points, pc_dim),
        "joint_state": torch.randn(batch_size, state_dim),
    }

    encoder = MoEDP3Encoder(
        encoder_type="dp3",
        pc_dim=pc_dim,
        pc_out_dim=128,
        point_wise=False,
        state_dim=state_dim,
        num_experts=4,
        top_k=2,
        moe_hidden_dim=128,
        moe_out_dim=None,
        moe_num_layers=2,
        lambda_load=0.1,
        beta_entropy=0.01,
        temperature=1.0,
        residual=True,
    )

    feat, aux = encoder(obs)
    print("=== MoEDP3Encoder Example ===")
    print("feat:", tuple(feat.shape), "out_shape:", encoder.out_shape)
    print("aux_loss:", float(aux.aux_loss))
    print("expert_token_count:", aux.expert_token_count.tolist())
    print("expert_activation_rate:", aux.expert_activation_rate.tolist())

    forced_topk_idx = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    forced_feat, forced_aux = encoder(obs, topk_idx=forced_topk_idx)
    print("forced_feat:", tuple(forced_feat.shape))
    print("forced_expert_token_count:", forced_aux.expert_token_count.tolist())


if __name__ == "__main__":
    example()
