import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_global_encoder
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import farthest_point_sample
from dexmani_policy.agents.obs_encoder.plugins.moe import MoE
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import StateMLP
from dexmani_policy.agents.core.base import UNetDiffusionAgent


class MoEObsEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        condition_type: str,
        state_out_dim: int = 64,
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_out_dim: int = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
    ):
        super().__init__()
        self.pc_encoder = build_pc_global_encoder(
            encoder_type, pc_dim, config={'output_channels': pc_out_dim}
        )
        self.state_mlp = StateMLP(state_dim, state_out_dim, hidden_channels=[64])
        in_dim = self.pc_encoder.out_dim + self.state_mlp.out_dim
        self.moe = MoE(
            dim=in_dim,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=moe_hidden_dim,
            out_dim=moe_out_dim if moe_out_dim is not None else in_dim,
            num_layers=moe_num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
        )
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        self.n_obs_steps = n_obs_steps
        self.condition_type = condition_type
        self.out_dim = self.moe.out_dim

    def _encode_feat(self, obs: dict) -> torch.Tensor:
        pc = obs['point_cloud'][..., :3] if self.use_coord_only else obs['point_cloud']
        if pc.shape[1] > self.num_points:
            pc, _ = farthest_point_sample(pc, self.num_points)
        return torch.cat([
            self.pc_encoder(pc)['global_token'],
            self.state_mlp(obs['joint_state']),
        ], dim=-1)                                      # (B*T, in_dim)

    def _reshape(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0] // self.n_obs_steps
        if self.condition_type == 'film':
            return feat.reshape(B, -1)
        return feat.reshape(B, self.n_obs_steps, -1)

    def encode(self, obs: dict):
        """训练路径：返回 (cond, aux_dict)"""
        z = self._encode_feat(obs)
        feat, aux = self.moe(z, return_aux=True)
        return self._reshape(feat), aux

    def forward(self, obs: dict) -> torch.Tensor:
        """推理路径：返回 cond"""
        return self._reshape(self.moe(self._encode_feat(obs)))


class MoEAgent(UNetDiffusionAgent):
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
        condition_type: str = 'film',
        state_out_dim: int = 64,
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_out_dim: int = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        **kwargs,
    ):
        obs_encoder = MoEObsEncoder(
            encoder_type, pc_dim, pc_out_dim, state_dim, num_points,
            n_obs_steps, condition_type, state_out_dim,
            num_experts=num_experts, top_k=top_k, moe_hidden_dim=moe_hidden_dim,
            moe_out_dim=moe_out_dim, moe_num_layers=moe_num_layers,
            lambda_load=lambda_load, beta_entropy=beta_entropy,
        )
        super().__init__(
            obs_encoder, condition_type, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs
        )

    def compute_loss(self, batch, **kwargs):
        # 训练时必须走 encode() 而非 forward()，以获取 MoE aux loss。
        # 推理时 BaseAgent.predict_action 调用 forward()（无 aux），两条路径不对称，子类不可省略此覆盖。
        cond, aux = self.obs_encoder.encode(self.preprocess(batch['obs']))
        nactions = self.normalizer['action'].normalize(batch['action'])
        action_loss, loss_dict = self.action_decoder.compute_loss(cond, nactions)
        total = action_loss + aux['loss']
        loss_dict.update({
            'loss': total,
            'loss_action': action_loss,
            'loss_moe_aux': aux['loss'],
            'loss_moe_load_balance': aux['load_balance_loss'],
            'loss_moe_entropy': aux['entropy_loss'],
        })
        return total, loss_dict


def example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, A, N = 2, 2, 16, 19, 256

    agent = MoEAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        encoder_type='idp3', pc_dim=3, pc_out_dim=64, state_dim=A,
        num_points=N, condition_type='film',
        num_experts=4, top_k=2, moe_hidden_dim=64, moe_num_layers=1,
        down_dims=[64, 128], diffusion_step_embed_dim=64,
        num_training_steps=10, num_inference_steps=3,
    ).to(device)

    obs = {
        'point_cloud': torch.randn(B * T, N, 3, device=device),
        'joint_state': torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    print('=== MoEAgent smoke test ===')
    print(f'obs point_cloud: {obs["point_cloud"].shape}')
    print(f'obs joint_state: {obs["joint_state"].shape}')
    print(f'action:          {action.shape}')

    cond, aux = agent.obs_encoder.encode(obs)
    print(f'cond:            {cond.shape}')
    print(f'aux loss:        {aux["loss"].item():.4f}')

    from dexmani_policy.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    normalizer.fit({'action': action, 'joint_state': obs['joint_state'].reshape(B, T, A)}, mode='limits')
    agent.load_normalizer_from_dataset(normalizer)

    batch = {
        'obs': {
            'point_cloud': obs['point_cloud'].reshape(B, T, N, 3),
            'joint_state': obs['joint_state'].reshape(B, T, A),
        },
        'action': action,
    }
    loss, loss_dict = agent.compute_loss(batch)
    print(f'loss:            {loss.item():.4f}  keys={list(loss_dict.keys())}')

    result = agent.predict_action({
        'point_cloud': obs['point_cloud'].reshape(B, T, N, 3),
        'joint_state': obs['joint_state'].reshape(B, T, A),
    })
    print(f'pred_action:     {result["pred_action"].shape}')
    print(f'control_action:  {result["control_action"].shape}')
    print('=== PASSED ===')


if __name__ == '__main__':
    example()
