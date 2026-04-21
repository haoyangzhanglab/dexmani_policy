import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_global_encoder
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import farthest_point_sample
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import StateMLP
from dexmani_policy.agents.core.base import UNetDiffusionAgent


class DP3ObsEncoder(nn.Module):
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
    ):
        super().__init__()
        self.pc_encoder = build_pc_global_encoder(
            encoder_type, pc_dim, config={'output_channels': pc_out_dim}
        )
        self.state_mlp = StateMLP(state_dim, state_out_dim, hidden_channels=[64])
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        self.n_obs_steps = n_obs_steps
        self.condition_type = condition_type
        self.out_dim = self.pc_encoder.out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        pc = obs['point_cloud'][..., :3] if self.use_coord_only else obs['point_cloud']
        if pc.shape[1] > self.num_points:
            pc, _ = farthest_point_sample(pc, self.num_points)
        feat = torch.cat([
            self.pc_encoder(pc)['global_token'],
            self.state_mlp(obs['joint_state']),
        ], dim=-1)                                      # (B*T, out_dim)
        B = feat.shape[0] // self.n_obs_steps
        if self.condition_type == 'film':
            return feat.reshape(B, -1)
        return feat.reshape(B, self.n_obs_steps, -1)


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
        condition_type: str = 'film',
        state_out_dim: int = 64,
        **kwargs,
    ):
        obs_encoder = DP3ObsEncoder(
            encoder_type, pc_dim, pc_out_dim, state_dim,
            num_points, n_obs_steps, condition_type, state_out_dim,
        )
        super().__init__(
            obs_encoder, condition_type, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs
        )


def example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, A, N = 2, 2, 16, 19, 256

    agent = DP3Agent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        encoder_type='idp3', pc_dim=3, pc_out_dim=64, state_dim=A,
        num_points=N, condition_type='film',
        down_dims=[64, 128], diffusion_step_embed_dim=64,
        num_training_steps=10, num_inference_steps=3,
    ).to(device)

    obs = {
        'point_cloud': torch.randn(B * T, N, 3, device=device),
        'joint_state': torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    print('=== DP3Agent smoke test ===')
    print(f'obs point_cloud: {obs["point_cloud"].shape}')
    print(f'obs joint_state: {obs["joint_state"].shape}')
    print(f'action:          {action.shape}')

    cond = agent.obs_encoder(obs)
    print(f'cond:            {cond.shape}')

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
