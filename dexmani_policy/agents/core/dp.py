import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import StateMLP
from dexmani_policy.agents.core.base import UNetDiffusionAgent


class DPObsEncoder(nn.Module):
    def __init__(
        self,
        rgb_backbone_name: str,
        state_dim: int,
        n_obs_steps: int,
        condition_type: str,
        state_out_dim: int = 64,
    ):
        super().__init__()
        self.backbone, self.image_processor = build_backbone(rgb_backbone_name)
        self.state_mlp = StateMLP(state_dim, state_out_dim, hidden_channels=[64])
        self.n_obs_steps = n_obs_steps
        self.condition_type = condition_type
        self.out_dim = self.backbone.out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        rgb = self.image_processor.process_images(obs['rgb'])['image']
        feat = torch.cat([
            self.backbone(rgb)['global_token'],
            self.state_mlp(obs['joint_state']),
        ], dim=-1)                                      # (B*T, out_dim)
        B = feat.shape[0] // self.n_obs_steps
        if self.condition_type == 'film':
            return feat.reshape(B, -1)
        return feat.reshape(B, self.n_obs_steps, -1)


class DPAgent(UNetDiffusionAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        rgb_backbone_name: str,
        state_dim: int,
        condition_type: str = 'film',
        state_out_dim: int = 64,
        **kwargs,
    ):
        obs_encoder = DPObsEncoder(
            rgb_backbone_name, state_dim, n_obs_steps, condition_type, state_out_dim
        )
        super().__init__(
            obs_encoder, condition_type, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs
        )


def example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, A = 2, 2, 16, 19

    agent = DPAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        rgb_backbone_name='resnet', state_dim=A, condition_type='film',
        down_dims=[64, 128], diffusion_step_embed_dim=64,
        num_training_steps=10, num_inference_steps=3,
    ).to(device)

    obs = {
        'rgb': torch.rand(B * T, 3, 224, 224, device=device),
        'joint_state': torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    print('=== DPAgent smoke test ===')
    print(f'obs rgb:         {obs["rgb"].shape}')
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
            'rgb': obs['rgb'].reshape(B, T, 3, 224, 224),
            'joint_state': obs['joint_state'].reshape(B, T, A),
        },
        'action': action,
    }
    loss, loss_dict = agent.compute_loss(batch)
    print(f'loss:            {loss.item():.4f}  keys={list(loss_dict.keys())}')

    result = agent.predict_action({
        'rgb': obs['rgb'].reshape(B, T, 3, 224, 224),
        'joint_state': obs['joint_state'].reshape(B, T, A),
    })
    print(f'pred_action:     {result["pred_action"].shape}')
    print(f'control_action:  {result["control_action"].shape}')
    print('=== PASSED ===')


if __name__ == '__main__':
    example()
