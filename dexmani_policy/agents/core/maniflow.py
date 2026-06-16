import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_patch_tokenizer
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.core.base import DiTXFlowMatchAgent


class ManiFlowObsEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        pc_encoder_config: dict = None,
        fps_random_config: dict = None,
    ):
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("fps_random_config", fps_random_config)
        self.pc_encoder = build_pc_patch_tokenizer(encoder_type, pc_dim, pc_encoder_config)
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}
        patch_seq_len, pc_out_dim = self.pc_encoder.out_shape  # (num_patches, channels)
        self.num_obs_tokens = (patch_seq_len + 1) * n_obs_steps   # (patches + global) * T
        self.obs_token_dim = pc_out_dim + self.state_mlp.out_dim

    def get_global_token(self, patch_token, patch_center) -> torch.Tensor:
        return self.pc_encoder.get_global_token(patch_token, patch_center)

    def forward(self, obs: dict):
        pc = preprocess_point_cloud(obs['point_cloud'], self.num_points,
                                     self.use_coord_only, self.fps_random_config)
        pc_outputs = self.pc_encoder(pc, return_global_token=True)
        patch_token, _, global_token = pc_outputs[0], pc_outputs[1], pc_outputs[2]
        pc_feat = torch.cat([global_token, patch_token], dim=1)

        state_feat = self.state_mlp(obs['joint_state'])
        state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)
        feat = torch.cat([pc_feat, state_feat], dim=-1)  # (B*T, K+1, obs_token_dim)

        B = feat.shape[0] // self.n_obs_steps
        # DiTX uses token-based condition: (B*T, K+1, D) → (B, T*(K+1), D)
        return feat.reshape(B, -1, self.obs_token_dim), {}


class ManiFlowAgent(DiTXFlowMatchAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        encoder_type: str,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        state_out_dim: int = 64,
        pc_encoder_config: dict = None,
        fps_random_config: dict = None,
        **kwargs,
    ):
        obs_encoder = ManiFlowObsEncoder(
            encoder_type, pc_dim, state_dim, num_points,
            n_obs_steps, state_out_dim, pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        super().__init__(
            obs_encoder,
            num_obs_tokens=obs_encoder.num_obs_tokens,
            obs_token_dim=obs_encoder.obs_token_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            **kwargs,
        )


def example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, A, N = 2, 2, 16, 19, 1024

    agent = ManiFlowAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        encoder_type='pointnext_tokenizer', pc_dim=3, state_dim=A, num_points=N,
        pc_encoder_config={
            'stem_channels': 48,
            'token_channels': 128,
            'num_patches': 96,
            'patch_radii': (0.05, 0.10),
            'patch_neighbors': (20, 48),
        },
        n_layers=2, hidden_dim=128, n_head=4, mlp_ratio=2.0,
        p_drop_attn=0.0, timestep_embed_dim=64, target_t_embed_dim=64,
        denoise_timesteps=5,
    ).to(device)

    obs = {
        'point_cloud': torch.randn(B * T, N, 3, device=device),
        'joint_state': torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    print('=== ManiFlowAgent smoke test ===')
    print(f'obs point_cloud:  {obs["point_cloud"].shape}')
    print(f'obs joint_state:  {obs["joint_state"].shape}')
    print(f'action:           {action.shape}')

    cond, _ = agent.obs_encoder(obs)
    print(f'cond (tokens):    {cond.shape}  '
          f'[B, T*(K+1), obs_token_dim] = [{B}, {cond.shape[1]}, {cond.shape[2]}]')

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
    # compute_loss 需要 EMA teacher，smoke test 直接用 agent 自身充当
    import copy
    ema_agent = copy.deepcopy(agent)
    loss, loss_dict = agent.compute_loss(batch, ema_backbone=ema_agent.action_decoder.model)
    print(f'loss:             {loss.item():.4f}  keys={list(loss_dict.keys())}')

    result = agent.predict_action({
        'point_cloud': obs['point_cloud'].reshape(B, T, N, 3),
        'joint_state': obs['joint_state'].reshape(B, T, A),
    })
    print(f'pred_action:      {result["pred_action"].shape}')
    print(f'control_action:   {result["control_action"].shape}')
    print('=== PASSED ===')


if __name__ == '__main__':
    example()
