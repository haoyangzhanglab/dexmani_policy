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
        state_out_dim: int = 64,
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_out_dim: int = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        use_boost: bool = False,
        boost_start_epoch: int = 0,
        boost_interval: int = 100,
        boost_experts_per_step: int = 4,
        boost_topk_per_step: int = 1,
        use_enhanced_gate: bool = False,
        gate_hidden_dim: int = None,
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        self.pc_encoder = build_pc_global_encoder(
            encoder_type, pc_dim, config={'output_channels': pc_out_dim}
        )
        self.state_mlp = StateMLP(state_dim, state_out_dim, hidden_channels=[64])
        in_dim = self.pc_encoder.out_dim + self.state_mlp.out_dim
        # Align with official MoE-DP: Linear projection before MoE so the
        # encoder output can be mapped to a fixed embedding dimension
        # (analogous to cond_obs_emb in the official code).
        proj_dim = moe_out_dim if moe_out_dim is not None else in_dim
        self.obs_proj = nn.Linear(in_dim, proj_dim)
        self.moe = MoE(
            dim=proj_dim,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=moe_hidden_dim,
            out_dim=moe_out_dim if moe_out_dim is not None else proj_dim,
            num_layers=moe_num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
            use_boost=use_boost,
            boost_start_epoch=boost_start_epoch,
            boost_interval=boost_interval,
            boost_experts_per_step=boost_experts_per_step,
            boost_topk_per_step=boost_topk_per_step,
            use_enhanced_gate=use_enhanced_gate,
            gate_hidden_dim=gate_hidden_dim,
            gate_dropout=gate_dropout,
        )
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.moe.out_dim

    def encode_feat(self, obs: dict) -> torch.Tensor:
        pc = obs['point_cloud'][..., :3] if self.use_coord_only else obs['point_cloud']
        if pc.shape[1] > self.num_points:
            pc, _ = farthest_point_sample(pc, self.num_points)
        return torch.cat([
            self.pc_encoder(pc)['global_token'],
            self.state_mlp(obs['joint_state']),
        ], dim=-1)

    def forward(self, obs: dict, return_aux=True, override_idx=None):
        z = self.encode_feat(obs)
        z = self.obs_proj(z)
        if override_idx is not None:
            override_idx = override_idx.repeat_interleave(self.n_obs_steps)
        feat, aux = self.moe(z, return_aux=return_aux, override_idx=override_idx)
        B = feat.shape[0] // self.n_obs_steps
        return feat.reshape(B, -1), aux


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
        state_out_dim: int = 64,
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = 256,
        moe_out_dim: int = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        use_boost: bool = False,
        boost_start_epoch: int = 0,
        boost_interval: int = 100,
        boost_experts_per_step: int = 4,
        boost_topk_per_step: int = 1,
        use_enhanced_gate: bool = False,
        gate_hidden_dim: int = None,
        gate_dropout: float = 0.0,
        **kwargs,
    ):
        obs_encoder = MoEObsEncoder(
            encoder_type, pc_dim, pc_out_dim, state_dim, num_points,
            n_obs_steps, state_out_dim,
            num_experts=num_experts, top_k=top_k, moe_hidden_dim=moe_hidden_dim,
            moe_out_dim=moe_out_dim, moe_num_layers=moe_num_layers,
            lambda_load=lambda_load, beta_entropy=beta_entropy,
            use_boost=use_boost,
            boost_start_epoch=boost_start_epoch,
            boost_interval=boost_interval,
            boost_experts_per_step=boost_experts_per_step,
            boost_topk_per_step=boost_topk_per_step,
            use_enhanced_gate=use_enhanced_gate,
            gate_hidden_dim=gate_hidden_dim,
            gate_dropout=gate_dropout,
        )
        super().__init__(
            obs_encoder, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs
        )

    def set_epoch(self, epoch: int):
        """Trainer hook – triggers boost expert/top_k schedule at epoch start."""
        if hasattr(self.obs_encoder, 'moe'):
            self.obs_encoder.moe.update_expert_num(epoch)

    def _build_cond(self, obs_dict, override_idx=None):
        obs = self.preprocess(obs_dict)
        cond, aux = self.obs_encoder(obs, override_idx=override_idx)
        return cond, aux

    def compute_loss(self, batch, **kwargs):
        cond, aux = self._build_cond(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        action_loss, loss_dict = self.action_decoder.compute_loss(cond, nactions, **kwargs)
        total_loss = action_loss + aux['loss']
        loss_dict['loss'] = total_loss
        loss_dict['loss_action'] = action_loss
        for k, v in aux.items():
            if k == 'loss':
                continue
            if torch.is_tensor(v) and v.numel() > 1:
                continue  # non-scalar aux (e.g. router_probs, dispatch, f_i, p_i) — skip logging
            loss_dict[f'aux_{k}'] = v
        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict, denoise_timesteps=None, override_idx=None):
        cond, _ = self._build_cond(obs_dict, override_idx=override_idx)
        return self.predict_action_from_cond(cond, denoise_timesteps)


def example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, A, N = 2, 2, 16, 19, 256

    agent = MoEAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        encoder_type='idp3', pc_dim=3, pc_out_dim=64, state_dim=A,
        num_points=N,
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

    cond, aux = agent.obs_encoder(obs)
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

    # test override_idx
    override = torch.tensor([0, 1], dtype=torch.long, device=device)
    result_ov = agent.predict_action({
        'point_cloud': obs['point_cloud'].reshape(B, T, N, 3),
        'joint_state': obs['joint_state'].reshape(B, T, A),
    }, override_idx=override)
    print(f'override pred:   {result_ov["pred_action"].shape}')
    print(f'override ctrl:   {result_ov["control_action"].shape}')

    # test boost mechanism
    print('\n--- Boost test ---')
    agent_boost = MoEAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        encoder_type='idp3', pc_dim=3, pc_out_dim=64, state_dim=A,
        num_points=N,
        num_experts=8, top_k=2, moe_hidden_dim=64, moe_num_layers=1,
        use_boost=True, boost_start_epoch=0, boost_interval=50,
        boost_experts_per_step=4, boost_topk_per_step=1,
        down_dims=[64, 128], diffusion_step_embed_dim=64,
        num_training_steps=10, num_inference_steps=3,
    ).to(device)

    moe = agent_boost.obs_encoder.moe
    assert moe.use_boost
    print(f'base experts={moe.num_experts} top_k={moe.top_k}')

    agent_boost.set_epoch(0)
    print(f'epoch 0: active_experts={moe.current_num_experts} active_top_k={moe.current_top_k}')
    assert moe.current_num_experts == 4 and moe.current_top_k == 1, \
        f"Expected 4 experts, top_k=1 at epoch 0, got {moe.current_num_experts}, {moe.current_top_k}"

    agent_boost.set_epoch(50)
    print(f'epoch 50: active_experts={moe.current_num_experts} active_top_k={moe.current_top_k}')
    assert moe.current_num_experts == 8 and moe.current_top_k == 2, \
        f"Expected 8 experts, top_k=2 at epoch 50, got {moe.current_num_experts}, {moe.current_top_k}"

    agent_boost.set_epoch(200)
    print(f'epoch 200: active_experts={moe.current_num_experts} active_top_k={moe.current_top_k}')
    assert moe.current_num_experts == 8 and moe.current_top_k == 2, \
        "Should cap at base values"

    # test enhanced gate
    print('\n--- Enhanced gate test ---')
    agent_gate = MoEAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A,
        encoder_type='idp3', pc_dim=3, pc_out_dim=64, state_dim=A,
        num_points=N,
        num_experts=4, top_k=2, moe_hidden_dim=64, moe_num_layers=1,
        use_enhanced_gate=True, gate_dropout=0.1,
        down_dims=[64, 128], diffusion_step_embed_dim=64,
        num_training_steps=10, num_inference_steps=3,
    ).to(device)
    gate_router = agent_gate.obs_encoder.moe.router
    assert isinstance(gate_router, nn.Sequential), \
        f"Enhanced gate should be nn.Sequential, got {type(gate_router)}"
    print(f'gate type: {type(gate_router).__name__} (len={len(gate_router)})')

    # forward through enhanced gate
    feat, aux_gate = agent_gate.obs_encoder(obs)
    print(f'enhanced gate cond: {feat.shape}, aux loss: {aux_gate["loss"].item():.4f}')
    print('=== PASSED ===')


if __name__ == '__main__':
    example()
