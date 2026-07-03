"""R3D Agent: Uni3D + OneWayTransformer + Diffusion."""

import torch

from dexmani_policy.agents.obs_encoder.pointcloud.r3d_obs_encoder import R3DObsEncoder
from dexmani_policy.agents.action_decoders.backbone.one_way_transformer import OneWayTransformerBackbone
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.core.base import BaseAgent

class R3DAgent(BaseAgent):
    """R3D policy agent.

    Constructs R3DObsEncoder + OneWayTransformerBackbone + Diffusion.

    When ``use_aux_ee`` is enabled, predicts wrist pose (pos3+rot6d6=9dim)
    as an auxiliary target alongside the primary joint action (19dim),
    matching the official R3D-Policy ``use_target_ee`` design.
    Overrides ``_get_dim_groups`` to compute separate joint/EE loss
    components and ``control_action_dim`` to return joint-only at inference.
    """

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        pc_dim: int,
        num_points: int,
        state_dim: int,
        pc_encoder_config: dict = None,
        state_out_dim: int = 64,
        fps_random_config: dict = None,
        # Backbone params
        timestep_embed_dim: int = 128,
        embedding_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        # Diffusion params
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = 'sample',
        modality_dropout_probs: dict = None,
        # EE auxiliary loss — wrist pose (pos3 + rot6d6 = 9dim)
        use_aux_ee: bool = False,
        joint_dim: int = 19,
        ee_dim: int = 9,
        # Auxiliary loss weight (applied to non-joint dim groups)
        aux_loss_weight: float = 0.1,
    ):
        self.use_aux_ee = use_aux_ee
        self.joint_dim = joint_dim
        self.ee_dim = ee_dim

        obs_encoder = R3DObsEncoder(
            pc_dim=pc_dim,
            num_points=num_points,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            state_out_dim=state_out_dim,
            fps_random_config=fps_random_config,
        )

        backbone = OneWayTransformerBackbone(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            num_obs_tokens=obs_encoder.num_obs_tokens,
            obs_token_dim=obs_encoder.obs_token_dim,
            pc_pe_dim=obs_encoder.pc_pe_dim,
            timestep_embed_dim=timestep_embed_dim,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attention_downsample_rate=attention_downsample_rate,
            use_aux_ee=use_aux_ee,
            joint_dim=joint_dim,
            ee_dim=ee_dim if use_aux_ee else None,
        )

        action_decoder = Diffusion(
            backbone,
            num_training_steps=num_training_steps,
            num_inference_steps=num_inference_steps,
            prediction_type=prediction_type,
            aux_loss_weight=aux_loss_weight,
        )

        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

    # ------------------------------------------------------------------
    # Per-head loss groups (overrides BaseAgent._get_dim_groups)
    # ------------------------------------------------------------------

    def _get_dim_groups(self):
        if not self.use_aux_ee:
            return None
        return {
            'joint': (0, self.joint_dim),
            'ee': (self.joint_dim, self.joint_dim + self.ee_dim),
        }

    @property
    def control_action_dim(self):
        return self.joint_dim if self.use_aux_ee else self.action_dim

# Smoke test

def _run_smoke(use_aux_ee=False):
    """Run smoke test for R3DAgent with optional auxiliary loss."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, joint_A = 2, 2, 16, 19
    ee_A = 9  # wrist pose: pos3 + rot6d6
    N = 256  # small N for speed

    action_dim = joint_A + ee_A if use_aux_ee else joint_A
    tag = 'aux_ee' if use_aux_ee else 'standard'

    agent = R3DAgent(
        horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=action_dim,
        pc_dim=6, num_points=N, state_dim=19,
        pc_encoder_config={
            'pc_model': 'eva02_tiny_patch14_224',
            'embed_dim': 128,
            'num_group': 64,
            'group_size': 32,
            'pc_in_channels': 6,
            'use_pretrained_weights': False,
        },
        state_out_dim=32,
        timestep_embed_dim=64,
        embedding_dim=128,
        depth=2, num_heads=4, mlp_dim=512,
        num_training_steps=10,
        num_inference_steps=3,
        prediction_type='sample',
        use_aux_ee=use_aux_ee,
        joint_dim=joint_A,
        ee_dim=ee_A,
    ).to(device)

    # Create sample data
    pc_data = (torch.rand(B * T, N, 6, device=device) * 2 - 1).float()
    state_data = torch.randn(B * T, 19, device=device)
    action = torch.randn(B, H, action_dim, device=device)

    print(f'=== R3DAgent smoke test {tag} ===')
    print(f'  action_dim:   {action_dim}')
    print(f'  obs pc:       {pc_data.shape}')
    print(f'  obs state:    {state_data.shape}')
    print(f'  action:       {action.shape}')

    # Test encoder
    obs_flat = {'point_cloud': pc_data, 'joint_state': state_data}
    cond, aux = agent.obs_encoder(obs_flat)
    print(f'  cond (enc):   {cond.shape}  [B, T*K, D+D_s+D]')

    # Fit normalizer
    from dexmani_policy.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    normalizer.fit({
        'action': action,
        'joint_state': state_data.reshape(B, T, 19),
    }, mode='limits')
    agent.load_normalizer_from_dataset(normalizer)

    # Test compute_loss
    batch = {
        'obs': {
            'point_cloud': pc_data.reshape(B, T, N, 6),
            'joint_state': state_data.reshape(B, T, 19),
        },
        'action': action,
    }
    agent.train()
    loss, loss_dict = agent.compute_loss(batch)
    print(f'  loss:         {loss.item():.4f}  keys={list(loss_dict.keys())}')
    if use_aux_ee:
        assert 'loss_joint' in loss_dict, f'Missing loss_joint in {loss_dict.keys()}'
        assert 'loss_ee' in loss_dict, f'Missing loss_ee in {loss_dict.keys()}'

    # Test predict_action
    agent.eval()
    with torch.no_grad():
        result = agent.predict_action({
            'point_cloud': pc_data.reshape(B, T, N, 6),
            'joint_state': state_data.reshape(B, T, 19),
        })
    print(f'  pred_action:     {result["pred_action"].shape}')
    print(f'  control_action:  {result["control_action"].shape}')
    assert result["control_action"].shape == (B, 8, joint_A), \
        f'Bad control_action shape: {result["control_action"].shape}, expected (B, 8, {joint_A})'
    print('=== PASSED ===')

def example():
    """Self-contained smoke test for R3DAgent — 2 paths."""
    _run_smoke()                # standard (joint only)
    _run_smoke(use_aux_ee=True) # aux_ee (joint + wrist pose)

if __name__ == '__main__':
    example()
