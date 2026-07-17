"""DQRISEAgent — VQ-VAE hand quantisation + joint diffusion."""

from typing import Any, Dict

from pathlib import Path

import torch

from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.core.dp3 import DP3ObsEncoder
from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager


class DQRISEAgent(BaseAgent):
    """VQ-VAE hand quantisation + joint diffusion policy."""

    def __init__(
        self,
        # action space
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        tcp_dim: int = 9,
        codebook_path: str | None = None,
        codebook_num_groups: int = 2,
        codebook_size: int = 4,
        # obs encoder
        encoder_type: str = 'idp3',
        pc_dim: int = 6,
        pc_out_dim: int = 128,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        fps_random_config: dict | None = None,
        # diffusion backbone
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = 'sample',
        cond_predict_scale: bool = True,
        modality_dropout_probs: dict | None = None,
    ):
        # derive dimensions
        self.tcp_dim = tcp_dim
        self.hand_dim = action_dim - tcp_dim
        self.codebook_num_groups = codebook_num_groups
        self.diffusion_action_dim = tcp_dim + 1

        # codebook
        self.codebook_manager = CodebookManager(
            hand_dim=self.hand_dim,
            num_groups=codebook_num_groups,
            codebook_size=codebook_size,
        )
        if codebook_path is not None:
            self.codebook_manager.load(codebook_path)

        # obs encoder (mirrors DP3Agent)
        obs_encoder = DP3ObsEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            fps_random_config=fps_random_config,
        )

        # diffusion backbone (reduced action dim: tcp_dim+1 instead of action_dim)
        backbone = ConditionalUnet1D(
            input_dim=self.diffusion_action_dim,
            context_dim=obs_encoder.out_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        action_decoder = Diffusion(
            backbone,
            num_training_steps,
            num_inference_steps,
            prediction_type,
        )

        # BaseAgent init (action_dim stays 19 for normalizer)
        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

    def compute_loss(self, batch: Dict[str, Any], **kwargs):
        cond, aux = self._build_cond(batch['obs'])

        # normalise full action
        normed = self.normalizer['action'].normalize(batch['action'])  # (B, H, A)

        # split
        tcp_part = normed[..., :self.tcp_dim]                          # (B, H, tcp_dim)
        hand_part = normed[..., self.tcp_dim:]                          # (B, H, hand_dim)

        # VQ-encode hand → continuous index
        B, H, _ = hand_part.shape
        hand_flat = hand_part.reshape(B * H, self.hand_dim).float()  # (B*H, hand_dim) — force float32
        idx_flat = self.codebook_manager.hand_pose_to_continuous_index(hand_flat)
        index = idx_flat.reshape(B, H, 1)                               # (B, H, 1)
        index = index.to(dtype=tcp_part.dtype)

        # VQ index usage monitoring
        num_codes = self.codebook_manager.get_num_codes()
        discrete_idx = ((index + 1.0) / 2.0 * (num_codes - 1)).round().long().clamp(0, num_codes - 1)
        idx_hist = torch.bincount(discrete_idx.flatten(), minlength=num_codes).float()
        idx_hist = idx_hist / idx_hist.sum()
        idx_entropy = -(idx_hist * (idx_hist + 1e-12).log()).sum()
        idx_used = (idx_hist > 0.01).sum().item()

        # joint action for diffusion
        joint_action = torch.cat([tcp_part, index], dim=-1)             # (B, H, tcp_dim+1)

        action_loss, loss_dict = self.action_decoder.compute_loss(cond, joint_action)
        loss_dict['vq_idx_entropy'] = idx_entropy.item()
        loss_dict['vq_idx_used'] = idx_used

        # merge auxiliary losses (e.g. MoE)
        aux_loss = aux.get('loss')
        if aux_loss is not None:
            loss_dict['loss'] = action_loss + aux_loss
            loss_dict['loss_action'] = loss_dict.get('loss_action', action_loss)
            return action_loss + aux_loss, loss_dict

        return action_loss, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict, denoise_timesteps=None) -> Dict:
        cond, _ = self._build_cond(obs_dict)
        return self.predict_action_from_cond(cond, denoise_timesteps)

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None) -> Dict:
        B = cond.shape[0]

        # DDIM sampling in reduced action space
        template = torch.zeros(
            B, self.horizon, self.diffusion_action_dim,
            device=cond.device, dtype=cond.dtype,
        )
        pred_diffusion = self.action_decoder.predict_action(
            cond, template, denoise_timesteps,
        )                                                          # (B, horizon, tcp_dim+1)

        # split
        tcp_pred = pred_diffusion[..., :self.tcp_dim]              # (B, horizon, tcp_dim)
        idx_pred = pred_diffusion[..., -1]                         # (B, horizon)

        # VQ index → hand pose via codebook lookup
        idx_flat = idx_pred.reshape(B * self.horizon)              # (B*horizon,)
        hand_flat, _ = self.codebook_manager.continuous_index_to_hand_pose(idx_flat)
        hand_pred = hand_flat.reshape(B, self.horizon, self.hand_dim)  # (B, horizon, hand_dim)

        # Ensure dtype consistency: sorted_hand_poses is float32, but the
        # diffusion output may be bfloat16 under autocast.
        hand_pred = hand_pred.to(dtype=tcp_pred.dtype)

        # assemble full action in normalised space
        full_action = torch.cat([tcp_pred, hand_pred], dim=-1)     # (B, horizon, action_dim)

        # unnormalise
        pred = self.normalizer['action'].unnormalize(full_action)  # (B, horizon, action_dim)

        # extract control window
        start = self.n_obs_steps - 1
        control_action = pred[:, start:start + self.n_action_steps]

        return {
            'pred_action': pred,
            'control_action': control_action,
        }

    # Misc

    def __repr__(self) -> str:
        return (
            f'DQRISEAgent(action_dim={self.action_dim}, '
            f'tcp_dim={self.tcp_dim}, hand_dim={self.hand_dim}, '
            f'diffusion_action_dim={self.diffusion_action_dim}, '
            f'codes={self.codebook_manager.get_num_codes()})'
        )


def example():
    import tempfile
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, H, A, N = 2, 2, 16, 19, 256
    hand_dim = 10  # A - tcp_dim

    # Build a dummy codebook (4 codes, raw space) for the example
    codebook = np.random.uniform(0, 65535, (4, hand_dim)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f, format_version=2, sorted_hand_poses=codebook,
                 hand_dim=hand_dim, num_groups=1, codebook_size=4,
                 hand_min=0.0, hand_max=65535.0)
        codebook_path = f.name

    try:
        agent = DQRISEAgent(
            horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A, tcp_dim=9,
            codebook_path=codebook_path, codebook_num_groups=1,
            codebook_size=4,
            encoder_type='idp3', pc_dim=3, pc_out_dim=64, state_dim=A,
            num_points=N,
            down_dims=(64, 128), diffusion_step_embed_dim=64,
            num_training_steps=10, num_inference_steps=3,
        ).to(device)

        obs = {
            'point_cloud': torch.randn(B * T, N, 3, device=device),
            'joint_state': torch.randn(B * T, A, device=device),
        }
        action = torch.randn(B, H, A, device=device)

        print('=== DQRISEAgent smoke test ===')
        print(f'obs point_cloud: {obs["point_cloud"].shape}')
        print(f'obs joint_state: {obs["joint_state"].shape}')
        print(f'action:          {action.shape}')

        cond, _ = agent.obs_encoder(obs)
        print(f'cond:            {cond.shape}')

        from dexmani_policy.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.fit(
            {'action': action,
             'joint_state': obs['joint_state'].reshape(B, T, A)},
            mode='limits',
        )
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
    finally:
        Path(codebook_path).unlink(missing_ok=True)


if __name__ == '__main__':
    example()
