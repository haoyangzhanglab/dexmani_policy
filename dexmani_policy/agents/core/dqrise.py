"""DQRISEAgent — quantised hand state + joint arm/index diffusion."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import torch

from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.core.dp3 import DP3ObsEncoder
from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager


class DQRISEAgent(BaseAgent):
    """DQ-RISE policy with a self-contained runtime hand codebook."""

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        tcp_dim: int = 9,
        codebook_path: str | None = None,
        codebook_num_groups: int = 2,
        codebook_size: int = 4,
        encoder_type: str = "idp3",
        pc_dim: int = 6,
        pc_out_dim: int = 128,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        fps_random_config: dict | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        cond_predict_scale: bool = True,
        modality_dropout_probs: dict | None = None,
    ) -> None:
        if tcp_dim <= 0 or tcp_dim >= action_dim:
            raise ValueError(
                f"Expected 0 < tcp_dim < action_dim, got {tcp_dim}, {action_dim}"
            )

        hand_dim = action_dim - tcp_dim
        diffusion_action_dim = tcp_dim + 1

        # Construct locally first.  CodebookManager is now an nn.Module, so it
        # must be attached only after BaseAgent/nn.Module initialisation.
        codebook_manager = CodebookManager(
            hand_dim=hand_dim,
            num_groups=codebook_num_groups,
            codebook_size=codebook_size,
        )
        if codebook_path is not None:
            codebook_manager.load(codebook_path)

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
        backbone = ConditionalUnet1D(
            input_dim=diffusion_action_dim,
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
        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

        self.tcp_dim = int(tcp_dim)
        self.hand_dim = int(hand_dim)
        self.codebook_num_groups = int(codebook_num_groups)
        self.codebook_size = int(codebook_size)
        self.diffusion_action_dim = int(diffusion_action_dim)
        self.codebook_manager = codebook_manager
        self._normalizer_checked = False
        self._missing_codebook_normalizer_warned = False

    # ------------------------------------------------------------------
    # Normalizer/codebook consistency
    # ------------------------------------------------------------------

    def load_normalizer_from_dataset(self, normalizer):
        super().load_normalizer_from_dataset(normalizer)
        self._validate_codebook_normalizer()

    def _validate_codebook_normalizer(
        self, *, rtol: float = 1e-5, atol: float = 1e-6
    ) -> None:
        if not self.codebook_manager.is_loaded:
            return
        if "action" not in self.normalizer.params_dict:
            return
        if not self.codebook_manager.has_hand_normalizer:
            if not self._missing_codebook_normalizer_warned:
                warnings.warn(
                    "The codebook does not contain hand-normalizer metadata. "
                    "Runtime can continue, but coordinate consistency cannot be "
                    "verified. Re-extract the codebook with the fixed extractor.",
                    RuntimeWarning,
                )
                self._missing_codebook_normalizer_warned = True
            return

        params = self.normalizer["action"].params_dict
        policy_scale = params["scale"][-self.hand_dim :].detach().cpu()
        policy_offset = params["offset"][-self.hand_dim :].detach().cpu()
        code_scale = self.codebook_manager.hand_normalizer_scale.detach().cpu()
        code_offset = self.codebook_manager.hand_normalizer_offset.detach().cpu()

        try:
            torch.testing.assert_close(
                policy_scale, code_scale, rtol=rtol, atol=atol
            )
            torch.testing.assert_close(
                policy_offset, code_offset, rtol=rtol, atol=atol
            )
        except AssertionError as exc:
            raise ValueError(
                "The policy action normalizer and VQ codebook hand normalizer "
                "do not match. Training/inference would use incompatible hand "
                "coordinates. Rebuild both from the same episode subset and "
                "dataset."
            ) from exc
        self._normalizer_checked = True

    def _require_codebook(self) -> None:
        if not self.codebook_manager.is_loaded:
            raise RuntimeError(
                "DQRISEAgent has no hand codebook. Supply codebook_path when "
                "constructing a fresh model, or load a new self-contained "
                "policy checkpoint before training/inference."
            )
        if not self._normalizer_checked and self.normalizer.is_fitted(["action"]):
            self._validate_codebook_normalizer()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict[str, Any], **kwargs):
        self._require_codebook()
        cond, aux = self._build_cond(batch["obs"])

        normed = self.normalizer["action"].normalize(batch["action"])
        if not torch.isfinite(normed).all():
            raise ValueError(
                "NaN or Inf detected in normalised action. "
                "Check the data pipeline (Zarr integrity, normalizer fit range)."
            )
        tcp_part = normed[..., : self.tcp_dim]
        hand_part = normed[..., self.tcp_dim :]

        batch_size, horizon, _ = hand_part.shape
        hand_flat = hand_part.reshape(-1, self.hand_dim).float()
        index = self.codebook_manager.hand_pose_to_continuous_index(hand_flat)
        index = index.reshape(batch_size, horizon, 1).to(tcp_part.dtype)

        joint_action = torch.cat([tcp_part, index], dim=-1)
        action_loss, loss_dict = self.action_decoder.compute_loss(
            cond, joint_action, **kwargs
        )

        # Explicitly mark this as a mini-batch nearest-prototype statistic.
        num_codes = self.codebook_manager.get_num_codes()
        safe_max = max(num_codes - 1, 1)
        discrete_idx = torch.floor(
            ((index + 1.0) * 0.5 * safe_max).clamp(0, max(num_codes - 1, 0))
            + 0.5
        ).long()
        counts = torch.bincount(discrete_idx.flatten(), minlength=num_codes).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        loss_dict["batch_nn_code_entropy"] = entropy.detach().item()
        loss_dict["batch_nn_code_used_1pct"] = int(
            (probabilities > 0.01).sum().item()
        )

        aux_loss = aux.get("loss")
        if aux_loss is not None:
            total = action_loss + aux_loss
            loss_dict["loss"] = total
            loss_dict["loss_action"] = loss_dict.get("loss_action", action_loss)
            return total, loss_dict
        return action_loss, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict, denoise_timesteps=None) -> Dict:
        self._require_codebook()
        cond, _ = self._build_cond(obs_dict)
        return self.predict_action_from_cond(cond, denoise_timesteps)

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None) -> Dict:
        self._require_codebook()
        batch_size = cond.shape[0]
        template = torch.zeros(
            batch_size,
            self.horizon,
            self.diffusion_action_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        reduced_action = self.action_decoder.predict_action(
            cond, template, denoise_timesteps
        )

        tcp_pred = reduced_action[..., : self.tcp_dim]
        idx_pred = reduced_action[..., -1]
        hand_flat, discrete_idx = (
            self.codebook_manager.continuous_index_to_hand_pose(
                idx_pred.reshape(-1)
            )
        )
        hand_pred = hand_flat.reshape(
            batch_size, self.horizon, self.hand_dim
        ).to(tcp_pred.dtype)

        normalized_full_action = torch.cat([tcp_pred, hand_pred], dim=-1)
        pred = self.normalizer["action"].unnormalize(normalized_full_action)
        start = self.n_obs_steps - 1
        control_action = pred[:, start : start + self.n_action_steps]

        return {
            "pred_action": pred,
            "control_action": control_action,
            "pred_code_index": discrete_idx.reshape(batch_size, self.horizon),
            "pred_code_continuous": idx_pred,
        }

    def __repr__(self) -> str:
        return (
            f"DQRISEAgent(action_dim={self.action_dim}, tcp_dim={self.tcp_dim}, "
            f"hand_dim={self.hand_dim}, "
            f"diffusion_action_dim={self.diffusion_action_dim}, "
            f"codes={self.codebook_manager.get_num_codes()}, "
            f"codebook_loaded={self.codebook_manager.is_loaded})"
        )


def example() -> None:
    """Minimal construction example; project dependencies are still required."""
    import tempfile
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, obs_steps, horizon, action_dim, points = 2, 2, 16, 19, 256
    hand_dim = action_dim - 7
    codebook = np.random.uniform(0, 65535, (16, hand_dim)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as file:
        np.savez(
            file,
            format_version=3,
            pose_space="affine_raw",
            sorted_hand_poses=codebook,
            pca_permutation=np.arange(16),
            hand_dim=hand_dim,
            num_groups=2,
            codebook_size=4,
            hand_min=0.0,
            hand_max=65535.0,
            metadata_json="{}",
        )
        path = file.name

    try:
        agent = DQRISEAgent(
            horizon=horizon,
            n_obs_steps=obs_steps,
            n_action_steps=8,
            action_dim=action_dim,
            tcp_dim=7,
            codebook_path=path,
            encoder_type="idp3",
            pc_dim=3,
            pc_out_dim=64,
            state_dim=action_dim,
            num_points=points,
            down_dims=(64, 128),
            diffusion_step_embed_dim=64,
            num_training_steps=10,
            num_inference_steps=3,
        ).to(device)
        print(agent)
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    example()
