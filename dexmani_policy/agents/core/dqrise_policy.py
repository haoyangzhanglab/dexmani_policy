"""DQRISEPolicy — quantised hand state + TCP/index diffusion."""

from __future__ import annotations

from typing import Any

import torch

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.core.base_policy import BasePolicy
from dexmani_policy.agents.core.dp3_policy import DP3ObservationEncoder
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook


class DQRISEPolicy(BasePolicy):
    """DQ-RISE policy with a self-contained runtime hand codebook."""

    def set_normalization_spec(
        self, spec: dict[str, str], feature_dims: dict[str, int] | None = None
    ) -> None:
        from dexmani_policy.utils.config import validate_dq_normalization

        validate_dq_normalization(spec)
        super().set_normalization_spec(spec, feature_dims)

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
        clip_sample: bool = True,
    ) -> None:
        if tcp_dim <= 0 or tcp_dim >= action_dim:
            raise ValueError(f"Expected 0 < tcp_dim < action_dim, got {tcp_dim}, {action_dim}")

        hand_dim = action_dim - tcp_dim
        diffusion_action_dim = tcp_dim + 1

        # Attach the codebook module only after BasePolicy initialises nn.Module.
        hand_codebook = HandStateCodebook(
            hand_dim=hand_dim,
            num_groups=codebook_num_groups,
            codebook_size=codebook_size,
        )

        obs_encoder = DP3ObservationEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            fps_random_config=fps_random_config,
        )
        backbone = ConditionalUNet1D(
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
            clip_sample=clip_sample,
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

        self.tcp_dim = tcp_dim
        self.hand_dim = hand_dim
        self.diffusion_action_dim = diffusion_action_dim
        self.hand_codebook = hand_codebook
        self.codebook_path = codebook_path
        self.register_load_state_dict_post_hook(self.validate_restored_codebook)

    def initialize_training(self) -> None:
        if self.codebook_path is None:
            raise ValueError("Fresh DQ-RISE training requires codebook_path")
        self.hand_codebook.load(self.codebook_path)
        self.hand_data = self.hand_codebook.artifact_metadata["hand_data"]
        self.validate_codebook_normalizer()

    def validate_restored_codebook(self, module: torch.nn.Module, incompatible_keys) -> None:
        self.validate_codebook_normalizer()

    def load_normalizer_from_dataset(self, normalizer) -> None:
        super().load_normalizer_from_dataset(normalizer)
        self.validate_codebook_normalizer()

    def validate_codebook_normalizer(self, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
        if not self.hand_codebook.is_loaded:
            return
        if "action" not in self.normalizer.params_dict:
            return

        params = self.normalizer["action"].params_dict
        policy_scale = params["scale"][-self.hand_dim :].detach().cpu()
        policy_offset = params["offset"][-self.hand_dim :].detach().cpu()
        code_scale = self.hand_codebook.hand_normalizer_scale.detach().cpu()
        code_offset = self.hand_codebook.hand_normalizer_offset.detach().cpu()

        if not (
            torch.allclose(policy_scale, code_scale, rtol=rtol, atol=atol)
            and torch.allclose(policy_offset, code_offset, rtol=rtol, atol=atol)
        ):
            raise ValueError(
                "The policy action normalizer and VQ codebook hand normalizer "
                "do not match. Training/inference would use incompatible hand "
                "coordinates. Rebuild the codebook with "
                "python -m dexmani_policy.training.pretrain.hand_prior.vqvae "
                "--policy-config using this Policy's data recipe and normalization."
            )

    def require_codebook(self) -> None:
        if not self.hand_codebook.is_loaded:
            raise RuntimeError(
                "DQRISEPolicy has no hand codebook. Supply codebook_path when "
                "initializing fresh training, or load a complete "
                "policy checkpoint before training/inference."
            )

    def compute_loss(self, batch: dict[str, Any], **kwargs) -> tuple[torch.Tensor, dict]:
        self.validate_batch(batch)
        self.require_codebook()
        cond = self.encode_observation(batch["obs"])

        normed = self.normalizer["action"].normalize(batch["action"])
        if not torch.isfinite(normed).all():
            raise ValueError(
                "NaN or Inf detected in normalised action. "
                "Check the data pipeline (Zarr integrity, normalizer fit range)."
            )
        tcp_part = normed[..., : self.tcp_dim]
        hand_part = normed[..., self.tcp_dim :]

        index, discrete_idx = self.hand_codebook.encode(hand_part)
        index = index.to(tcp_part.dtype)

        joint_action = torch.cat([tcp_part, index], dim=-1)
        action_loss, loss_dict = self.action_decoder.compute_loss(cond, joint_action, **kwargs)

        num_codes = self.hand_codebook.num_codes
        counts = torch.bincount(discrete_idx.flatten(), minlength=num_codes).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        loss_dict["batch_nn_code_entropy"] = entropy.detach()
        loss_dict["batch_nn_code_used_1pct"] = (probabilities > 0.01).sum().detach()

        return action_loss, loss_dict

    @torch.no_grad()
    def predict_action_from_cond(
        self, cond: torch.Tensor, inference_steps: int | None = None
    ) -> dict:
        self.require_codebook()
        batch_size = cond.shape[0]
        template = torch.zeros(
            batch_size,
            self.horizon,
            self.diffusion_action_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        reduced_action = self.action_decoder.predict_action(
            cond, template, inference_steps=inference_steps
        )

        tcp_pred = reduced_action[..., : self.tcp_dim]
        idx_pred = reduced_action[..., -1]
        hand_pred, discrete_idx = self.hand_codebook.decode(idx_pred)
        hand_pred = hand_pred.to(tcp_pred.dtype)

        normalized_full_action = torch.cat([tcp_pred, hand_pred], dim=-1)
        pred = self.normalizer["action"].unnormalize(normalized_full_action)
        start = self.n_obs_steps - 1
        control_action = pred[:, start : start + self.n_action_steps]

        return {
            "pred_action": pred,
            "control_action": control_action,
            "pred_code_index": discrete_idx,
            "pred_code_continuous": idx_pred,
        }

    def __repr__(self) -> str:
        return (
            f"DQRISEPolicy(action_dim={self.action_dim}, tcp_dim={self.tcp_dim}, "
            f"hand_dim={self.hand_dim}, "
            f"diffusion_action_dim={self.diffusion_action_dim}, "
            f"codes={self.hand_codebook.num_codes}, "
            f"codebook_loaded={self.hand_codebook.is_loaded})"
        )
