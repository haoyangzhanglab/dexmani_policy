"""DQ-RISE: diffusion over arm joints and quantized hand-state indices."""

from __future__ import annotations

from typing import Any

import torch

from dexmani_policy.agents.action_decoders.backbones.unet1d import (
    ConditionalUNet1D,
    validate_unet_horizon,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
from dexmani_policy.agents.obs_encoder.pointcloud.global_sparse3d import (
    Sparse3DGlobalObservationEncoder,
)
from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import validate_action_clipping, validate_dq_normalization


class DQRISEPolicy(BasePolicy):
    requires_hand_pretraining = True

    @classmethod
    def validate_model_config(cls, cfg, spec) -> None:
        Sparse3DGlobalObservationEncoder.validate_config(cfg, spec)
        if cfg.action_key != "action" or (cfg.agent.tcp_dim, cfg.agent.action_dim) != (7, 19):
            raise ValueError("DQ-RISE requires joint action=7+12")
        if (cfg.horizon, cfg.n_action_steps) != (16, 8):
            raise ValueError("DQ-RISE requires horizon=16, n_action_steps=8")
        if cfg.env_runner.env_kwargs.control_mode != "joint":
            raise ValueError("DQ-RISE requires joint control")
        if cfg.get("use_aux_ee", False):
            raise ValueError("DQ-RISE does not support auxiliary action targets")
        validate_dq_normalization(spec)
        validate_action_clipping(spec, cfg.agent.get("clip_sample", True))
        validate_unet_horizon(cfg.horizon, cfg.agent.get("down_dims", (256, 512)))

    def set_normalization_spec(
        self, spec: dict[str, str], feature_dims: dict[str, int]
    ) -> None:
        validate_dq_normalization(spec)
        super().set_normalization_spec(spec, feature_dims)

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        tcp_dim: int = 7,
        codebook_path: str | None = None,
        codebook_num_groups: int = 2,
        codebook_size: int = 4,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 20,
        prediction_type: str = "epsilon",
        cond_predict_scale: bool = True,
        modality_dropout_probs: dict | None = None,
        clip_sample: bool = True,
    ) -> None:
        if (horizon, n_action_steps, action_dim, tcp_dim) != (16, 8, 19, 7):
            raise ValueError("DQ-RISE requires horizon=16, n_action_steps=8 and joint action=7+12")

        hand_dim = action_dim - tcp_dim
        diffusion_action_dim = tcp_dim + 1

        obs_encoder = Sparse3DGlobalObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        backbone = ConditionalUNet1D(
            input_dim=diffusion_action_dim,
            context_dim=obs_encoder.out_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
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
        self.hand_codebook = HandStateCodebook(
            hand_dim=hand_dim,
            num_groups=codebook_num_groups,
            codebook_size=codebook_size,
        )
        self.codebook_path = codebook_path
        self.register_load_state_dict_post_hook(self.validate_restored_codebook)

    def initialize_training(self) -> None:
        if self.codebook_path is None:
            raise ValueError("Fresh DQ-RISE training requires codebook_path")
        self.hand_codebook.load(self.codebook_path)
        self.hand_data = self.hand_codebook.artifact_metadata["hand_data"]
        if self.hand_data.get("action_key") != "action":
            raise ValueError("DQ-RISE requires a Codebook pretrained with action_key=action")
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

    def compute_loss(
        self, batch: dict[str, Any], *, log_metrics: bool = True
    ) -> tuple[torch.Tensor, dict]:
        self.validate_batch(batch)
        self.require_codebook()
        condition = self.encode_observation(batch["obs"])

        normed = self.normalizer["action"].normalize(batch["action"])
        arm_part = normed[..., : self.tcp_dim]
        hand_part = normed[..., self.tcp_dim :]
        index, discrete_idx = self.hand_codebook.encode(hand_part)
        index = index.to(arm_part.dtype)

        diffusion_target = torch.cat([arm_part, index], dim=-1)
        action_loss, loss_dict = self.action_decoder.compute_loss(
            condition, diffusion_target, log_metrics=log_metrics
        )

        if not log_metrics:
            return action_loss, loss_dict

        with torch.no_grad():
            num_codes = self.hand_codebook.num_codes
            counts = torch.bincount(discrete_idx.flatten(), minlength=num_codes).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            loss_dict["batch_nn_code_entropy"] = entropy
            loss_dict["batch_nn_code_used_1pct"] = (probabilities > 0.01).sum()

        return action_loss, loss_dict

    @torch.no_grad()
    def predict_action_from_condition(
        self, condition: torch.Tensor, inference_steps: int | None = None
    ) -> dict:
        self.require_codebook()
        template = condition.new_zeros(
            (condition.shape[0], self.horizon, self.diffusion_action_dim)
        )
        reduced_action = self.action_decoder.predict_action(
            condition, template, inference_steps=inference_steps
        )

        arm_pred = reduced_action[..., : self.tcp_dim]
        idx_pred = reduced_action[..., -1]
        hand_pred, discrete_idx = self.hand_codebook.decode(idx_pred)
        hand_pred = hand_pred.to(arm_pred.dtype)

        normalized_full_action = torch.cat([arm_pred, hand_pred], dim=-1)
        prediction = self.normalizer["action"].unnormalize(normalized_full_action)
        return {
            **self.format_action_output(prediction),
            "pred_code_index": discrete_idx,
            "pred_code_continuous": idx_pred,
        }

    def extra_repr(self) -> str:
        return (
            f"action_dim={self.action_dim}, tcp_dim={self.tcp_dim}, "
            f"hand_dim={self.hand_dim}, "
            f"diffusion_action_dim={self.diffusion_action_dim}, "
            f"codes={self.hand_codebook.num_codes}, "
            f"codebook_loaded={self.hand_codebook.is_loaded}"
        )
