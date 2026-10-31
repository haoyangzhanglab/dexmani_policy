"""DQ-RISE: flow matching over arm joints and quantized hand-state indices."""

from __future__ import annotations

from typing import Any

import torch

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.artifacts import load_codebook
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, validate_hand_coordinates
from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
from dexmani_policy.agents.policies.base import BasePolicy


class DQRISEPolicy(BasePolicy):
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        tcp_dim: int = 7,
        codebook_num_groups: int = 2,
        codebook_size: int = 4,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        fusion_stage: str = "post_global",
        semantic_pool_sigma: float | None = None,
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.0,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 10,
        modality_dropout_probs: dict | None = None,
    ) -> None:
        if (n_obs_steps, action_dim, tcp_dim) != (2, 19, 7):
            raise ValueError("DQ-RISE requires n_obs_steps=2 and joint action=7+12")
        hand_dim = action_dim - tcp_dim
        flow_action_dim = tcp_dim + 1
        obs_encoder = SpatialObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            rgb_backbone_config=rgb_backbone_config,
            fusion_stage=fusion_stage,
            semantic_pool_sigma=semantic_pool_sigma,
        )
        backbone = DiTX(
            horizon=horizon,
            action_dim=flow_action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            timestep_embed_dim=timestep_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            p_drop_attn=p_drop_attn,
        )
        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
            t_sample_mode="uniform",
            time_shift_alpha=1.0,
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
        self.flow_action_dim = flow_action_dim
        self.hand_codebook = HandStateCodebook(
            hand_dim=hand_dim,
            num_groups=codebook_num_groups,
            codebook_size=codebook_size,
        )

    def initialize_training(self, data_description, codebook_path) -> dict[str, torch.Tensor]:
        if codebook_path is None:
            raise ValueError("Fresh DQ-RISE training requires initialization.codebook")
        validate_hand_coordinates(data_description["attrs"])
        metadata, hand_affine = load_codebook(codebook_path, self.hand_codebook)
        if data_description["action_key"] != "action" or metadata["hand_data"]["action_key"] != "action":
            raise ValueError("DQ-RISE requires joint policy data and a codebook pretrained with action_key=action")
        if data_description["split_manifest"] != metadata["split_manifest"]:
            raise ValueError("Pretraining/policy split mismatch")
        return hand_affine

    def compute_loss(
        self, batch: dict[str, Any], *, log_metrics: bool = True, **kwargs: Any,
    ) -> tuple[torch.Tensor, dict]:
        condition = self.encode_observation(batch["obs"])

        normed = self.normalizer["action"].normalize(batch["action"])
        arm_part = normed[..., : self.tcp_dim]
        hand_part = normed[..., self.tcp_dim :]
        index, discrete_idx = self.hand_codebook.encode(hand_part)
        index = index.to(arm_part.dtype)

        flow_target = torch.cat([arm_part, index], dim=-1)
        action_loss, loss_dict = self.action_decoder.compute_loss(
            condition, flow_target, log_metrics=log_metrics, **kwargs,
        )

        if not log_metrics:
            return action_loss, loss_dict

        with torch.no_grad():
            num_codes = self.hand_codebook.num_codes
            counts = torch.zeros(num_codes, device=discrete_idx.device, dtype=torch.float32)
            counts.scatter_add_(0, discrete_idx.flatten(), counts.new_ones(discrete_idx.numel()))
            probabilities = counts / counts.sum().clamp_min(1.0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            loss_dict["batch_nn_code_entropy"] = entropy
            loss_dict["batch_nn_code_used_1pct"] = (probabilities > 0.01).sum()

        return action_loss, loss_dict

    @torch.no_grad()
    def predict_action_from_condition(
        self, condition: torch.Tensor, inference_steps: int | None = None,
        *, generator: torch.Generator | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if rtc_prefix is not None or delay_steps or rtc_guidance_cap:
            raise NotImplementedError("DQ-RISE does not support RTC")
        reduced_action = self.action_decoder.predict_action(
            condition, (condition.shape[0], self.horizon, self.flow_action_dim),
            inference_steps=inference_steps, generator=generator,
        )

        arm_pred = reduced_action[..., : self.tcp_dim]
        idx_pred = reduced_action[..., -1]
        hand_pred, discrete_idx = self.hand_codebook.decode(idx_pred)
        hand_pred = hand_pred.to(arm_pred.dtype)

        normalized_full_action = torch.cat([arm_pred, hand_pred], dim=-1)
        prediction = self.normalizer["action"].unnormalize(normalized_full_action)
        return {
            **self.format_action_output(prediction),
            "pred_code_index": discrete_idx[:, self.action_start_index :],
            "pred_code_continuous": idx_pred[:, self.action_start_index :],
        }

    def extra_repr(self) -> str:
        return (
            f"action_dim={self.action_dim}, tcp_dim={self.tcp_dim}, "
            f"hand_dim={self.hand_dim}, "
            f"flow_action_dim={self.flow_action_dim}, "
            f"codes={self.hand_codebook.num_codes}"
        )
