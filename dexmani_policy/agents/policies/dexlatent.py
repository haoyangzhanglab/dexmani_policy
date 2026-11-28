from __future__ import annotations

from typing import Literal

import torch

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder
from dexmani_policy.agents.policies.base import Policy


class DexLatent(Policy):
    required_identity_fields = frozenset({"point_cloud", "camera_intrinsic", "camera_extrinsic"})
    supported_action_keys = ("action",)
    arm_dim = 7
    decoder_action_dim = 39

    @classmethod
    def build_for_training(cls, cfg, dataset, field_dims, *, checkpoint=None, resuming=False):
        from dexmani_policy.training.data import iter_validated_actions
        from dexmani_policy.training.initialization import create_policy, initialize_policy

        if checkpoint is not None:
            if not resuming:
                for _ in iter_validated_actions(dataset):
                    pass
            return super().build_for_training(cfg, dataset, field_dims, checkpoint=checkpoint, resuming=resuming)

        model = create_policy(cfg, dataset)
        source = cfg.initialization.dexlatent
        hand_affine = model.hand_prior.initialize_weights(source)
        initialize_policy(model, cfg, dataset, field_dims, hand_affine=hand_affine)
        # Latent limits depend on the hand affine inherited from its prior.
        latents = (model._encode_action_parts(chunk)[1] for chunk in dataset.iter_normalization_data("action"))
        model.normalizer.fit_field("hand_latent", latents)
        return model

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        pc_dim: int = 3,
        num_points: int = 1024,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
        n_layers: int = 8,
        hidden_dim: int = 256,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.0,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 10,
        time_shift_alpha: float = 1.0,
    ) -> None:
        if action_dim != 19:
            raise ValueError("DexLatent requires 19D joint actions/state")

        obs_encoder = RGBPointObservationEncoder(
            pc_dim=pc_dim,
            state_dim=19,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            rgb_backbone_config=rgb_backbone_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_semantic_sampling=rgb_semantic_sampling,
        )

        backbone = DiTX(
            horizon=horizon,
            action_dim=self.decoder_action_dim,
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
            time_shift_alpha=time_shift_alpha,
        )

        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
        )
        self.hand_prior = DexLatentHandPrior()

    def configure_normalizer(self, feature_dims, feature_shapes) -> None:
        super().configure_normalizer(
            feature_dims, {**feature_shapes, "hand_latent": (self.hand_prior.hand_mean_head.out_features,)},
        )

    @torch.no_grad()
    def _encode_action_parts(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalizer["action"].normalize(actions)
        return normalized[..., : self.arm_dim], self.hand_prior.encode(normalized[..., self.arm_dim :])

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        arm, latent = self._encode_action_parts(actions)
        return torch.cat((arm, self.normalizer["hand_latent"].normalize(latent)), dim=-1)

    def _decode_actions(self, prediction: torch.Tensor) -> torch.Tensor:
        arm = prediction[..., : self.arm_dim]
        latent = self.normalizer["hand_latent"].unnormalize(prediction[..., self.arm_dim :])
        hand = self.hand_prior.decode(latent)
        return self.normalizer["action"].unnormalize(torch.cat((arm, hand), dim=-1))
