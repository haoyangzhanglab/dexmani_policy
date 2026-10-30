"""Structural action tokens with flow matching and shared spatial observations."""

from __future__ import annotations

from dexmani_policy.agents.action_decoders.backbones.sat import SAT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
from dexmani_policy.agents.policies.base import BasePolicy


class SATPolicy(BasePolicy):
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
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
        p_drop_attn: float = 0.0,
        timestep_embed_dim: int = 256,
        num_inference_steps: int = 10,
        modality_dropout_probs: dict | None = None,
    ) -> None:
        if n_obs_steps != 2 or action_dim != 19:
            raise ValueError("SAT requires n_obs_steps=2 and joint action_dim=19")
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
        backbone = SAT(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            timestep_embed_dim=timestep_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
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

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        rgb = config.get("rgb_backbone_config")
        if rgb is None or rgb.get("architecture") is None:
            raise ValueError("Dense checkpoints require saved DINO architecture for offline restoration")
        rgb["load_pretrained"] = False
