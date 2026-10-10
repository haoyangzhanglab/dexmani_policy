"""Hand–object interaction tokens with standard flow matching and eight DiT-X blocks."""

from __future__ import annotations

from dexmani_policy.agents.action_decoders.backbone.ditx import DiTX
from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.obs_encoder.interaction.encoder import InteractionObsEncoder


class InteractionFlowAgent(BaseAgent):
    """Keep the action recipe fixed while testing interaction representations.

    Geometry enters in one common metric frame. ``wrist_pose_key='split'`` uses
    the simulation fields ``eef_pos`` and ``eef_rot6d``; ``'eef_pose'`` uses the
    measured XYZ + rotation-6D pose. Actions retain BaseAgent's ``[B,H,Da]``
    layout, normalization and execution-window semantics.
    """

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        state_dim: int = 19,
        pc_dim: int = 6,
        token_dim: int = 192,
        num_patches: int = 128,
        group_size: int = 32,
        patch_depth: int = 4,
        num_scene_tokens: int = 16,
        tactile_input_key: str = "contact_force",
        tactile_fusion: str = "query",
        edge_frame: str = "wrist",
        wrist_pose_key: str = "eef_pose",
        fps_random_config: dict | None = None,
        tactile_dropout_prob: float = 0.0,
        use_tactile_valid: bool = False,
        use_modality_gate: bool = True,
        # 标准 flow matching，固定 8 层 DiT-X。
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        timestep_embed_dim: int = 128,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        num_inference_steps: int = 10,
        t_sample_mode: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        time_shift_alpha: float = 1.0,
        num_flow_train_timesteps: int = 10,
        modality_dropout_probs: dict | None = None,
    ) -> None:
        if type(n_layers) is not int or n_layers != 8:
            raise ValueError(
                "InteractionFlowAgent fixes the action backbone to 8 DiT-X layers"
            )
        if wrist_pose_key not in {"eef_pose", "split"}:
            raise ValueError("wrist_pose_key must be 'eef_pose' or 'split'")
        wrist_fields = (
            ("eef_pos", "eef_rot6d") if wrist_pose_key == "split" else ("eef_pose",)
        )
        metric_fields = ("point_cloud", "fingertip_points", *wrist_fields)
        protected_fields = {
            *metric_fields,
            "contact_force",
            tactile_input_key,
            "tactile_valid",
        }
        for field, probability in (modality_dropout_probs or {}).items():
            if field in protected_fields and probability != 0:
                raise ValueError(
                    f"modality_dropout_probs.{field} would destroy geometric or missing-sensor "
                    "semantics; use tactile_dropout_prob for masked tactile dropout"
                )

        obs_encoder = InteractionObsEncoder(
            n_obs_steps=n_obs_steps,
            state_dim=state_dim,
            pc_dim=pc_dim,
            token_dim=token_dim,
            num_patches=num_patches,
            group_size=group_size,
            patch_depth=patch_depth,
            num_scene_tokens=num_scene_tokens,
            tactile_input_key=tactile_input_key,
            tactile_fusion=tactile_fusion,
            edge_frame=edge_frame,
            wrist_pose_key=wrist_pose_key,
            fps_random_config=fps_random_config,
            tactile_dropout_prob=tactile_dropout_prob,
            use_tactile_valid=use_tactile_valid,
            use_modality_gate=use_modality_gate,
        )
        backbone = DiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=token_dim,
            timestep_embed_dim=timestep_embed_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
            t_sample_mode=t_sample_mode,
            beta_s=beta_s,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            time_shift_alpha=time_shift_alpha,
            num_flow_train_timesteps=num_flow_train_timesteps,
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
        self.metric_observation_fields = metric_fields
        self.use_tactile_valid = use_tactile_valid

    def set_normalization_spec(self, spec):
        """Reject affine normalization that would change distances and wrist axes."""
        for field in self.metric_observation_fields:
            if spec.get(field) != "identity":
                raise ValueError(
                    f"Interaction geometry requires normalization.{field}: identity; "
                    "XYZ must remain in meters in the same frame and rotation-6D unscaled"
                )
        if self.use_tactile_valid and spec.get("tactile_valid") != "identity":
            raise ValueError(
                "tactile_valid is a boolean mask and requires identity normalization"
            )
        super().set_normalization_spec(spec)
