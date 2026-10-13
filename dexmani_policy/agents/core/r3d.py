"""R3D Agent: Uni3D + OneWayTransformer + Diffusion."""

from dexmani_policy.agents.action_decoders.backbone.one_way_transformer import (
    OneWayTransformerBackbone,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.obs_encoder.pointcloud.r3d_obs_encoder import R3DObsEncoder


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
        state_dim: int,
        pc_encoder_config: dict = None,
        state_out_dim: int = 256,
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
        prediction_type: str = "sample",
        modality_dropout_probs: dict = None,
        # EE auxiliary loss — wrist pose (pos3 + rot6d6 = 9dim)
        use_aux_ee: bool = False,
        joint_dim: int = 19,
        ee_dim: int = 9,
        # Auxiliary loss weight (applied to non-joint dim groups).
        # Forwarded to Diffusion; must match when use_aux_ee=True.
        aux_loss_weight: float = 1.0,
        clip_sample: bool = True,
    ):
        self.use_aux_ee = use_aux_ee
        self.joint_dim = joint_dim
        self.ee_dim = ee_dim

        obs_encoder = R3DObsEncoder(
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

    def initialize_training(self):
        self.obs_encoder.pc_encoder.initialize_training()

    def _get_dim_groups(self):
        if not self.use_aux_ee:
            return None
        return {
            "joint": (0, self.joint_dim),
            "ee": (self.joint_dim, self.joint_dim + self.ee_dim),
        }

    @property
    def control_action_dim(self):
        return self.joint_dim if self.use_aux_ee else self.action_dim
