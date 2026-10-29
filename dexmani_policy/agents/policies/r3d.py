"""R3D diffusion policy with Uni3D observation tokens."""

from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.r3d import R3DTransformer
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy


class R3DObservationEncoder(nn.Module):
    """Concatenate Uni3D features, broadcast state and XYZ encoding as [B, T*K, 2*D+D_s]."""

    consumed_observation_fields = ("joint_state", "point_cloud")

    def __init__(
        self,
        state_dim: int,
        n_obs_steps: int,
        pc_encoder_config: dict | None = None,
        state_out_dim: int = 256,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        if fps_random_config:
            pc_encoder_config.setdefault("fps_random_config", fps_random_config)

        self.pc_encoder = Uni3DEncoder(**pc_encoder_config)
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.n_obs_steps = n_obs_steps

        self.pc_pe_dim = self.pc_encoder.out_dim
        self.obs_token_dim = 2 * self.pc_pe_dim + self.state_mlp.out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        state = obs["joint_state"]

        # R3D clamps normalized XYZRGB; other encoders retain their own input contract.
        pc = obs["point_cloud"].float().clamp(min=-1 - 1e-6, max=1 + 1e-6)

        patch_tokens, pc_pe = self.pc_encoder(pc)

        state_emb = self.state_mlp(state)
        state_emb = state_emb.unsqueeze(1).expand(-1, patch_tokens.shape[1], -1)

        tokens = torch.cat((patch_tokens, state_emb, pc_pe), dim=-1)
        return tokens.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)


class R3DPolicy(BasePolicy):
    supported_execution_modes = ("sync", "async", "rtc")

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        pc = dict(config.get("pc_encoder_config") or {})
        pc.update(use_pretrained_weights=False, pretrained_weights_path=None)
        config.pc_encoder_config = pc

    @classmethod
    def validate_normalization_recipe(cls, spec: dict[str, str], *, action_key: str) -> None:
        if spec["point_cloud"] != "limits":
            raise ValueError("R3D fixed normalized point-cloud clamp requires limits normalization")
        super().validate_normalization_recipe(spec, action_key=action_key)

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        state_dim: int,
        pc_encoder_config: dict | None = None,
        state_out_dim: int = 256,
        fps_random_config: dict | None = None,
        timestep_embed_dim: int = 128,
        embedding_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        modality_dropout_probs: dict | None = None,
        use_aux_ee: bool = False,
        joint_dim: int = 19,
        ee_dim: int = 9,
        aux_loss_weight: float = 1.0,
        clip_sample: bool = True,
    ) -> None:
        if use_aux_ee and (joint_dim != 19 or ee_dim != 9 or action_dim != joint_dim + ee_dim):
            raise ValueError("R3D auxiliary prediction requires joint19 + ee9")
        self.use_aux_ee = use_aux_ee
        self.joint_dim = joint_dim
        self.ee_dim = ee_dim
        self.aux_loss_weight = aux_loss_weight

        obs_encoder = R3DObservationEncoder(
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            state_out_dim=state_out_dim,
            fps_random_config=fps_random_config,
        )

        backbone = R3DTransformer(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
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

    def initialize_training(self, data_description) -> None:
        self.obs_encoder.pc_encoder.initialize_training()

    def loss_groups(self) -> dict[str, tuple[int, int, float]] | None:
        if not self.use_aux_ee:
            return None
        return {
            "joint": (0, self.joint_dim, 1.0),
            "ee": (self.joint_dim, self.joint_dim + self.ee_dim, self.aux_loss_weight),
        }

    @property
    def control_action_dim(self) -> int:
        return self.joint_dim if self.use_aux_ee else self.action_dim
