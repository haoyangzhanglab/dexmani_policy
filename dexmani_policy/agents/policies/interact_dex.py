from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction.gated_tactile_fusion import GatedTactileFusion
from dexmani_policy.agents.obs_encoder.interaction.hand_history import HandHistoryEncoder
from dexmani_policy.agents.obs_encoder.interaction.hand_scene import HandSceneEncoder
from dexmani_policy.agents.obs_encoder.interaction.scene_aggregator import SceneAggregator
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import HandStateEncoder, _rotation_6d_to_matrix
from dexmani_policy.agents.policies.base import BasePolicy, parameter_groups
from dexmani_policy.common.config import positive_int, require_tactile_source, validate_future_force_offsets


class InterActDexObservationEncoder(nn.Module):
    consumed_observation_fields = (
        "joint_state", "point_cloud", "eef_pose", "fingertip_xyz", "finger_tactile_force",
    )
    obs_token_dim = 192

    def __init__(
        self,
        tactile_encoder: nn.Module,
        tactile_source: str,
        pc_dim: int,
        num_points: int,
        n_obs_steps: int,
        history_mixer: bool = True,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_obs_steps = positive_int(n_obs_steps, "n_obs_steps")
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.tactile_source = tactile_source
        self.pc_encoder = Sparse3DEncoder(
            input_channels=pc_dim, fps_random_config=fps_random_config, **(pc_encoder_config or {}),
        )
        self.hand_state = HandStateEncoder(token_dim=self.obs_token_dim)
        self.hand_scene = HandSceneEncoder(
            token_channels=self.obs_token_dim, scene_channels=self.pc_encoder.out_dim,
        )
        self.tactile_fusion = GatedTactileFusion(
            token_channels=self.obs_token_dim, tactile_encoder=tactile_encoder,
        )
        self.scene_aggregator = SceneAggregator(token_channels=self.pc_encoder.out_dim)
        self.scene_projection = nn.Linear(self.pc_encoder.out_dim, self.obs_token_dim)
        self.hand_history = HandHistoryEncoder(n_obs_steps) if history_mixer else None

    @staticmethod
    def validate_normalization(spec):
        Sparse3DEncoder.validate_normalization(spec)
        for field in ("eef_pose", "fingertip_xyz"):
            if spec[field] != "identity":
                raise ValueError(f"InterActDex requires normalization.{field}: identity")
        if spec["finger_tactile_force"] not in {"rms", "max_abs"}:
            raise ValueError("InterActDex requires finger_tactile_force rms or max_abs normalization")

    @property
    def effective_config(self) -> dict:
        return {
            "pc_encoder_config": self.pc_encoder.effective_config,
            "fps_random_config": self.pc_encoder.fps_random_config,
            "tactile_encoder": self.tactile_fusion.tactile_encoder.effective_config,
            "tactile_source": self.tactile_source,
            "n_obs_steps": self.n_obs_steps,
            "history_mixer": self.hand_history is not None,
        }

    def forward(self, obs: dict) -> torch.Tensor:
        if obs["joint_state"].shape[0] % self.n_obs_steps:
            raise ValueError("Flattened observation frames must be divisible by n_obs_steps")
        pc = preprocess_point_cloud(
            obs["point_cloud"], self.num_points, self.use_coord_only,
            self.pc_encoder.fps_random_config, training=self.training,
        )
        patches = self.pc_encoder(pc)
        scene = self.scene_projection(self.scene_aggregator(
            patches["tokens"], patches["xyz"],
        )["scene_token"])
        hand = self.hand_state(obs["joint_state"], obs["eef_pose"], obs["fingertip_xyz"])
        relation = self.hand_scene(
            pc, patches, hand["anchor_xyz"], hand["hand_tokens"],
            rotation_world_from_eef=hand["rotation_world_from_eef"],
            query_context=scene.mean(dim=1),
        )
        tactile = obs["finger_tactile_force"]
        if self.tactile_source == "sim":
            # The scalar RMS/MaxAbs affine commutes with this world-to-EEF rotation.
            with torch.autocast(device_type=tactile.device.type, enabled=False):
                tactile = tactile.float() @ hand["rotation_world_from_eef"].float()
        # Fusion already includes both the hand and scene residuals.
        interaction = self.tactile_fusion(
            relation, tactile=tactile,
        )["interaction_tokens"]
        scene = scene.unflatten(0, (-1, self.n_obs_steps))
        interaction = interaction.unflatten(0, (-1, self.n_obs_steps))
        if self.hand_history is not None:
            interaction = self.hand_history(interaction)
        # DiTX assigns one frame position per 18 tokens; Flow integrates in this dtype.
        return torch.cat((interaction, scene), dim=2).flatten(1, 2).float()


class InterActDexPolicy(BasePolicy):
    supported_action_keys = ("action",)

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        tactile_source: str,
        tactile_encoder: nn.Module,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        history_mixer: bool = True,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        timestep_embed_dim: int = 128,
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        num_inference_steps: int = 10,
        future_force_loss_weight: float = 0.0,
        future_force_offsets: tuple[int, ...] = (1, 4, 8, 13),
    ) -> None:
        future_force_loss_weight = float(future_force_loss_weight)
        if not math.isfinite(future_force_loss_weight) or future_force_loss_weight < 0:
            raise ValueError("future_force_loss_weight must be finite and non-negative")
        future_force_offsets = validate_future_force_offsets(future_force_offsets)
        if (action_dim, state_dim) != (19, 19):
            raise ValueError("InterActDex requires 19D joint state and actions")
        require_tactile_source(("finger_tactile_force",), tactile_source)
        if tactile_encoder.input_key != "finger_tactile_force":
            raise ValueError("InterActDex requires aggregate finger_tactile_force")
        if (tactile_encoder.num_tokens, tactile_encoder.out_dim) != (5, 64):
            raise ValueError("InterActDex requires five ordered 64D tactile tokens")

        obs_encoder = InterActDexObservationEncoder(
            tactile_encoder=tactile_encoder,
            tactile_source=tactile_source,
            pc_dim=pc_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            history_mixer=history_mixer,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        backbone = DiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.obs_token_dim,
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
            t_sample_mode="uniform",
            time_shift_alpha=1.0,
            num_inference_steps=num_inference_steps,
        )
        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
        )
        self.future_force_loss_weight = future_force_loss_weight
        self.future_force_offsets = future_force_offsets
        self.future_force_head = nn.Sequential(
            nn.Linear(obs_encoder.obs_token_dim, 128), nn.SiLU(),
            nn.Linear(128, 3 * len(future_force_offsets)),
        ) if future_force_loss_weight > 0 else None

    @property
    def effective_config(self) -> dict:
        return {
            **super().effective_config,
            "future_force_loss_weight": self.future_force_loss_weight,
            "future_force_offsets": list(self.future_force_offsets),
        }

    def loss_groups(self) -> dict[str, tuple[int, int, float]]:
        return {"arm": (0, 7, 0.5), "hand": (7, 19, 0.5)}

    def compute_loss(
        self, batch: dict, *, log_metrics: bool = True, **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        if self.future_force_head is None:
            return super().compute_loss(batch, log_metrics=log_metrics, **kwargs)

        batch_size = batch["action"].shape[0]
        future_steps = len(self.future_force_offsets)
        force = batch["future_force"]
        valid = batch["future_force_valid"]
        if force.shape != (batch_size, future_steps, 5, 3):
            raise ValueError("future_force must have shape [B, len(future_force_offsets), 5, 3]")
        if valid.shape != (batch_size, future_steps):
            raise ValueError("future_force_valid must have shape [B, len(future_force_offsets)]")

        condition = self.encode_observation(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        kwargs["dim_groups"] = self.loss_groups()
        action_loss, metrics = self.action_decoder.compute_loss(
            condition, actions, log_metrics=log_metrics, **kwargs,
        )
        # The auxiliary head sees only the last observed fingers, after history mixing.
        fingers = condition.unflatten(1, (self.n_obs_steps, -1))[:, -1, 1:6]
        prediction = self.future_force_head(fingers).reshape(batch_size, 5, future_steps, 3)
        prediction = prediction.transpose(1, 2)
        force = torch.where(valid[..., None, None], force.float(), 0.0)
        if self.obs_encoder.tactile_source == "sim":
            with torch.autocast(device_type=force.device.type, enabled=False):
                # All future targets share the query frame; future poses are never inputs.
                rotation = _rotation_6d_to_matrix(batch["obs"]["eef_pose"][:, -1, 3:].float())
                force = force @ rotation[:, None]
        target = self.normalizer["finger_tactile_force"].normalize(force)
        element_loss = F.smooth_l1_loss(
            prediction.float(), target.float(), reduction="none", beta=1.0,
        )
        mask = valid.float()
        future_force_loss = (element_loss * mask[..., None, None]).sum() / (
            15 * mask.sum().clamp_min(1)
        )
        weighted_loss = self.future_force_loss_weight * future_force_loss
        loss = action_loss + weighted_loss
        if log_metrics:
            metrics.update(
                loss=loss.detach(),
                loss_future_force=future_force_loss.detach(),
                loss_future_force_weighted=weighted_loss.detach(),
                future_force_valid_fraction=mask.mean().detach(),
            )
        return loss, metrics

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay) -> list[dict]:
        groups = super().optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
        if self.future_force_head is not None:
            groups += parameter_groups(self.future_force_head, lr=lr, weight_decay=weight_decay)
        return groups
