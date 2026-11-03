from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones.ditx import CoupledDiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction.hand_interaction import (
    HandHistoryEncoder, HandSceneFusion, SceneTokenizer,
)
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import HandStateEncoder, force_world_to_eef
from dexmani_policy.agents.policies.base import Policy, parameter_groups
from dexmani_policy.common.config import positive_int, require_tactile_source


class InterActDexEncoder(nn.Module):
    out_dim = 192

    def __init__(
        self,
        tactile_encoder: nn.Module,
        tactile_source: str,
        pc_dim: int,
        num_points: int,
        n_obs_steps: int,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.num_points = num_points
        self.tactile_source = tactile_source
        self.pc_encoder = Sparse3DEncoder(
            input_channels=pc_dim, fps_random_config=fps_random_config, **(pc_encoder_config or {}),
        )
        self.hand_state = HandStateEncoder(token_dim=self.out_dim)
        self.hand_scene = HandSceneFusion(self.pc_encoder.out_dim, tactile_encoder)
        self.scene_tokenizer = SceneTokenizer(self.pc_encoder.out_dim)
        self.scene_projection = nn.Linear(self.pc_encoder.out_dim, self.out_dim)
        self.hand_history = HandHistoryEncoder(n_obs_steps)

    @property
    def effective_config(self) -> dict:
        return {
            "pc_encoder_config": self.pc_encoder.effective_config,
            "fps_random_config": self.pc_encoder.fps_random_config,
            "tactile_encoder": self.hand_scene.tactile_encoder.effective_config,
            "tactile_source": self.tactile_source,
            "n_obs_steps": self.n_obs_steps,
        }

    def forward(self, obs: dict) -> torch.Tensor:
        pc = preprocess_point_cloud(
            obs["point_cloud"], self.num_points,
            self.pc_encoder.fps_random_config, training=self.training,
        )
        patches = self.pc_encoder(pc)
        scene = self.scene_projection(self.scene_tokenizer(
            patches["tokens"], patches["xyz"],
        ))
        hand = self.hand_state(obs["joint_state"], obs["eef_pose"], obs["fingertip_xyz"])
        interaction = self.hand_scene(
            pc, patches, hand["anchor_xyz"], hand["hand_tokens"],
            rotation_world_from_eef=hand["rotation_world_from_eef"],
            query_context=scene.mean(dim=1), tactile=obs["finger_tactile_force"],
        )
        scene = scene.unflatten(0, (-1, self.n_obs_steps))
        interaction = interaction.unflatten(0, (-1, self.n_obs_steps))
        interaction = self.hand_history(interaction)
        # Each frame contains six body tokens followed by twelve scene tokens.
        return torch.cat((interaction, scene), dim=2).flatten(1, 2).float()


class InterActDex(Policy):
    supported_execution_modes = ("sync", "async", "rtc")
    supported_action_keys = ("action",)
    loss_groups = {"arm": (0, 7, 0.5), "hand": (7, 19, 0.5)}

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        tactile_source: str,
        tactile_encoder: nn.Module,
        pc_dim: int = 6,
        num_points: int = 1024,
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
        interaction_horizon: int = 4,
        time_shift_alpha: float = 1.0,
    ) -> None:
        future_force_loss_weight = float(future_force_loss_weight)
        if not math.isfinite(future_force_loss_weight) or future_force_loss_weight < 0:
            raise ValueError("future_force_loss_weight must be finite and non-negative")
        positive_int(interaction_horizon, "interaction_horizon")
        future_steps = horizon - (n_obs_steps - 1)
        if interaction_horizon > future_steps:
            raise ValueError(
                f"interaction_horizon K ({interaction_horizon}) exceeds available future actions ({future_steps})"
            )
        if action_dim != 19:
            raise ValueError("InterActDex requires 19D joint state and actions")
        require_tactile_source(("finger_tactile_force",), tactile_source)
        if tactile_encoder.input_key != "finger_tactile_force":
            raise ValueError("InterActDex requires aggregate finger_tactile_force")
        if (tactile_encoder.num_tokens, tactile_encoder.out_dim) != (5, 64):
            raise ValueError("InterActDex requires five ordered 64D tactile tokens")

        obs_encoder = InterActDexEncoder(
            tactile_encoder=tactile_encoder,
            tactile_source=tactile_source,
            pc_dim=pc_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        backbone = CoupledDiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            interaction_horizon=interaction_horizon,
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
            time_shift_alpha=time_shift_alpha,
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
        self.interaction_horizon = interaction_horizon
        self.future_force_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 3),
        ) if future_force_loss_weight > 0 else None
        if self.future_force_head is not None:
            nn.init.zeros_(self.future_force_head[-1].weight)
            nn.init.zeros_(self.future_force_head[-1].bias)

    @property
    def tactile_source(self) -> str:
        return self.obs_encoder.tactile_source

    @property
    def effective_config(self) -> dict:
        return {
            **super().effective_config,
            "time_shift_alpha": self.action_decoder.time_shift_alpha,
            "future_force_loss_weight": self.future_force_loss_weight,
            "interaction_horizon": self.interaction_horizon,
        }

    def _preprocess_obs(self, obs_dict: dict) -> dict:
        if self.tactile_source == "sim":
            obs_dict = {**obs_dict, "finger_tactile_force": force_world_to_eef(
                obs_dict["finger_tactile_force"], obs_dict["eef_pose"],
            )}
        return super()._preprocess_obs(obs_dict)

    def _compute_future_force_loss(
        self, prediction: torch.Tensor, batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force = batch["future_force"]
        valid = batch["future_force_valid"]

        with torch.autocast(device_type=force.device.type, enabled=False):
            current = batch["obs"]["finger_tactile_force"][:, -1:].float()
            delta = torch.where(valid[..., None, None], force.float() - current, 0.0)
            if self.tactile_source == "sim":
                # Rotate the world-frame difference using only the query pose.
                delta = force_world_to_eef(delta, batch["obs"]["eef_pose"][:, -1:])
            target = delta * self.normalizer["finger_tactile_force"].scale
        element_loss = F.smooth_l1_loss(
            prediction.float(), target.float(), reduction="none", beta=1.0,
        )
        mask = valid.float()
        future_force_loss = (element_loss * mask[..., None, None]).sum() / (
            15 * mask.sum().clamp_min(1)
        )
        return future_force_loss, mask

    def compute_loss(
        self, batch: dict, *, log_metrics: bool = True, **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        if self.future_force_head is None:
            return super().compute_loss(batch, log_metrics=log_metrics, **kwargs)

        condition = self._encode_obs(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        kwargs["dim_groups"] = self.loss_groups
        kwargs["return_features"] = True
        action_loss, metrics, features = self.action_decoder.compute_loss(
            condition, actions, log_metrics=log_metrics, **kwargs,
        )
        fingers = features[:, self.horizon + 1:self.horizon + 6]
        prediction = self.future_force_head(fingers).unsqueeze(1)
        future_force_loss, mask = self._compute_future_force_loss(prediction, batch)
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
