from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction.relative_3d_attention import Relative3DCrossAttention
from dexmani_policy.agents.obs_encoder.interaction.tactile_gated_fusion import TactileGatedFusion
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.point_transformer import PointTransformerEncoder
from dexmani_policy.agents.obs_encoder.proprio import HandGeometryEncoder, StateEncoder
from dexmani_policy.agents.policies.base import Policy, parameter_groups
from dexmani_policy.common.config import TACTILE_FEATURE_SHAPES, positive_int


class InterActFeatures(NamedTuple):
    interaction: torch.Tensor
    joint: torch.Tensor
    scene: torch.Tensor


class GlobalScenePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(192)
        self.score = nn.Linear(192, 1)
        self.output_norm = nn.LayerNorm(192)

    def forward(self, scene):
        weights = self.score(self.norm(scene)).softmax(dim=1)
        return self.output_norm((weights * scene).sum(1, keepdim=True))


class InterActDexEncoder(nn.Module):
    out_dim = 192

    def __init__(self, pc_dim, num_points, n_obs_steps, pc_encoder_config=None, fps_random_config=None):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.num_points = num_points
        self.pc_encoder = PointTransformerEncoder(input_channels=pc_dim, fps_random_config=fps_random_config, **(pc_encoder_config or {}))
        if self.pc_encoder.out_dim != 192:
            raise ValueError("InterActDex requires 192D scene features")
        self.hand_geometry = HandGeometryEncoder()
        self.cross_attention = Relative3DCrossAttention()
        self.tactile_fusion = TactileGatedFusion()
        self.joint_encoder = StateEncoder(19, 192, (128,))
        self.scene_pool = GlobalScenePool()

    @property
    def effective_config(self):
        return dict(pc_encoder_config=self.pc_encoder.effective_config,
                    fps_random_config=self.pc_encoder.fps_random_config)

    def forward(self, obs):
        pc = preprocess_point_cloud(obs["point_cloud"], self.num_points, self.pc_encoder.fps_random_config, training=self.training)
        patches = self.pc_encoder(pc)
        hand = self.hand_geometry(obs["eef_pose"], obs["fingertip_xyz"])
        geometry = self.cross_attention(hand["tokens"], patches["tokens"], hand["anchors_base"], patches["xyz"])
        interaction = self.tactile_fusion(geometry, obs["finger_tactile_force"])
        joint = self.joint_encoder(obs["joint_state"]).unsqueeze(1)
        scene = self.scene_pool(patches["tokens"])
        return InterActFeatures(*(x.unflatten(0, (-1, self.n_obs_steps)) for x in (interaction, joint, scene)))


class FutureForceHead(nn.Module):
    def __init__(self, n_obs_steps):
        super().__init__()
        self.temporal_logits = nn.Parameter(torch.zeros(n_obs_steps))
        self.mlp = nn.Sequential(nn.Linear(384, 128), nn.SiLU(), nn.Linear(128, 3))
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, interaction):
        if interaction.ndim != 4 or interaction.shape[1:] != (len(self.temporal_logits), 6, 192):
            raise ValueError("interaction must be [B,n_obs_steps,6,192]")
        pooled = (interaction * self.temporal_logits.softmax(0)[None, :, None, None]).sum(1)
        context = pooled.mean(1, keepdim=True).expand(-1, 5, -1)
        return self.mlp(torch.cat((pooled[:, 1:], context), -1))


class InterActDex(Policy):
    supports_rtc = True
    supported_action_keys = ("action",)
    loss_groups = {"arm": (0, 7, 0.5), "hand": (7, 19, 0.5)}

    def __init__(self, horizon, n_obs_steps, n_action_steps, action_dim,
                 pc_dim=6, num_points=1024, pc_encoder_config=None, fps_random_config=None,
                 timestep_embed_dim=128, n_layers=8, hidden_dim=768, n_head=8,
                 mlp_ratio=4.0, p_drop_attn=0.1, qkv_bias=False, qk_norm=False,
                 num_inference_steps=10, future_force_loss_weight=0.1,
                 future_force_horizon=8, time_shift_alpha=1.0):
        positive_int(future_force_horizon, "future_force_horizon")
        if future_force_horizon > horizon - n_obs_steps + 1:
            raise ValueError("future_force_horizon exceeds future actions")
        if not math.isfinite(future_force_loss_weight) or future_force_loss_weight < 0:
            raise ValueError("future_force_loss_weight must be finite and non-negative")
        if action_dim != 19:
            raise ValueError("InterActDex requires 19D joint actions")
        obs_encoder = InterActDexEncoder(pc_dim, num_points, n_obs_steps, pc_encoder_config, fps_random_config)
        backbone = DiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
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
        self.future_force_horizon = future_force_horizon
        self.future_force_head = FutureForceHead(n_obs_steps) if future_force_loss_weight > 0 else None

    @property
    def effective_config(self):
        return {**super().effective_config, "time_shift_alpha": self.action_decoder.time_shift_alpha,
                "future_force_loss_weight": self.future_force_loss_weight,
                "future_force_horizon": self.future_force_horizon}

    def configure_normalizer(self, feature_dims, feature_shapes):
        if TACTILE_FEATURE_SHAPES.keys() & feature_dims.keys() != {"finger_tactile_force"}:
            raise ValueError("InterActDex requires aggregate finger_tactile_force")
        super().configure_normalizer(feature_dims, feature_shapes)

    def _encode_features(self, obs):
        if any(value.shape[:2] != (obs["joint_state"].shape[0], self.n_obs_steps) for value in obs.values()):
            raise ValueError("All observations must share [B,n_obs_steps] axes")
        return self.obs_encoder(self._preprocess_obs(obs))

    @staticmethod
    def _pack_action_context(features):
        return torch.cat(features, dim=2).flatten(1, 2).float()

    def _encode_obs(self, obs):
        return self._pack_action_context(self._encode_features(obs))

    def _compute_future_force_loss(self, prediction, batch, *, log_metrics):
        target, valid = batch["future_force"], batch["future_force_valid"]
        if target.shape != prediction.shape or valid.shape != prediction.shape[:1]:
            raise ValueError("Future force labels must be [B,5,3] with [B] validity")
        with torch.autocast(device_type=target.device.type, enabled=False):
            errors = F.smooth_l1_loss(prediction.float(), target, reduction="none").mean((-1, -2))
            mask = valid.to(errors.dtype)
            denominator = mask.sum().clamp_min(1)
            loss = (errors * mask).sum() / denominator
            metrics = {}
            if log_metrics:
                with torch.no_grad():
                    mae = ((prediction.detach().float() - target).abs().mean((-1, -2)) * mask).sum() / denominator
                    metrics = dict(loss_future_force=loss.detach(),
                                   future_force_valid_fraction=mask.mean(), future_force_mae_raw=mae)
        return loss, metrics

    def compute_loss(self, batch, *, log_metrics=True, **kwargs):
        features = self._encode_features(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        action_loss, metrics = self.action_decoder.compute_loss(
            self._pack_action_context(features), actions, dim_groups=self.loss_groups, log_metrics=log_metrics, **kwargs)
        if self.future_force_head is None:
            return action_loss, metrics
        prediction = self.future_force_head(features.interaction)
        force_loss, force_metrics = self._compute_future_force_loss(prediction, batch, log_metrics=log_metrics)
        weighted = self.future_force_loss_weight * force_loss
        loss = action_loss + weighted
        if log_metrics:
            metrics.update(force_metrics)
            metrics.update(loss=loss.detach(), loss_future_force_weighted=weighted.detach())
        return loss, metrics

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay):
        groups = super().optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
        if self.future_force_head is not None:
            groups += parameter_groups(self.future_force_head, lr=lr, weight_decay=weight_decay)
        return groups
