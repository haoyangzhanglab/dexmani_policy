"""Compose scene, finger-aligned interaction and proprioception tokens."""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.obs_encoder.finger_aligned_fusion import FingerAlignedFusion
from dexmani_policy.agents.obs_encoder.pointcloud.interaction_encoder import (
    PointPatchInteractionEncoder,
)
from dexmani_policy.agents.obs_encoder.pointcloud.point_patch import PointPatchEncoder
from dexmani_policy.agents.obs_encoder.pointcloud.scene_compressor import PointPatchSceneCompressor
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.obs_encoder.tactile.xhand_frame_encoder import XHandFrameEncoder


class InteractionObsEncoder(nn.Module):
    """BaseAgent's flattened [B*T,...] observations -> (context, aux).

    point_cloud, eef_pose and fingertip_points must share a metric coordinate
    frame and use identity normalization. eef_pose is XYZ + rot6d (first two
    rotation columns); assemble it from observed state upstream when necessary.
    joint_state and tactile channels are normalized upstream. Forces retain
    their sensor convention; no force-frame transform is inferred here.

    Each frame contributes [scene(S), wrist+fingers(6), state(1)], oldest first.
    ConsistencyDiTX supplies frame position embeddings; this module adds only
    stream identity. A configured tactile_valid_key must be an unnormalized
    bool [B*T,5] field; otherwise all readings are considered available. Callers
    must mark dropped readings invalid, not replace them with valid zeros.
    """

    def __init__(
        self,
        state_dim: int,
        pc_dim: int = 6,
        n_obs_steps: int = 2,
        token_channels: int = 192,
        num_scene_tokens: int = 16,
        tactile_input_key: str = "contact_force",
        tactile_valid_key: str | None = None,
        pc_encoder_config: dict | None = None,
        interaction_encoder_config: dict | None = None,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        if state_dim < 1 or n_obs_steps < 1:
            raise ValueError("state_dim and n_obs_steps must be positive")
        self.state_dim = state_dim
        self.pc_dim = pc_dim
        self.n_obs_steps = n_obs_steps
        self.obs_token_dim = token_channels
        self.num_scene_tokens = num_scene_tokens
        self.tactile_valid_key = None
        self.validate_finite = validate_finite

        pc_config = dict(pc_encoder_config or {})
        interaction_config = dict(interaction_encoder_config or {})
        if {"input_channels", "token_channels"} & pc_config.keys():
            raise ValueError("configure point channels through pc_dim and token_channels")
        if "token_channels" in interaction_config:
            raise ValueError("configure shared interaction channels through token_channels")
        if interaction_config.pop("self_depth", 0) != 0:
            raise ValueError("interaction self_depth must be 0; hand attention follows tactile fusion")
        if pc_config.get("semantic_channels") is not None:
            raise ValueError("InteractionObsEncoder consumes geometry, not image semantic inputs")
        self.pc_encoder = PointPatchEncoder(
            input_channels=pc_dim, token_channels=token_channels, **pc_config
        )
        if not 1 <= num_scene_tokens <= self.pc_encoder.num_patches:
            raise ValueError("require 1 <= num_scene_tokens <= num_patches")
        self.interaction_encoder = PointPatchInteractionEncoder(
            token_channels=token_channels, self_depth=0, **interaction_config
        )
        self.scene_compressor = PointPatchSceneCompressor(
            token_channels=token_channels, num_scene_tokens=num_scene_tokens
        )
        self.tactile_encoder = XHandFrameEncoder(tactile_input_key, validate_finite)
        self.fusion = FingerAlignedFusion(token_channels, validate_finite=validate_finite)
        self.state_mlp = create_state_mlp(state_dim, token_channels)
        self.stream_embedding = nn.Parameter(torch.empty(3, token_channels))
        nn.init.normal_(self.stream_embedding, std=0.02)
        if tactile_valid_key is not None and (
            not isinstance(tactile_valid_key, str)
            or not tactile_valid_key
            or tactile_valid_key in self.consumed_observation_fields
        ):
            raise ValueError("tactile_valid_key must be a distinct nonempty observation key")
        self.tactile_valid_key = tactile_valid_key

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        fields = ("point_cloud", "eef_pose", "fingertip_points", "joint_state", "contact_force")
        if self.tactile_encoder.input_key == "tactile_force":
            fields += ("tactile_force",)
        if self.tactile_valid_key is not None:
            fields += (self.tactile_valid_key,)
        return fields

    @torch.no_grad()
    def _geometry_metadata(
        self,
        eef_pose: torch.Tensor,
        fingertips: torch.Tensor,
        interaction: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # Geometry descriptors stay FP32 under AMP; null mass is stop-gradient.
        with torch.autocast(device_type=eef_pose.device.type, enabled=False):
            wrist = eef_pose.float()
            first, second = wrist[:, 3:6], wrist[:, 6:9]
            x = F.normalize(first, dim=-1)
            second = second - (second * x).sum(-1, keepdim=True) * x
            if self.validate_finite and bool(
                ((first.norm(dim=-1) < 1e-6) | (second.norm(dim=-1) < 1e-6)).any()
            ):
                raise ValueError("eef_pose rot6d must contain two non-collinear rotation columns")
            y = F.normalize(second, dim=-1)
            rotation = torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)
            scale = self.interaction_encoder.metric_scale
            relative = (fingertips.float() - wrist[:, None, :3]) @ rotation / scale
            distance = interaction["min_observed_distance"][:, 1:].float()
            observed = torch.isfinite(distance)
            support = interaction["near_has_support"][:, 1:].any(-1)
            null_mass = interaction["near_null_mass"][:, 1:].float().mean(-1)
            return torch.cat(
                (relative, (distance / scale).clamp(max=10)[..., None],
                 observed[..., None].float(), support[..., None].float(), null_mass[..., None]),
                dim=-1,
            )

    def forward(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return [B,T*(S+7),D] and optional diagnostics, never auxiliary losses."""
        pointcloud = obs["point_cloud"]
        if pointcloud.ndim != 3 or pointcloud.shape[-1] < self.pc_dim:
            raise ValueError("point_cloud must have shape [B*T,N,C>=pc_dim]")
        flat_batch = pointcloud.shape[0]
        if flat_batch == 0 or flat_batch % self.n_obs_steps:
            raise ValueError("observation batch must be positive and divisible by n_obs_steps")
        batch_size = flat_batch // self.n_obs_steps
        for key in self.consumed_observation_fields:
            value = obs[key]
            if value.ndim < 1 or value.shape[0] != flat_batch or value.device != pointcloud.device:
                raise ValueError(f"{key}: all observations must share flattened batch and device")
            if key != self.tactile_valid_key and not value.is_floating_point():
                raise TypeError(f"{key} must be floating point")

        pointcloud = pointcloud[..., :self.pc_dim].float()
        eef_pose, fingertips = obs["eef_pose"], obs["fingertip_points"]
        force, state = obs["contact_force"], obs["joint_state"]
        if fingertips.shape == (flat_batch, 15):
            fingertips = fingertips.reshape(flat_batch, 5, 3)
        if force.shape == (flat_batch, 15):
            force = force.reshape(flat_batch, 5, 3)
        for value, shape, name in (
            (eef_pose, (flat_batch, 9), "eef_pose"),
            (fingertips, (flat_batch, 5, 3), "fingertip_points"),
            (force, (flat_batch, 5, 3), "contact_force"),
            (state, (flat_batch, self.state_dim), "joint_state"),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.validate_finite:
            for value in (pointcloud, eef_pose, fingertips, state):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("geometry and joint_state observations must be finite")

        valid = None
        if self.tactile_valid_key is not None:
            valid = obs[self.tactile_valid_key]
            if valid.shape != (flat_batch, 5) or valid.dtype != torch.bool:
                raise ValueError("tactile validity must be bool [B*T,5]")
            valid = valid.unflatten(0, (batch_size, self.n_obs_steps))
        sensor = force if self.tactile_encoder.input_key == "contact_force" else obs["tactile_force"]
        tactile = self.tactile_encoder(sensor.unflatten(0, (batch_size, self.n_obs_steps)), valid)

        # The scene and hand branches share exactly the same points and patches.
        patches = self.pc_encoder(pointcloud, return_intermediate=True)
        interaction = self.interaction_encoder(
            pointcloud, patches, eef_pose, fingertips, return_intermediate=True
        )
        meta = self._geometry_metadata(eef_pose, fingertips, interaction)
        history_shape = (batch_size, self.n_obs_steps)
        fused = self.fusion(
            interaction["interaction_token"].unflatten(0, history_shape),
            tactile,
            force.unflatten(0, history_shape),
            meta.unflatten(0, history_shape),
            valid,
            return_intermediate=return_intermediate,
        )
        scene = self.scene_compressor(patches["patch_token"], patches["patch_center"])["scene_token"]
        scene = scene.unflatten(0, history_shape)
        state = self.state_mlp(state.to(dtype=self.stream_embedding.dtype)).unflatten(0, history_shape)
        hand = fused["hand_token"]
        # Concatenate within each frame BEFORE flattening; the decoder assigns
        # time embeddings to contiguous, equally sized frame groups.
        streams = self.stream_embedding.to(scene.dtype)
        context = torch.cat(
            (scene + streams[0], hand + streams[1], state[:, :, None] + streams[2]), dim=2
        ).flatten(1, 2)
        aux = {}
        if return_intermediate:
            aux = {
                **fused,
                "scene_token": scene,
                "state_token": state,
                "geometry_meta": meta.unflatten(0, history_shape),
            }
        return context, aux

    @property
    def out_dim(self) -> int:
        return self.obs_token_dim

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.n_obs_steps * (self.num_scene_tokens + 7), self.obs_token_dim)
