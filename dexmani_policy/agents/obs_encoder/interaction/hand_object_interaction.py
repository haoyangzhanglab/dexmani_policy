"""Hand-object interaction encoding from RGB-D geometry, kinematics and tactile force."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..pointcloud.ops import preprocess_point_cloud
from ..pointcloud.point_transformer import LocalPatchEncoder, SpatialContextEncoder
from ..pointcloud.rgb_patch_fusion import RGBPatchFusion, project_rgb_patches
from ..rgb.build import build_rgb_encoder
from .geometric_relation import GeoAwareRelationAttention
from .hand_kinematics import HandKinematicEncoder
from .tactile import FingerTactileEncoder, TactileFusion


class HandObjectInteractionEncoder(nn.Module):
    """Encode six wrist-first hand landmarks against RGB-aligned 3D scene memory."""

    out_dim = GeoAwareRelationAttention.embed_dim

    def __init__(
        self,
        pc_dim: int = 3,
        num_points: int = 1024,
        patch_config: dict | None = None,
        spatial_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_backbone_config: dict | None = None,
        rgb_patch_sampling: Literal["center", "group_mean"] = "center",
    ) -> None:
        super().__init__()
        if rgb_backbone_name not in ("dino", "dinov3", "siglip"):
            raise ValueError("rgb_backbone_name must be 'dino', 'dinov3' or 'siglip'")
        if rgb_patch_sampling not in ("center", "group_mean"):
            raise ValueError("rgb_patch_sampling must be 'center' or 'group_mean'")
        self.num_points = num_points
        self.rgb_backbone_name = rgb_backbone_name
        self.rgb_patch_sampling = rgb_patch_sampling
        fps_config = {
            "use_random": True, "use_random_start": True,
            "use_shuffle_output": False,
            **(fps_random_config or {}),
        }
        self.patch_encoder = LocalPatchEncoder(
            input_channels=pc_dim, channels=192,
            fps_random_config=fps_config, **(patch_config or {}),
        )
        self.rgb_encoder, self.image_processor = build_rgb_encoder(
            rgb_backbone_name, rgb_backbone_config,
        )
        scene_channels = self.patch_encoder.out_dim
        self.rgb_fusion = RGBPatchFusion(self.rgb_encoder.out_dim, scene_channels)
        self.spatial_encoder = SpatialContextEncoder(
            channels=self.rgb_fusion.out_dim, **(spatial_config or {}),
        )
        self.hand_encoder = HandKinematicEncoder()
        self.tactile_encoder = FingerTactileEncoder()
        self.tactile_fusion = TactileFusion()
        self.geometric_relation = GeoAwareRelationAttention(
            rope_scale=self.spatial_encoder.effective_config["rope_scale"],
            rope_base=self.spatial_encoder.effective_config["rope_base"],
        )

    @property
    def effective_config(self) -> dict:
        return {
            "pc_dim": self.patch_encoder.input_channels,
            "num_points": self.num_points,
            "patch_config": {
                key: value for key, value in self.patch_encoder.effective_config.items()
                if key != "channels"
            },
            "spatial_config": {
                key: value for key, value in self.spatial_encoder.effective_config.items()
                if key != "channels"
            },
            "fps_random_config": dict(self.patch_encoder.fps_random_config),
            "rgb_backbone_name": self.rgb_backbone_name,
            "rgb_patch_sampling": self.rgb_patch_sampling,
            "rgb_backbone_config": {
                **self.rgb_encoder.effective_config, **self.image_processor.config,
            },
        }

    def _encode_scene_memory(
        self, obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_cloud = obs["point_cloud"]
        if point_cloud[..., :3].eq(0).all(dim=(1, 2)).any():
            raise ValueError("point_cloud contains an empty frame (all XYZ coordinates are zero)")
        points = preprocess_point_cloud(
            point_cloud, self.num_points,
            self.patch_encoder.fps_random_config, training=self.training,
        )
        patches = self.patch_encoder(points)

        image_grid, projection = project_rgb_patches(
            obs, points, patches, image_processor=self.image_processor,
            rgb_encoder=self.rgb_encoder, sampling=self.rgb_patch_sampling,
        )
        scene_points_base = patches["xyz"]
        scene_tokens = self.rgb_fusion(patches["tokens"], image_grid, projection)
        scene_tokens = self.spatial_encoder(scene_tokens, scene_points_base)
        return scene_tokens, scene_points_base, projection.valid

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return six wrist-first interaction tokens [B, 6, 192]."""
        return self.forward_with_aux(obs)["tokens"]

    def forward_with_aux(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return tokens and intermediate tensors for geometry/projection diagnostics."""
        scene_tokens, scene_points_base, projection_valid = self._encode_scene_memory(obs)
        hand_tokens, hand_landmarks_base = self.hand_encoder(obs["eef_pose"], obs["fingertip_xyz"])
        tactile_features = self.tactile_encoder(obs["finger_tactile_force"])
        hand_tokens = self.tactile_fusion(hand_tokens, tactile_features)
        interaction_tokens = self.geometric_relation(
            hand_tokens, scene_tokens, hand_landmarks_base, scene_points_base,
        )
        return {
            "tokens": interaction_tokens,
            "scene_tokens": scene_tokens,
            "scene_points_base": scene_points_base,
            "hand_landmarks_base": hand_landmarks_base,
            "local_tokens": tactile_features["local_tokens"],
            "global_token": tactile_features["global_token"],
            "rgb_projection_valid": projection_valid,
        }
