"""Hand-object interaction encoding from RGB-D geometry, kinematics and tactile force."""

from __future__ import annotations

from typing import Literal

import torch

from ..rgb_point import RGBPointSceneEncoder
from .geometric_relation import GeoAwareRelationAttention
from .hand_kinematics import HandKinematicEncoder
from .tactile import FingerTactileEncoder, TactileFusion


class HandObjectInteractionEncoder(RGBPointSceneEncoder):
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
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
    ) -> None:
        super().__init__(
            pc_dim=pc_dim, num_points=num_points, patch_config=dict(channels=self.out_dim, **(patch_config or {})),
            spatial_config=spatial_config or {}, fps_random_config=fps_random_config,
            rgb_backbone_name=rgb_backbone_name, rgb_backbone_config=rgb_backbone_config,
            rgb_semantic_sampling=rgb_semantic_sampling,
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
            **super().effective_config,
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
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return six wrist-first interaction tokens [B, 6, 192]."""
        return self.forward_with_aux(obs)["tokens"]

    def forward_with_aux(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return tokens and intermediate tensors for geometry/projection diagnostics."""
        if obs["point_cloud"][..., :3].eq(0).all(dim=(1, 2)).any():
            raise ValueError("point_cloud contains an empty frame (all XYZ coordinates are zero)")
        scene_tokens, scene_points_base, projection_valid = self.encode_scene(obs)
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
