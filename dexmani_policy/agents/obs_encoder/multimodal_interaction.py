"""Separate scene and tactile-aware hand tokens in a shared metric Base-aligned frame."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .interaction.relative_3d_attention import Relative3DCrossAttention
from .interaction.tactile_gated_fusion import TactileGatedFusion
from .pointcloud.ops import preprocess_point_cloud
from .pointcloud.point_transformer import LocalPatchEncoder, SpatialContextEncoder
from .pointcloud.rgb_patch_fusion import RGBPatchFusion, project_rgb_patches
from .proprio import HandGeometryEncoder
from .rgb.build import build_rgb_encoder


class MultimodalInteractionEncoder(nn.Module):
    def __init__(
        self,
        pc_dim: int = 3,
        num_points: int = 1024,
        patch_config: dict | None = None,
        spatial_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dinov3",
        rgb_backbone_config: dict | None = None,
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
        hand_geometry_config: dict | None = None,
        relative_geometry_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if rgb_backbone_name not in ("dino", "dinov3", "siglip"):
            raise ValueError("rgb_backbone_name must be 'dino', 'dinov3' or 'siglip'")
        if rgb_semantic_sampling not in ("center", "group_mean"):
            raise ValueError("rgb_semantic_sampling must be 'center' or 'group_mean'")
        self.num_points = num_points
        self.rgb_backbone_name = rgb_backbone_name
        self.rgb_semantic_sampling = rgb_semantic_sampling
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
        self.rgb_fusion = RGBPatchFusion(image_channels=self.rgb_encoder.out_dim, token_channels=192)
        self.spatial_encoder = SpatialContextEncoder(channels=384, **(spatial_config or {}))
        self.scene_projection = nn.Linear(384, 192)
        self.hand_encoder = HandGeometryEncoder(**(hand_geometry_config or {}))
        self.relative_attention = Relative3DCrossAttention(geometry_scale=relative_geometry_scale)
        self.tactile_fusion = TactileGatedFusion()

    def initialize_weights(self, source: str | None) -> None:
        """Load RGB backbone weights from a local or supported pretrained source."""
        if source is None:
            return
        self.rgb_encoder.initialize_weights(source)

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
            "rgb_semantic_sampling": self.rgb_semantic_sampling,
            "rgb_backbone_config": {
                **self.rgb_encoder.effective_config, **self.image_processor.config,
            },
            "hand_geometry_config": {
                "wrist_scale": self.hand_encoder.wrist_scale,
                "finger_scale": self.hand_encoder.finger_scale,
            },
            "relative_geometry_scale": self.relative_attention.geometry_scale,
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inputs have a flat BT axis; metric geometry and source-frame forces are unchanged."""
        points = preprocess_point_cloud(
            obs["point_cloud"], self.num_points,
            self.patch_encoder.fps_random_config, training=self.training,
        )
        patches = self.patch_encoder(points)

        image_grid, projection = project_rgb_patches(
            obs, points, patches, image_processor=self.image_processor,
            rgb_encoder=self.rgb_encoder, sampling=self.rgb_semantic_sampling,
        )
        scene = self.rgb_fusion(patches["tokens"], image_grid, projection)
        scene = self.spatial_encoder(scene, patches["xyz"])
        scene = self.scene_projection(scene)

        hand = self.hand_encoder(obs["eef_pose"], obs["fingertip_xyz"])
        hand_tokens = self.relative_attention(
            body_tokens=hand["tokens"],
            scene_tokens=scene,
            anchors_base=hand["anchors_base"],
            scene_xyz_base=patches["xyz"],
        )
        hand_tokens = self.tactile_fusion(hand_tokens, obs["finger_tactile_force"])
        return {
            "scene_tokens": scene,
            "scene_xyz_base": patches["xyz"],
            "hand_tokens": hand_tokens,
            "hand_anchors_base": hand["anchors_base"],
        }
