"""RGB–point observation tokens with late joint-state conditioning."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .pointcloud.camera_projection import ImageProjection, project_to_image
from .pointcloud.ops import index_points, preprocess_point_cloud
from .pointcloud.point_transformer import LocalPatchEncoder, SpatialContextEncoder
from .pointcloud.rgb_patch_fusion import RGBPatchFusion
from .proprio import StateEncoder
from .rgb.build import build_rgb_encoder


class RGBPointObservationEncoder(nn.Module):
    def __init__(
        self,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        n_obs_steps: int = 2,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dinov3",
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
    ) -> None:
        super().__init__()
        if rgb_semantic_sampling not in ("center", "group_mean"):
            raise ValueError("rgb_semantic_sampling must be 'center' or 'group_mean'")
        self.rgb_semantic_sampling = rgb_semantic_sampling
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.rgb_backbone_name = rgb_backbone_name

        fps_config = {
            "use_random": True, "use_random_start": True,
            "use_shuffle_output": False,
            **(fps_random_config or {}),
        }
        patch_config = dict(pc_encoder_config or {})
        spatial_config = {
            key: patch_config.pop(key)
            for key in ("depth", "num_heads", "mlp_ratio", "rope_scale", "rope_base")
            if key in patch_config
        }
        self.patch_encoder = LocalPatchEncoder(
            input_channels=pc_dim, fps_random_config=fps_config, **patch_config,
        )
        self.rgb_encoder, self.image_processor = build_rgb_encoder(
            rgb_backbone_name, rgb_backbone_config,
        )
        self.rgb_fusion = RGBPatchFusion(self.rgb_encoder.out_dim, self.patch_encoder.out_dim)
        self.spatial_encoder = SpatialContextEncoder(
            channels=2 * self.patch_encoder.out_dim, **spatial_config,
        )
        self.out_dim = self.spatial_encoder.out_dim
        self.joint_encoder = StateEncoder(state_dim, self.out_dim)

        self.token_type_embedding = nn.Parameter(torch.empty(1, 2, self.out_dim))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        self.token_norm = nn.LayerNorm(self.out_dim)

    @property
    def effective_config(self) -> dict:
        return {
            # Store the local width; RGB fusion doubles it for spatial attention.
            "pc_encoder_config": {
                **self.spatial_encoder.effective_config, **self.patch_encoder.effective_config,
            },
            "fps_random_config": self.patch_encoder.fps_random_config,
            "rgb_backbone_name": self.rgb_backbone_name,
            "rgb_semantic_sampling": self.rgb_semantic_sampling,
            "rgb_backbone_config": {
                **self.rgb_encoder.effective_config, **self.image_processor.config,
            },
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        bt = obs["point_cloud"].shape[0]
        if obs["rgb"].ndim != 4 or obs["rgb"].shape[:2] != (bt, 3):
            raise ValueError("rgb must have shape [BT,3,H,W]")
        if obs["joint_state"].shape[0] != bt:
            raise ValueError("joint_state and point_cloud must have the same BT")
        for name, width in (("camera_intrinsic", 9), ("camera_extrinsic", 12)):
            if obs[name].shape != (bt, width):
                raise ValueError(f"{name} must have shape [BT,{width}]")
        if bt % self.n_obs_steps:
            raise ValueError("BT must be divisible by n_obs_steps")

        points = preprocess_point_cloud(
            obs["point_cloud"], self.num_points,
            self.patch_encoder.fps_random_config, training=self.training,
        )
        patches = self.patch_encoder(points)

        rgb, image_transform = self.image_processor.process_images(obs["rgb"], return_transform=True)
        # Sampling coordinates assume the patch grid covers the whole processed image.
        if any(size % self.rgb_encoder.patch_size for size in rgb.shape[-2:]):
            raise ValueError("RGB input height and width must be divisible by the encoder patch_size")
        patch_grid = tuple(size // self.rgb_encoder.patch_size for size in rgb.shape[-2:])
        patch_tokens = self.rgb_encoder(rgb)["patch_tokens"]
        if patch_tokens.shape[1] != patch_grid[0] * patch_grid[1]:
            raise ValueError("RGB patch token count must match the processed image patch grid")
        image_grid = patch_tokens.unflatten(1, patch_grid)
        sample_xyz = patches["xyz"].unsqueeze(2)
        if self.rgb_semantic_sampling == "group_mean":
            sample_xyz = index_points(points[..., :3], patches["neighbor_idx"])
        projection = project_to_image(
            points_world=sample_xyz.flatten(1, 2),
            intrinsics=obs["camera_intrinsic"].reshape(bt, 3, 3),
            T_camera_from_world=obs["camera_extrinsic"].reshape(bt, 3, 4),
            image_transform=image_transform,
            image_hw=(rgb.shape[-2], rgb.shape[-1]),
        )

        group_shape = sample_xyz.shape[1:3]
        projection = ImageProjection(
            projection.grid_xy.unflatten(1, group_shape),
            projection.valid.unflatten(1, group_shape),
        )
        tokens = self.rgb_fusion(patches["tokens"], image_grid, projection)
        spatial = self.spatial_encoder(tokens, patches["xyz"])
        joint = self.joint_encoder(obs["joint_state"]).unsqueeze(1)
        spatial = spatial + self.token_type_embedding[:, :1].to(spatial.dtype)
        joint = joint + self.token_type_embedding[:, 1:].to(joint.dtype)
        frames = self.token_norm(torch.cat((spatial, joint), dim=1))
        return frames.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)
