"""Spatial observation tokens with aligned RGB semantics and proprioception."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .pointcloud.camera_projection import project_to_image
from .pointcloud.ops import preprocess_point_cloud
from .pointcloud.point_transformer import GlobalPointTransformer, PointPatchEncoder
from .pointcloud.rgb_patch_fusion import RGBPatchFusion
from .proprio import StateEncoder
from .rgb.build import build_rgb_encoder


class SpatialObservationEncoder(nn.Module):
    out_dim = 256

    def __init__(
        self,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        n_obs_steps: int = 2,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.rgb_backbone_name = rgb_backbone_name

        fps_config = {
            "use_random": True, "use_random_start": True,
            "use_shuffle_output": False,
            **(fps_random_config or {}),
        }
        rgb_config = {
            "tune_mode": "freeze",
            **(rgb_backbone_config or {}),
        }

        patch_config = dict(pc_encoder_config or {})
        global_config = {
            key: patch_config.pop(key)
            for key in ("depth", "num_heads", "mlp_ratio", "rope_scale", "rope_base")
            if key in patch_config
        }
        self.patch_encoder = PointPatchEncoder(
            input_channels=pc_dim, fps_random_config=fps_config, **patch_config,
        )
        self.rgb_encoder, self.image_processor = build_rgb_encoder(rgb_backbone_name, rgb_config)
        self.rgb_fusion = RGBPatchFusion(self.rgb_encoder.out_dim, self.patch_encoder.out_dim)
        self.global_encoder = GlobalPointTransformer(
            channels=2 * self.patch_encoder.out_dim, **global_config,
        )
        self.output_projection = nn.Linear(self.global_encoder.out_dim, self.out_dim)
        self.state_encoder = StateEncoder(state_dim, self.out_dim)

        self.modality_embedding = nn.Parameter(torch.empty(1, 2, self.out_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.output_norm = nn.LayerNorm(self.out_dim)

    @property
    def effective_config(self) -> dict:
        return {
            "pc_encoder_config": {
                **self.global_encoder.effective_config, **self.patch_encoder.effective_config,
            },
            "fps_random_config": self.patch_encoder.fps_random_config,
            "rgb_backbone_name": self.rgb_backbone_name,
            "rgb_backbone_config": {
                **self.rgb_encoder.effective_config, **self.image_processor.config,
            },
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
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
        image_grid = self.rgb_encoder(rgb)["patch_tokens"].unflatten(1, patch_grid)
        projection = project_to_image(
            patches["xyz"], obs["camera_intrinsic"].reshape(-1, 3, 3),
            obs["camera_extrinsic"].reshape(-1, 3, 4),
            image_transform, (rgb.shape[-2], rgb.shape[-1]),
        )

        tokens = self.rgb_fusion(patches["tokens"], image_grid, projection)
        tokens = self.global_encoder(tokens, patches["xyz"])

        spatial = self.output_projection(tokens)
        state = self.state_encoder(obs["joint_state"]).unsqueeze(1)
        spatial = spatial + self.modality_embedding[:, :1].to(spatial.dtype)
        state = state + self.modality_embedding[:, 1:].to(state.dtype)
        frames = self.output_norm(torch.cat((spatial, state), dim=1))
        return frames.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)
