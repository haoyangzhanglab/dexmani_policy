"""Spatial observation tokens with aligned RGB semantics and proprioception."""

from __future__ import annotations

import math

import torch
from torch import nn

from .pointcloud.feature_alignment import FeatureAlignment
from .pointcloud.ops import preprocess_point_cloud
from .pointcloud.projection import project_points
from .pointcloud.sparse3d_encoder import Sparse3DEncoder
from .proprio import StateEncoder
from .rgb.build import build_rgb_encoder


class SpatialObservationEncoder(nn.Module):
    consumed_observation_fields = (
        "joint_state", "point_cloud", "rgb", "camera_intrinsic", "camera_extrinsic",
    )
    out_dim = 256

    @staticmethod
    def prepare_rgb_restore_config(rgb):
        if rgb is None or rgb.get("architecture") is None:
            raise ValueError("Dense checkpoints require saved DINO architecture for offline restoration")
        return {**rgb, "load_pretrained": False}

    @staticmethod
    def validate_normalization(spec: dict[str, str]) -> None:
        Sparse3DEncoder.validate_normalization(spec)
        if spec.get("joint_state") != "limits":
            raise ValueError("Spatial observations require normalization.joint_state: limits")
        for key in ("rgb", "camera_intrinsic", "camera_extrinsic"):
            if spec.get(key) != "identity":
                raise ValueError(f"Spatial observations require normalization.{key}: identity")

    def __init__(
        self,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        n_obs_steps: int = 2,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        fusion_stage: str = "post_global",
        semantic_pool_sigma: float | None = None,
    ) -> None:
        super().__init__()
        if fusion_stage not in {"post_global", "pre_global"}:
            raise ValueError("fusion_stage must be post_global or pre_global")
        if semantic_pool_sigma is not None and (
            not math.isfinite(semantic_pool_sigma) or semantic_pool_sigma <= 0
        ):
            raise ValueError("semantic_pool_sigma must be None or finite and positive")
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.fusion_stage = fusion_stage
        self.semantic_pool_sigma = semantic_pool_sigma

        fps_config = {
            "use_random": True, "use_random_start": True,
            "use_shuffle_output": False, "random_noise_scale": 0.0,
            **(fps_random_config or {}),
        }
        pc_config = {"channels": 192, "depth": 2, "num_heads": 8, **(pc_encoder_config or {})}
        rgb_config = {
            "model_name": "facebook/dinov2-small", "tune_mode": "freeze",
            **(rgb_backbone_config or {}),
        }

        self.pc_encoder = Sparse3DEncoder(
            input_channels=pc_dim, fps_random_config=fps_config, **pc_config,
        )
        self.rgb_encoder, self.image_processor = build_rgb_encoder("dino", rgb_config)
        self.alignment = FeatureAlignment(self.rgb_encoder.out_dim, self.pc_encoder.out_dim)
        self.spatial_projection = nn.Linear(self.pc_encoder.out_dim, self.out_dim)
        self.state_mlp = StateEncoder(state_dim, self.out_dim)

        self.modality_embedding = nn.Parameter(torch.empty(1, 2, self.out_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.output_norm = nn.LayerNorm(self.out_dim)

    @property
    def effective_config(self) -> dict:
        return {
            "pc_encoder_config": self.pc_encoder.effective_config,
            "fps_random_config": self.pc_encoder.fps_random_config,
            "rgb_backbone_config": {
                **self.rgb_encoder.effective_config, **self.image_processor.config,
            },
            "fusion_stage": self.fusion_stage,
            "semantic_pool_sigma": self.semantic_pool_sigma,
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        points = preprocess_point_cloud(
            obs["point_cloud"], self.num_points, self.pc_encoder.input_channels == 3,
            self.pc_encoder.fps_random_config, training=self.training,
        )
        patches = self.pc_encoder.encode_patches(points)

        rgb, image_transform = self.image_processor.process_images(obs["rgb"], return_transform=True)
        patch_grid = tuple(size // self.rgb_encoder.patch_size for size in rgb.shape[-2:])
        image_grid = self.rgb_encoder(rgb)["patch_tokens"].unflatten(1, patch_grid)[:, None]
        projection = project_points(
            points[..., :3], obs["camera_intrinsic"].reshape(-1, 1, 3, 3),
            obs["camera_extrinsic"].reshape(-1, 1, 3, 4),
            image_transform[:, None], (rgb.shape[-2], rgb.shape[-1]),
        )

        member_weight = None
        if self.semantic_pool_sigma is not None:
            with torch.no_grad():
                distance_squared = patches["relative_xyz"].float().square().sum(-1)
                member_weight = torch.exp(-distance_squared / (2 * self.semantic_pool_sigma**2))

        tokens = patches["tokens"]
        if self.fusion_stage == "post_global":
            tokens = self.pc_encoder.encode_global(tokens, patches["rope_cos"], patches["rope_sin"])
        tokens = self.alignment(
            tokens, image_grid, projection, patches["neighbor_indices"], member_weight=member_weight,
        )["tokens"]
        if self.fusion_stage == "pre_global":
            tokens = self.pc_encoder.encode_global(tokens, patches["rope_cos"], patches["rope_sin"])

        spatial = self.spatial_projection(tokens)
        state = self.state_mlp(obs["joint_state"]).unsqueeze(1)
        spatial = spatial + self.modality_embedding[:, :1].to(spatial.dtype)
        state = state + self.modality_embedding[:, 1:].to(state.dtype)
        frames = self.output_norm(torch.cat((spatial, state), dim=1))
        return frames.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)

    @property
    def out_shape(self) -> tuple[int, int]:
        return self.n_obs_steps * (self.pc_encoder.num_patches + 1), self.out_dim
