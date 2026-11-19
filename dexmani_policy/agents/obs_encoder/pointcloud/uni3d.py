"""Uni3D patch Transformer adapted from R3D-Policy."""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from dexmani_policy.agents.modules import build_mlp

from .ops import index_points, sample_point_patches

logger = logging.getLogger(__name__)


def random_point_dropout(pointcloud, max_dropout_ratio=0.875):
    """Replace points with the first point using a dropout rate sampled per cloud."""
    batch_size, num_points, _ = pointcloud.shape
    ratios = torch.rand(batch_size, 1, device=pointcloud.device) * max_dropout_ratio
    mask = torch.rand(batch_size, num_points, device=pointcloud.device) <= ratios
    return torch.where(mask.unsqueeze(-1), pointcloud[:, :1, :], pointcloud)


class PointNetPatchEncoder(nn.Module):
    """PointNet-style patch encoder: conv1 → max pool → cat → conv2 → max pool."""

    def __init__(self, in_channels, out_channels, hidden_dims):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = build_mlp([in_channels, hidden_dims[0], hidden_dims[0]])
        self.conv2 = build_mlp([hidden_dims[0] * 2, hidden_dims[1], out_channels])

    def forward(self, point_patches):
        features = self.conv1(point_patches)
        context = features.max(dim=2, keepdim=True).values
        features = torch.cat((context.expand_as(features), features), dim=-1)
        return self.conv2(features).max(dim=2).values


class PointNetPatchEmbedding(nn.Module):
    """FPS and kNN grouping followed by PointNet patch embedding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches,
        patch_size,
        fps_random_config=None,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.fps_random_config = dict(fps_random_config or {})
        self.patch_encoder = PointNetPatchEncoder(in_channels, out_channels, [128, 512])

    def forward(self, pointcloud):
        centers, neighbors = sample_point_patches(
            pointcloud[..., :3].contiguous(),
            self.num_patches,
            self.patch_size,
            self.fps_random_config,
            self.training,
            torch_knn=True,
        )
        patch_features = index_points(pointcloud, neighbors)
        patch_features[..., :3].sub_(centers.unsqueeze(2))
        return self.patch_encoder(patch_features), centers


class RandomFourierPositionEmbedding3D(nn.Module):
    """Fourier feature encoding of 3D coordinates (adapted from SAM)."""

    def __init__(self, num_pos_feats):
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((3, num_pos_feats)),
        )

    def forward(self, coords):
        phases = 2 * math.pi * (coords @ self.positional_encoding_gaussian_matrix)
        return torch.cat([phases.sin(), phases.cos()], dim=-1)


class Uni3DEncoder(nn.Module):
    """Encode XYZ or XYZRGB using internal 6-channel (relative XYZ + RGB) patches."""

    def __init__(
        self,
        pc_model="eva02_tiny_patch14_224",
        embed_dim=256,
        num_group=512,
        group_size=32,
        patch_dropout=0.0,
        drop_path_rate=0.0,
        fps_random_config=None,
    ):
        super().__init__()
        import timm

        self.transformer = timm.create_model(
            pc_model, pretrained=False, drop_path_rate=drop_path_rate
        )
        transformer_dim = self.transformer.embed_dim
        self.out_dim = embed_dim

        self.patch_embed = PointNetPatchEmbedding(
            in_channels=6,
            out_channels=512,
            num_patches=num_group,
            patch_size=group_size,
            fps_random_config=fps_random_config,
        )

        self.pos_embed = build_mlp([3, 128, transformer_dim], normalize=False)
        self.patch_proj = nn.Linear(self.patch_embed.patch_encoder.out_channels, transformer_dim)
        self.out_proj = nn.Linear(transformer_dim, embed_dim)
        self.pe_layer = RandomFourierPositionEmbedding3D(embed_dim // 2)

        if not 0 <= patch_dropout < 1.0:
            raise ValueError("patch dropout probability must be in [0, 1)")
        self.patch_dropout = patch_dropout

        # Image-embedding parameters are unused by the point-cloud encoder.
        for name, p in self.transformer.named_parameters():
            if name in ("cls_token", "pos_embed") or name.startswith("patch_embed."):
                p.requires_grad_(False)

    def initialize_weights(self, source: str | None):
        """Load the local R3D trunk only for fresh training."""
        if source is None:
            logger.info("[Uni3DEncoder] Random initialization (training from scratch)")
            return
        safetensors_path = Path(source).expanduser() / "model.safetensors"

        from safetensors.torch import load_file

        checkpoint = load_file(str(safetensors_path))

        processed_state_dict = {
            key.removeprefix("pc_encoder."): value
            for key, value in checkpoint.items()
            if key.startswith("pc_encoder.")
        }

        # Fourier positions belong to this policy, outside the pretrained segmentation trunk.
        pretrained = nn.ModuleDict({
            name: module for name, module in self.named_children() if name != "pe_layer"
        })
        pretrained.load_state_dict(processed_state_dict, strict=True)
        logger.info("[Uni3DEncoder] Pretrained weights loaded from %s", safetensors_path)

    def forward(self, pointcloud):
        if pointcloud.ndim != 3 or pointcloud.shape[-1] not in (3, 6):
            raise ValueError(
                "Uni3D point cloud must have shape [B,N,C] with C=3 (XYZ) or "
                "C=6 (XYZRGB); XYZ input is padded with zero RGB."
            )
        if pointcloud.shape[-1] == 3:
            pointcloud = torch.cat((pointcloud, torch.zeros_like(pointcloud)), dim=-1)

        if self.training:
            pointcloud = random_point_dropout(pointcloud, max_dropout_ratio=0.8)

        patch_embed, centers = self.patch_embed(pointcloud)
        patch_embed = self.patch_proj(patch_embed)
        pos_embed = self.pos_embed(centers)

        x = patch_embed + pos_embed

        if self.training:
            if self.patch_dropout != 0.0:
                batch_size, num_tokens, _ = x.shape
                num_keep = max(1, int(num_tokens * (1 - self.patch_dropout)))
                scores = torch.randn(batch_size, num_tokens, device=x.device)
                keep = scores.topk(num_keep, dim=-1).indices
                x = index_points(x, keep)
                centers = index_points(centers, keep)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.norm(x)
        x = self.transformer.fc_norm(x)
        tokens = self.out_proj(x)
        pc_pe = self.pe_layer(centers)

        return tokens, pc_pe
