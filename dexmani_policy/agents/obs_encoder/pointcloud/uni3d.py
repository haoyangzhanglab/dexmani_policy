"""Uni3D point cloud encoder adapted from R3D-Policy.

FPS → k-NN patches → PointNet-style PatchEncoder (LayerNorm) →
ViT self-attention (timm) → dense per-patch tokens + Fourier spatial pc_pe.
"""

import torch
import torch.nn as nn
import os
import logging
from math import pi as _pi

from .ops import farthest_point_sample

logger = logging.getLogger(__name__)


def knn_points(query, key, k, sorted=False):
    """k-nearest neighbors via pairwise distance → (dist, indices)."""
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """Randomly drop up to max_dropout_ratio of points, replacing with first point."""
    B, N, _ = batch_pc.shape
    result = torch.clone(batch_pc)
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio
        drop_idx = torch.where(torch.rand(N, device=batch_pc.device) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx, :] = batch_pc[b, 0, :].unsqueeze(0)
    return result


class KNNGrouper(nn.Module):
    """FPS → K centers, k-NN → K groups of relative xyz + features."""

    def __init__(self, num_groups, group_size, radius=None,
                 centralize_features=False, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features
        self.fps_random_config = fps_random_config or {}

    def forward(self, xyz, features, use_fps=True):
        B, N, _ = xyz.shape
        with torch.no_grad():
            centers, _ = farthest_point_sample(xyz, num_samples=self.num_groups, **self.fps_random_config)
            _, knn_idx = knn_points(centers, xyz, self.group_size)

        batch_offset = torch.arange(B, device=xyz.device) * N
        batch_offset = batch_offset.reshape(-1, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(B, self.num_groups, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(B, self.num_groups, self.group_size, features.shape[-1])

        group_feats = torch.cat([nbr_xyz, nbr_feats], dim=-1)
        return dict(features=group_feats, centers=centers, knn_idx=knn_idx)


class PatchEncoder(nn.Module):
    """PointNet-style patch encoder: conv1 → max pool → cat → conv2 → max pool."""

    def __init__(self, in_channels, out_channels, hidden_dims):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches):
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand(-1, -1, x.shape[2], -1), x], dim=-1)
        x = self.conv2(x)
        return torch.max(x, dim=-2).values


class PatchEmbed(nn.Module):
    """KNNGrouper + PatchEncoder: group points into patches and encode."""

    def __init__(self, in_channels, out_channels, num_patches, patch_size,
                 radius=None, centralize_features=False, fps_random_config=None):
        super().__init__()
        self.grouper = KNNGrouper(
            num_patches, patch_size, radius=radius,
            centralize_features=centralize_features,
            fps_random_config=fps_random_config,
        )
        self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])

    def forward(self, coords, features):
        patches = self.grouper(coords, features)
        patches["embeddings"] = self.patch_encoder(patches["features"])
        return patches


class PatchDropout(nn.Module):
    """Randomly drop patch tokens during training."""

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]

        B, num_tokens, _ = x.shape
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = torch.randn(B, num_tokens, device=x.device)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class PositionEmbeddingRandom(nn.Module):
    """Fourier feature encoding of 3D coordinates (adapted from SAM)."""

    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        """coords: (..., 3) normalized to [-1, 1]."""
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * _pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords):
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            raise ValueError(
                f"Input coordinates must be normalized to [-1, 1]. "
                f"Bounds: ({coords.min().item():.4f}, {coords.max().item():.4f})"
            )
        return self._pe_encoding(coords)


class Uni3DPointcloudEncoder(nn.Module):
    """Patch-based ViT point cloud encoder (LayerNorm throughout).

    Pipeline: random_point_dropout → PatchEmbed → patch_proj + pos_embed →
    ViT self-attention → out_proj → tokens + Fourier pc_pe.
    Supports selective pretrained weight loading from safetensors.
    """

    def __init__(self,
                 pc_model='eva02_tiny_patch14_224',
                 embed_dim=256,
                 num_group=512,
                 group_size=32,
                 pc_in_channels=6,
                 patch_dropout=0.0,
                 drop_path_rate=0.0,
                 feature_mode='pointsam',
                 use_pretrained_weights=False,
                 pretrained_weights_path=None,
                 fps_random_config=None,
                 **kwargs):
        super().__init__()

        import timm
        self.transformer = timm.create_model(
            pc_model, pretrained=False, drop_path_rate=drop_path_rate
        )
        self.transformer_dim = self.transformer.embed_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.feature_mode = feature_mode

        self.patch_embed = PatchEmbed(
            in_channels=pc_in_channels,
            out_channels=512,
            num_patches=num_group, patch_size=group_size,
            fps_random_config=fps_random_config,
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim),
        )

        self.patch_proj = nn.Linear(self.patch_embed.patch_encoder.out_channels,
                                     self.transformer_dim)
        self.out_proj = nn.Linear(self.transformer_dim, embed_dim)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        if feature_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.transformer_dim))
            exclude_first = True
        else:
            exclude_first = False

        self.patch_dropout = PatchDropout(patch_dropout, exclude_first_token=exclude_first) \
            if patch_dropout > 0. else nn.Identity()

        # Freeze unused timm params (not in forward path) for DDP compatibility
        for name, p in self.transformer.named_parameters():
            if name in ('cls_token', 'pos_embed') or name.startswith('patch_embed.'):
                p.requires_grad_(False)

        if use_pretrained_weights:
            self._load_pretrained_weights(pretrained_weights_path)
        else:
            logger.info("[Uni3DPointcloudEncoder] Random initialization (training from scratch)")

    def _load_pretrained_weights(self, pretrained_weights_path):
        """Selectively load pretrained weights from safetensors (strict=False)."""
        if pretrained_weights_path is None:
            logger.warning("[Uni3DPointcloudEncoder] pretrained_weights_path is None, skipping")
            return

        safetensors_path = os.path.join(pretrained_weights_path, "model.safetensors")
        if not os.path.exists(safetensors_path):
            logger.warning(
                "[Uni3DPointcloudEncoder] Pretrained weights not found: %s", safetensors_path
            )
            return

        from safetensors.torch import load_file
        checkpoint = load_file(safetensors_path)

        # Remap keys: strip 'pc_encoder.' prefix
        processed_state_dict = {}
        for key in list(checkpoint.keys()):
            if key.startswith('pc_encoder.'):
                new_key = key.replace('pc_encoder.', '')
                processed_state_dict[new_key] = checkpoint[key]

        missing_keys, unexpected_keys = self.load_state_dict(
            processed_state_dict, strict=False
        )
        if missing_keys:
            logger.info("[Uni3DPointcloudEncoder] Missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.info("[Uni3DPointcloudEncoder] Unexpected keys: %s", unexpected_keys)
        logger.info(
            "[Uni3DPointcloudEncoder] Pretrained weights loaded from %s", pretrained_weights_path
        )

    @property
    def out_dim(self):
        return self.embed_dim

    @property
    def num_tokens(self):
        return self.num_group

    def forward(self, pcd, eval=False):
        if pcd.shape[-1] == 3:
            colors = torch.zeros_like(pcd)
            pcd = torch.cat([pcd, colors], dim=-1)
        elif pcd.shape[-1] > 6:
            pcd = pcd[..., :6]

        if not eval:
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)

        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()

        patches = self.patch_embed(pts, colors)
        patch_embed = patches["embeddings"]
        centers = patches["centers"]

        patch_embed = self.patch_proj(patch_embed)
        pos_embed = self.pos_embed(centers)

        if self.feature_mode == "cls":
            cls_tokens = self.cls_token.expand(patch_embed.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(pos_embed.size(0), -1, -1)
            patch_embed = torch.cat((cls_tokens, patch_embed), dim=1)
            pos_embed = torch.cat((cls_pos, pos_embed), dim=1)

        x = patch_embed + pos_embed

        if not eval:
            x = self.patch_dropout(x)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        if self.feature_mode == "cls":
            x = self.transformer.norm(x[:, 0, :])
        elif self.feature_mode == "max_pooling":
            x = self.transformer.norm(torch.max(x, dim=1)[0])
        else:
            x = self.transformer.norm(x)

        x = self.transformer.fc_norm(x)
        tokens = self.out_proj(x)
        pc_pe = self.pe_layer(centers)

        return tokens, pc_pe
