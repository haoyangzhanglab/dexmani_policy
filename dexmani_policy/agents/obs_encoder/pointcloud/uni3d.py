"""Uni3D patch Transformer adapted from R3D-Policy."""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from .ops import index_points, sample_point_patches
from .pointnet import build_mlp

logger = logging.getLogger(__name__)


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """Replace points with the first point using a dropout rate sampled per cloud."""
    B, N, _ = batch_pc.shape
    ratios = torch.rand(B, 1, device=batch_pc.device) * max_dropout_ratio
    mask = torch.rand(B, N, device=batch_pc.device) <= ratios
    return torch.where(mask.unsqueeze(-1), batch_pc[:, :1, :], batch_pc)


class PointNetPatchEncoder(nn.Module):
    """PointNet-style patch encoder: conv1 → max pool → cat → conv2 → max pool."""

    def __init__(self, in_channels, out_channels, hidden_dims):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = build_mlp([in_channels, hidden_dims[0], hidden_dims[0]])
        self.conv2 = build_mlp([hidden_dims[0] * 2, hidden_dims[1], out_channels])

    def forward(self, point_patches):
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand(-1, -1, x.shape[2], -1), x], dim=-1)
        x = self.conv2(x)
        return torch.max(x, dim=-2).values


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

    def forward(self, coords, features):
        centers, _, neighbors = sample_point_patches(
            coords,
            self.num_patches,
            self.patch_size,
            self.fps_random_config,
            self.training,
            knn_backend="torch",
        )
        relative_xyz = index_points(coords, neighbors) - centers.unsqueeze(2)
        attributes = index_points(features, neighbors)
        patch_features = torch.cat((relative_xyz, attributes), dim=-1)
        return {
            "embeddings": self.patch_encoder(patch_features),
            "centers": centers,
        }


class PatchDropout(nn.Module):
    """Randomly drop patch tokens during training."""

    def __init__(self, prob):
        super().__init__()
        if not 0 <= prob < 1.0:
            raise ValueError("patch dropout probability must be in [0, 1)")
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x, None

        B, num_tokens, _ = x.shape
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = torch.randn(B, num_tokens, device=x.device)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        return index_points(x, patch_indices_keep), patch_indices_keep


class RandomFourierPositionEmbedding3D(nn.Module):
    """Fourier feature encoding of 3D coordinates (adapted from SAM)."""

    def __init__(self, num_pos_feats=64, scale=1.0):
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
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
        pc_in_channels=6,
        patch_dropout=0.0,
        drop_path_rate=0.0,
        use_pretrained_weights=False,
        pretrained_weights_path=None,
        allow_random_init=False,
        fps_random_config=None,
    ):
        super().__init__()
        if type(pc_in_channels) is not int or pc_in_channels != 6:
            raise ValueError(
                f"Uni3D pc_in_channels must be the integer 6, got {pc_in_channels!r}; "
                "it is the internal patch feature width. XYZ input is padded with "
                "zero RGB and does not require pc_in_channels=3."
            )

        import timm

        self.transformer = timm.create_model(
            pc_model, pretrained=False, drop_path_rate=drop_path_rate
        )
        self.transformer_dim = self.transformer.embed_dim
        self.embed_dim = embed_dim

        self.patch_embed = PointNetPatchEmbedding(
            in_channels=pc_in_channels,
            out_channels=512,
            num_patches=num_group,
            patch_size=group_size,
            fps_random_config=fps_random_config,
        )

        self.pos_embed = build_mlp([3, 128, self.transformer_dim], normalize=False)

        self.patch_proj = nn.Linear(
            self.patch_embed.patch_encoder.out_channels, self.transformer_dim
        )
        self.out_proj = nn.Linear(self.transformer_dim, embed_dim)
        self.pe_layer = RandomFourierPositionEmbedding3D(embed_dim // 2)

        self.patch_dropout = PatchDropout(patch_dropout)

        # Freeze unused image-embedding parameters so DDP sees no unused gradients.
        for name, p in self.transformer.named_parameters():
            if name in ("cls_token", "pos_embed") or name.startswith("patch_embed."):
                p.requires_grad_(False)

        self.use_pretrained_weights = use_pretrained_weights
        self.pretrained_weights_path = pretrained_weights_path
        self.allow_random_init = allow_random_init

    def initialize_training(self):
        if self.use_pretrained_weights:
            self.load_pretrained_weights(self.pretrained_weights_path, self.allow_random_init)
        else:
            logger.info("[Uni3DEncoder] Random initialization (training from scratch)")

    def load_pretrained_weights(self, pretrained_weights_path, allow_random_init=False):
        """Load R3D weights, downloading missing files; require an explicit random-init opt-in."""
        if pretrained_weights_path is None:
            if allow_random_init:
                logger.warning(
                    "[Uni3DEncoder] pretrained_weights_path is None; "
                    "training from scratch (allow_random_init=True)."
                )
                return
            raise ValueError(
                "[Uni3DEncoder] use_pretrained_weights=True but "
                "pretrained_weights_path is None. Provide the path or set "
                "allow_random_init=True to train from scratch."
            )

        weights_dir = Path(pretrained_weights_path)
        if not weights_dir.is_absolute():
            weights_dir = Path(__file__).resolve().parents[4] / weights_dir
        safetensors_path = weights_dir / "model.safetensors"
        if not safetensors_path.exists():
            logger.info(
                "[Uni3DEncoder] Local weights not found, downloading from HuggingFace Hub..."
            )
            try:
                from huggingface_hub import hf_hub_download

                weights_dir.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    "eddie-cui/r3d-weights",
                    "model.safetensors",
                    local_dir=str(weights_dir),
                )
            except Exception as e:
                if allow_random_init:
                    logger.warning(
                        "[Uni3DEncoder] Auto-download failed (%s); "
                        "training from scratch (allow_random_init=True).",
                        e,
                    )
                    return
                raise RuntimeError(
                    f"[Uni3DEncoder] pretrained weights missing and "
                    f"auto-download failed: {e}. Run "
                    f"'bash scripts/utils/download_pretrained.sh' or set "
                    f"allow_random_init=True."
                ) from e

        from safetensors.torch import load_file

        checkpoint = load_file(str(safetensors_path))

        processed_state_dict = {
            key.removeprefix("pc_encoder."): value
            for key, value in checkpoint.items()
            if key.startswith("pc_encoder.")
        }

        missing_keys, unexpected_keys = self.load_state_dict(processed_state_dict, strict=False)

        total_expected = len(self.state_dict())
        loaded = total_expected - len(missing_keys)
        ratio = loaded / total_expected

        if missing_keys:
            logger.info("[Uni3DEncoder] Missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.info("[Uni3DEncoder] Unexpected keys: %s", unexpected_keys)

        # Reject checkpoints that miss most of this encoder.
        if ratio < 0.5:
            msg = (
                f"[Uni3DEncoder] Only {loaded}/{total_expected} ({ratio:.1%}) "
                f"expected keys loaded from {safetensors_path} — checkpoint is likely "
                f"for a different model."
            )
            if allow_random_init:
                logger.warning(
                    "%s Continuing with partial/random init (allow_random_init=True).",
                    msg,
                )
                return
            raise ValueError(msg)

        logger.info(
            "[Uni3DEncoder] Pretrained weights loaded from %s (%d/%d keys, %.1f%%)",
            safetensors_path,
            loaded,
            total_expected,
            ratio * 100,
        )

    @property
    def out_dim(self):
        return self.embed_dim

    def forward(self, pcd):
        if pcd.ndim != 3 or (pcd.shape[-1] != 3 and pcd.shape[-1] < 6):
            raise ValueError(
                "Uni3D point cloud must have shape [B,N,C] with C=3 (XYZ) or "
                "C>=6 (first six channels are XYZRGB); keep pc_in_channels=6 "
                "even for XYZ input, which is padded with zero RGB."
            )
        if pcd.shape[-1] == 3:
            colors = torch.zeros_like(pcd)
            pcd = torch.cat([pcd, colors], dim=-1)
        elif pcd.shape[-1] > 6:
            pcd = pcd[..., :6]

        if self.training:
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)

        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()

        patches = self.patch_embed(pts, colors)
        patch_embed = patches["embeddings"]
        centers = patches["centers"]

        patch_embed = self.patch_proj(patch_embed)
        pos_embed = self.pos_embed(centers)

        x = patch_embed + pos_embed

        if self.training:
            x, keep = self.patch_dropout(x)
            if keep is not None:
                centers = index_points(centers, keep)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.norm(x)
        x = self.transformer.fc_norm(x)
        tokens = self.out_proj(x)
        pc_pe = self.pe_layer(centers)

        return tokens, pc_pe
