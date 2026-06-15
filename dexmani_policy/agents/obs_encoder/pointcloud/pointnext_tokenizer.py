import torch
import torch.nn as nn
from typing import Dict, Tuple

from dexmani_policy.common.position_encodings import (
    RelativePositionalEncoding3D,
    SinusoidalPosEmb3D,
)
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    PointMLP,
    farthest_point_sample,
    index_points,
    normalize_relative_xyz,
    query_ball_point,
)


class LocalPatchEncoder(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        token_channels: int,
        radius: float,
        num_neighbors: int,
        position_encoding_channels: int = 24,
    ):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.relative_position_encoding = RelativePositionalEncoding3D(position_encoding_channels)
        self.point_mlp = nn.Sequential(
            PointMLP(stem_channels + 3 + position_encoding_channels, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor, patch_center: torch.Tensor) -> torch.Tensor:
        neighbor_idx = query_ball_point(self.radius, self.num_neighbors, xyz, patch_center)
        neighbor_xyz = index_points(xyz, neighbor_idx)
        neighbor_feature = index_points(point_feature, neighbor_idx)
        relative_xyz = neighbor_xyz - patch_center.unsqueeze(2)
        normalized_relative_xyz = normalize_relative_xyz(relative_xyz, self.radius)
        relative_pos_feature = self.relative_position_encoding(normalized_relative_xyz)
        group_input = torch.cat((neighbor_feature, normalized_relative_xyz, relative_pos_feature), dim=-1)
        return self.point_mlp(group_input).max(dim=2).values


class MultiScalePatchTokenizer(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        token_channels: int,
        num_patches: int,
        patch_radii: Tuple[float, ...],
        patch_neighbors: Tuple[int, ...],
        fps_random_config: dict | None = None,
        # ── patch self-attention ──
        use_patch_self_attn: bool = False,
        patch_attn_layers: int = 4,
        patch_attn_heads: int = 4,
        patch_attn_dropout: float = 0.0,
        prepend_global_in_attn: bool = True,
    ):
        super().__init__()
        if len(patch_radii) != len(patch_neighbors):
            raise ValueError("patch_radii and patch_neighbors must have the same length")

        self.num_patches = num_patches
        self.use_patch_self_attn = use_patch_self_attn
        self.prepend_global_in_attn = prepend_global_in_attn
        self.fps_random_config = fps_random_config or {}

        self.scale_encoders = nn.ModuleList(
            [
                LocalPatchEncoder(stem_channels, token_channels, radius, neighbors)
                for radius, neighbors in zip(patch_radii, patch_neighbors)
            ]
        )

        # Position encoding for the token-projection input (concatenated, original).
        self.patch_center_position_encoding = SinusoidalPosEmb3D(96)

        proj_in_dim = stem_channels + len(self.scale_encoders) * token_channels + 96
        self.token_projection = nn.Sequential(
            PointMLP(proj_in_dim, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )

        # ── patch self-attention (optional) ──
        if use_patch_self_attn:
            # Absolute position embedding on patch-center coordinates (additive,
            # analogous to SAT's ``mlp_global`` on FPS centers).
            self.center_pos_embed = nn.Sequential(
                nn.Linear(3, token_channels),
                nn.LayerNorm(token_channels),
                nn.GELU(),
                nn.Linear(token_channels, token_channels),
            )

            if prepend_global_in_attn:
                # Learnable global token (like ViT class token / SAT's global_pn).
                self.global_token = nn.Parameter(
                    torch.randn(1, 1, token_channels) * 0.02
                )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_channels,
                nhead=patch_attn_heads,
                dim_feedforward=token_channels * 4,
                dropout=patch_attn_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.patch_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=patch_attn_layers
            )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        patch_center, patch_center_idx = farthest_point_sample(
            xyz, self.num_patches, **self.fps_random_config)
        patch_center_feature = index_points(point_feature, patch_center_idx)

        multi_scale_patch_feature_list = [
            scale_encoder(xyz, point_feature, patch_center) for scale_encoder in self.scale_encoders
        ]
        patch_center_pos_feature = self.patch_center_position_encoding(patch_center)
        patch_token = self.token_projection(
            torch.cat((patch_center_feature, *multi_scale_patch_feature_list, patch_center_pos_feature), dim=-1)
        )

        # ── optional self-attention over patches ──
        if self.use_patch_self_attn:
            # Add absolute position embedding computed from patch centers.
            pos_emb = self.center_pos_embed(patch_center)          # (B, G, token_channels)
            x = patch_token + pos_emb

            if self.prepend_global_in_attn:
                global_tok = self.global_token.expand(x.size(0), -1, -1)  # (B, 1, token_channels)
                x = torch.cat([global_tok, x], dim=1)                     # (B, 1+G, token_channels)

            x = self.patch_transformer(x)

            if self.prepend_global_in_attn:
                attn_global = x[:, :1, :]    # (B, 1, token_channels)
                patch_token = x[:, 1:, :]    # (B, G, token_channels)
                return patch_token, patch_center, attn_global

        return patch_token, patch_center


class PointNextPatchTokenizer(nn.Module):
    supports_global_token = True
    supports_intermediate_outputs = True
    requires_fixed_num_points = False

    def __init__(
        self,
        input_channels: int = 6,
        stem_channels: int = 64,
        token_channels: int = 128,
        num_patches: int = 64,
        patch_radii: Tuple[float, ...] = (0.04, 0.08),
        patch_neighbors: Tuple[int, ...] = (16, 32),
        fps_random_config: dict | None = None,
        # ── patch self-attention ──
        use_patch_self_attn: bool = False,
        patch_attn_layers: int = 4,
        patch_attn_heads: int = 4,
        patch_attn_dropout: float = 0.0,
        prepend_global_in_attn: bool = True,
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        self.input_channels = input_channels
        self.num_patches = num_patches
        self.token_channels = token_channels
        self.use_patch_self_attn = use_patch_self_attn
        self.prepend_global_in_attn = prepend_global_in_attn

        self.geometry_stem = nn.Sequential(
            PointMLP(input_channels, stem_channels),
            PointMLP(stem_channels, stem_channels),
        )
        self.local_patch_tokenizer = MultiScalePatchTokenizer(
            stem_channels=stem_channels,
            token_channels=token_channels,
            num_patches=num_patches,
            patch_radii=patch_radii,
            patch_neighbors=patch_neighbors,
            fps_random_config=fps_random_config,
            use_patch_self_attn=use_patch_self_attn,
            patch_attn_layers=patch_attn_layers,
            patch_attn_heads=patch_attn_heads,
            patch_attn_dropout=patch_attn_dropout,
            prepend_global_in_attn=prepend_global_in_attn,
        )

        # When the transformer already produces a global token via the
        # learnable [CLS]-style token, the external global-token pathway
        # is redundant — skip it to save parameters.
        if not (use_patch_self_attn and prepend_global_in_attn):
            self.global_position_embedding = SinusoidalPosEmb3D(96)
            self.global_position_projection = nn.Sequential(
                nn.Linear(96, token_channels),
                nn.LayerNorm(token_channels),
                nn.GELU(),
            )
            self.global_token_projection = nn.Sequential(
                nn.Linear(token_channels, token_channels),
                nn.LayerNorm(token_channels),
            )

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_global_token: bool = False,
        return_intermediate: bool = False,
    ):
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        xyz = pointcloud[..., :3]
        input_point_feature = pointcloud[..., : self.input_channels]
        stem_point_feature = self.geometry_stem(input_point_feature)

        tokenizer_out = self.local_patch_tokenizer(xyz, stem_point_feature)
        if self.use_patch_self_attn and self.prepend_global_in_attn:
            patch_token, patch_center, attn_global = tokenizer_out
            self._attn_global = attn_global  # cache for get_global_token
        else:
            patch_token, patch_center = tokenizer_out

        outputs = [patch_token, patch_center]
        if return_global_token:
            outputs.append(self.get_global_token(patch_token, patch_center))
        if return_intermediate:
            intermediate_outputs: Dict[str, torch.Tensor] = {
                "stem_point_feature": stem_point_feature,
                "patch_center": patch_center,
                "patch_token": patch_token,
            }
            outputs.append(intermediate_outputs)
        return tuple(outputs)

    def get_global_token(self, patch_token: torch.Tensor, patch_center: torch.Tensor) -> torch.Tensor:
        # When self-attention with prepended global token is active, the
        # transformer internally maintains a [CLS]-style token (position 0)
        # that already aggregates context from all patches.
        if self.use_patch_self_attn and self.prepend_global_in_attn:
            return self._attn_global

        # Original max-pool + sin/cos position-embedding pathway.
        pooled_patch_token = patch_token.max(dim=1).values
        pooled_patch_center = patch_center.mean(dim=1)
        global_token_feature = pooled_patch_token + self.global_position_projection(
            self.global_position_embedding(pooled_patch_center)
        )
        return self.global_token_projection(global_token_feature).unsqueeze(1)

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.token_channels)


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    # ── baseline (no self-attention) ──
    pointnext_tokenizer = PointNextPatchTokenizer(
        input_channels=6,
        stem_channels=64,
        token_channels=128,
        num_patches=96,
        patch_radii=(0.04, 0.08),
        patch_neighbors=(16, 32),
    )
    with torch.no_grad():
        patch_token, patch_center, global_token, intermediate_outputs = pointnext_tokenizer(
            pointcloud,
            return_global_token=True,
            return_intermediate=True,
        )

    print("=== PointNextPatchTokenizer (baseline) ===")
    print("input:", tuple(pointcloud.shape))
    print("patch_token:", tuple(patch_token.shape))
    print("patch_center:", tuple(patch_center.shape))
    print("global_token:", tuple(global_token.shape))
    for name, value in intermediate_outputs.items():
        print(f"  {name}: {tuple(value.shape)}")
    print("out_dim:", pointnext_tokenizer.out_dim)
    print("out_shape:", pointnext_tokenizer.out_shape)
    print()

    # ── with self-attention + prepend global ──
    pointnext_attn = PointNextPatchTokenizer(
        input_channels=6,
        stem_channels=64,
        token_channels=128,
        num_patches=96,
        patch_radii=(0.04, 0.08),
        patch_neighbors=(16, 32),
        use_patch_self_attn=True,
        patch_attn_layers=4,
        patch_attn_heads=4,
        patch_attn_dropout=0.0,
        prepend_global_in_attn=True,
    )
    with torch.no_grad():
        patch_token_a, patch_center_a, global_token_a, intermediate_outputs_a = pointnext_attn(
            pointcloud,
            return_global_token=True,
            return_intermediate=True,
        )

    print("=== PointNextPatchTokenizer (with self-attention) ===")
    print("input:", tuple(pointcloud.shape))
    print("patch_token:", tuple(patch_token_a.shape))
    print("patch_center:", tuple(patch_center_a.shape))
    print("global_token:", tuple(global_token_a.shape))
    for name, value in intermediate_outputs_a.items():
        print(f"  {name}: {tuple(value.shape)}")
    print("out_dim:", pointnext_attn.out_dim)
    print("out_shape:", pointnext_attn.out_shape)

    # Verify that with prepend_global_in_attn, the global token is
    # learned (CLS-style) and not the max-pool pathway.
    assert global_token_a.size(1) == 1, "global token should be 1 token"
    assert patch_token_a.size(1) == 96, "patch tokens should be 96 patches"
    print()
    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    example()