import torch
import torch.nn as nn


class PointNetDense(nn.Module):
    """Dense per-point PointNet encoder (official ManiFlow DP3Encoder style).

    Processes each point independently through an MLP and keeps all
    per-point features — **no pooling, no patch aggregation, no global
    token**.  Every point becomes an observation token for the DiTX
    cross-attention, matching the official ManiFlow design where the
    transformer decides which points to attend to.

    Compatible with the patch-tokenizer registry interface via
    ``build_pc_patch_tokenizer("pointnet_dense", ...)``.

    Parameters:
        input_channels: Point cloud channels (3 for xyz, 6 for xyz+rgb).
        out_channels: Output feature dimension per point.
        num_points: Expected point count for out_shape metadata; sampling is
            performed by the caller.
    """

    supports_global_token = False

    def __init__(
        self,
        input_channels: int = 3,
        out_channels: int = 128,
        num_points: int = 256,
    ):
        super().__init__()
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")

        self.input_channels = input_channels
        self._out_channels = out_channels
        self._num_points = num_points

        layers = []
        in_dim = input_channels
        for h in (64, 128, 256):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        if input_channels == 6:
            # Official RGB branch: no norm/activation before final projection.
            layers.append(nn.Linear(256, 512))
            in_dim = 512
        self.mlp = nn.Sequential(*layers)

        self.final_proj = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, pointcloud: torch.Tensor):
        """Encode point cloud into per-point features.

        Args:
            pointcloud: ``(B, N, C)`` tensor.

        Returns:
            ``(B, N, out_channels)`` per-point feature tensor.
        """
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        x = pointcloud[..., : self.input_channels]
        x = self.mlp(x)
        x = self.final_proj(x)
        return x

    @property
    def out_dim(self) -> int:
        return self._out_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self._num_points, self._out_channels)
