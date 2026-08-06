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
        num_points: Number of points per frame (after FPS downsampling).
        hidden_dims: MLP hidden layer sizes.
    """

    supports_global_token = False

    def __init__(
        self,
        input_channels: int = 3,
        out_channels: int = 128,
        num_points: int = 256,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        self.input_channels = input_channels
        self._out_channels = out_channels
        self._num_points = num_points

        layers = []
        in_dim = input_channels
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        self.final_proj = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self, pointcloud: torch.Tensor, return_global_token: bool = False, **kwargs):
        """Encode point cloud into per-point features.

        Args:
            pointcloud: ``(B, N, C)`` tensor.
            return_global_token: Unused (accepted for interface compat).

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
