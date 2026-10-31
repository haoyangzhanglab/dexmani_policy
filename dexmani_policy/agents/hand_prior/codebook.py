"""PCA-ordered hand states with continuous indices in [-1, 1]."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dexmani_policy.common.config import positive_int


def validate_hand_coordinates(attrs):
    order = ["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"]
    if (
        attrs.get("hand_joint_order", order) != order
        or attrs.get("hand_units", "radians") != "radians"
    ):
        raise ValueError("Hand data must use XHand thumb/index/middle/ring/pinky order in radians")


def codebook_chunk_size(num_codes: int, requested: int) -> int:
    """Budget 16 MiB for an FP32 distance matrix, always allowing at least one row."""
    positive_int(requested, "chunk_size")
    return min(requested, max(1, (16 * 1024 * 1024) // (num_codes * 4)))


def codebook_distances(values: Tensor, prototypes: Tensor) -> Tensor:
    """Direct FP32 distances preserve nearest indices for nearly coincident poses."""
    with torch.autocast(device_type=values.device.type, enabled=False):
        return torch.cdist(
            values.float(), prototypes.float(), compute_mode="donot_use_mm_for_euclid_dist",
        )


class HandStateCodebook(nn.Module):
    """Store the ordered normalized prototypes used by the policy."""

    def __init__(
        self,
        hand_dim: int,
        num_groups: int = 2,
        codebook_size: int = 4,
    ) -> None:
        super().__init__()
        self.hand_dim = positive_int(hand_dim, "hand_dim")
        self.num_groups = positive_int(num_groups, "num_groups")
        self.codebook_size = positive_int(codebook_size, "codebook_size")
        self.register_buffer("sorted_hand_poses", torch.empty(self.num_codes, self.hand_dim))

    @property
    def num_codes(self) -> int:
        return self.codebook_size**self.num_groups

    @torch.no_grad()
    def set_prototypes(self, poses: Tensor) -> None:
        if poses.shape != self.sorted_hand_poses.shape:
            raise ValueError(f"Hand prototypes require shape {tuple(self.sorted_hand_poses.shape)}")
        if poses.is_complex() or poses.dtype == torch.bool:
            raise ValueError("Hand prototypes must contain real numeric values")
        poses = poses.detach().to(self.sorted_hand_poses)
        if not torch.isfinite(poses).all():
            raise ValueError("Hand prototypes must be finite in runtime dtype")
        self.sorted_hand_poses.copy_(poses)

    @torch.no_grad()
    def encode(self, hand: Tensor) -> tuple[Tensor, Tensor]:
        """Map normalized states to continuous and discrete nearest-prototype indices."""
        if hand.ndim == 0 or hand.shape[-1] != self.hand_dim:
            raise ValueError(f"Expected hand width {self.hand_dim}, got shape {tuple(hand.shape)}")
        hand = hand.float()
        if not torch.isfinite(hand).all():
            raise ValueError("Hand states must be finite before encoding")
        distances = codebook_distances(hand.reshape(-1, self.hand_dim), self.sorted_hand_poses)
        indices = distances.argmin(-1).reshape(hand.shape[:-1])
        continuous = indices.float() / max(self.num_codes - 1, 1) * 2.0 - 1.0
        return continuous.unsqueeze(-1), indices

    def decode(self, continuous: Tensor) -> tuple[Tensor, Tensor]:
        """Round continuous indices half-up before looking up normalized hand states."""
        if not torch.isfinite(continuous).all():
            raise ValueError("Continuous hand indices must be finite before quantization")
        scaled = (continuous.clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.num_codes - 1)
        indices = torch.floor(scaled + 0.5).long().clamp(0, self.num_codes - 1)
        return self.sorted_hand_poses[indices], indices

    def extra_repr(self) -> str:
        return f"hand_dim={self.hand_dim}, num_codes={self.num_codes}"
