"""PCA-ordered hand states with continuous indices in [-1, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
        num_codes: int,
    ) -> None:
        super().__init__()
        self.hand_dim = positive_int(hand_dim, "hand_dim")
        self.num_codes = positive_int(num_codes, "num_codes")
        self.register_buffer("sorted_hand_poses", torch.empty(self.num_codes, self.hand_dim, dtype=torch.float32))

    @torch.no_grad()
    def encode(self, hand: Tensor) -> Tensor:
        """Map normalized states to continuous nearest-prototype indices."""
        distances = codebook_distances(hand.reshape(-1, self.hand_dim), self.sorted_hand_poses)
        indices = distances.argmin(-1).reshape(hand.shape[:-1])
        continuous = indices.float() / max(self.num_codes - 1, 1) * 2.0 - 1.0
        return continuous.unsqueeze(-1)

    def decode(self, continuous: Tensor) -> Tensor:
        """Round continuous indices half-up before looking up normalized hand states."""
        if not torch.isfinite(continuous).all():
            raise ValueError("Continuous hand indices must be finite before quantization")
        scaled = (continuous.float().clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.num_codes - 1)
        indices = torch.floor(scaled + 0.5).long().clamp(0, self.num_codes - 1)
        return self.sorted_hand_poses[indices]

    def extra_repr(self) -> str:
        return f"hand_dim={self.hand_dim}, num_codes={self.num_codes}"


def load_codebook(path: str | Path) -> tuple[HandStateCodebook, dict[str, Tensor]]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        poses = torch.from_numpy(data["sorted_hand_poses"])
        affine = {
            key: torch.from_numpy(data[f"hand_normalizer_{key}"])
            for key in ("scale", "offset")
        }
    if poses.ndim != 2 or poses.shape[1] != 12:
        raise ValueError("Hand codebook requires sorted_hand_poses shape [K,12]")
    if poses.is_complex() or poses.dtype == torch.bool:
        raise ValueError("Hand prototypes must contain real numeric values")
    poses = poses.float()
    if not torch.isfinite(poses).all():
        raise ValueError("Hand prototypes must be finite in runtime dtype")
    codebook = HandStateCodebook(hand_dim=poses.shape[1], num_codes=poses.shape[0])
    for key, values in affine.items():
        if values.shape != (codebook.hand_dim,) or values.is_complex() or values.dtype == torch.bool:
            raise ValueError(f"Hand normalizer {key} requires {codebook.hand_dim} real values")
        affine[key] = values.float()
    if any(not torch.isfinite(value).all() for value in affine.values()) or (affine["scale"] == 0).any():
        raise ValueError("Hand normalizer must be finite and invertible in runtime dtype")
    codebook.sorted_hand_poses.copy_(poses)
    return codebook, affine
