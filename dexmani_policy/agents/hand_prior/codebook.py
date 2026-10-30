"""PCA-ordered hand states with continuous indices in [-1, 1]."""

from __future__ import annotations

from collections.abc import Mapping

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


def validate_dq_normalization(spec):
    """DQ/VQ hand prototypes require the existing bounded action coordinates."""
    if spec.get("action") not in {"auto", "limits"}:
        raise ValueError("DQ/VQ hand prototypes require action normalization auto or limits")


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
    """Store normalized prototypes and their Policy normalization as persistent buffers."""

    BUFFER_NAMES = (
        "sorted_hand_poses",
        "pca_permutation",
        "layer_weights",
        "hand_normalizer_scale",
        "hand_normalizer_offset",
    )

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
        self.register_buffer("sorted_hand_poses", torch.empty(0, hand_dim))
        self.register_buffer("pca_permutation", torch.empty(0, dtype=torch.long))
        self.register_buffer("layer_weights", torch.empty(0))
        self.register_buffer("hand_normalizer_scale", torch.empty(0))
        self.register_buffer("hand_normalizer_offset", torch.empty(0))

    @property
    def num_codes(self) -> int:
        return self.codebook_size**self.num_groups

    @property
    def is_loaded(self) -> bool:
        return self.sorted_hand_poses.numel() > 0

    def prepare_buffers(self, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Validate normalized poses before replacing runtime buffers."""
        expected = {
            "sorted_hand_poses": (self.num_codes, self.hand_dim),
            "pca_permutation": (self.num_codes,),
            "layer_weights": (self.num_groups,),
            "hand_normalizer_scale": (self.hand_dim,),
            "hand_normalizer_offset": (self.hand_dim,),
        }
        if values.keys() != expected.keys():
            raise ValueError(f"Codebook requires exactly these buffers: {tuple(expected)}")
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(values[name].shape)}")
            if values[name].is_complex() or values[name].dtype == torch.bool:
                raise ValueError(f"{name} must contain real numeric values")
        permutation = values["pca_permutation"]
        if permutation.is_floating_point():
            raise ValueError("pca_permutation must contain integer indices")
        expected_order = torch.arange(self.num_codes, device=permutation.device)
        if not torch.equal(permutation.long().sort().values, expected_order):
            raise ValueError("pca_permutation must cover every pose exactly once")
        converted = {
            name: values[name].detach().to(getattr(self, name), copy=True)
            for name in self.BUFFER_NAMES
        }
        if any(not torch.isfinite(value).all() for value in converted.values()):
            raise ValueError("Codebook buffers must be finite in runtime dtype")
        if (converted["hand_normalizer_scale"] == 0).any():
            raise ValueError("Hand normalizer scale must be nonzero")
        return converted

    def set_buffers(self, values: Mapping[str, Tensor]) -> None:
        for name, value in self.prepare_buffers(values).items():
            setattr(self, name, value)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prepared = dict(state_dict)
        try:
            values = {name: prepared[prefix + name] for name in self.BUFFER_NAMES}
            values = self.prepare_buffers(values)
        except (KeyError, ValueError, RuntimeError) as exc:
            error_msgs.append(f"Invalid hand codebook: {exc}")
            return
        for name, value in values.items():
            setattr(self, name, torch.empty_like(value))
        prepared.update({prefix + name: value for name, value in values.items()})
        super()._load_from_state_dict(
            prepared,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        return f"hand_dim={self.hand_dim}, num_codes={self.num_codes}, loaded={self.is_loaded}"
