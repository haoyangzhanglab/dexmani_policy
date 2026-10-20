"""PCA-ordered hand states with continuous indices in [-1, 1]."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from dexmani_policy.utils.config import positive_int


class HandStateCodebook(nn.Module):
    """Store affine-raw prototypes and their Policy normalization as persistent buffers."""

    FORMAT_VERSION = 4
    BUFFER_NAMES = (
        "hand_min",
        "hand_max",
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
        hand_min: float = 0.0,
        hand_max: float = 65535.0,
    ) -> None:
        super().__init__()
        self.hand_dim = positive_int(hand_dim, "hand_dim")
        self.num_groups = positive_int(num_groups, "num_groups")
        self.codebook_size = positive_int(codebook_size, "codebook_size")
        self.register_buffer("hand_min", torch.tensor(float(hand_min), dtype=torch.float32))
        self.register_buffer("hand_max", torch.tensor(float(hand_max), dtype=torch.float32))
        self.register_buffer("sorted_hand_poses", torch.empty(0, hand_dim))
        self.register_buffer("pca_permutation", torch.empty(0, dtype=torch.long))
        self.register_buffer("layer_weights", torch.empty(0))
        self.register_buffer("hand_normalizer_scale", torch.empty(0))
        self.register_buffer("hand_normalizer_offset", torch.empty(0))
        self.artifact_metadata = {}

    @property
    def num_codes(self) -> int:
        return self.codebook_size**self.num_groups

    @property
    def is_loaded(self) -> bool:
        return self.sorted_hand_poses.numel() > 0

    def prepare_buffers(self, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Validate once at artifact boundaries before replacing runtime buffers."""
        expected = {
            "hand_min": (),
            "hand_max": (),
            "sorted_hand_poses": (self.num_codes, self.hand_dim),
            "pca_permutation": (self.num_codes,),
            "layer_weights": (self.num_groups,),
            "hand_normalizer_scale": (self.hand_dim,),
            "hand_normalizer_offset": (self.hand_dim,),
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(values[name].shape)}")
        permutation = values["pca_permutation"]
        if (
            permutation.dtype == torch.bool
            or permutation.is_floating_point()
            or permutation.is_complex()
        ):
            raise ValueError("pca_permutation must contain integer indices")
        expected_order = torch.arange(self.num_codes, device=permutation.device)
        if not torch.equal(permutation.long().sort().values, expected_order):
            raise ValueError("pca_permutation must cover every pose exactly once")
        converted = {
            name: value.detach().to(getattr(self, name), copy=True)
            for name, value in values.items()
        }
        if any(not torch.isfinite(value).all() for value in converted.values()):
            raise ValueError("Codebook buffers must be finite in runtime dtype")
        span = converted["hand_max"] - converted["hand_min"]
        if not torch.isfinite(span) or span <= 0:
            raise ValueError("hand_min/hand_max must define a finite positive range")
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
        try:
            values = self.prepare_buffers(
                {name: state_dict[prefix + name] for name in self.BUFFER_NAMES}
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            error_msgs.append(f"Invalid hand codebook: {exc}")
            return
        for name, value in values.items():
            setattr(self, name, torch.empty_like(value))
        prepared = dict(state_dict)
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

    def to_affine_raw(self, hand: Tensor) -> Tensor:
        return (hand + 1.0) * 0.5 * (self.hand_max - self.hand_min) + self.hand_min

    def from_affine_raw(self, hand: Tensor) -> Tensor:
        return (hand - self.hand_min) / (self.hand_max - self.hand_min) * 2.0 - 1.0

    @torch.no_grad()
    def encode(self, hand: Tensor) -> tuple[Tensor, Tensor]:
        """Map normalized states to continuous and discrete nearest-prototype indices."""
        if hand.shape[-1] != self.hand_dim:
            raise ValueError(f"Expected hand width {self.hand_dim}, got {hand.shape[-1]}")
        raw = self.to_affine_raw(hand.float())
        distances = (raw.unsqueeze(-2) - self.sorted_hand_poses).square_().sum(-1)
        indices = distances.argmin(-1)
        continuous = indices.float() / max(self.num_codes - 1, 1) * 2.0 - 1.0
        return continuous.unsqueeze(-1), indices

    def decode(self, continuous: Tensor) -> tuple[Tensor, Tensor]:
        """Round continuous indices half-up before looking up normalized hand states."""
        if not torch.isfinite(continuous).all():
            raise ValueError("Continuous hand indices must be finite before quantization")
        scaled = (continuous.clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.num_codes - 1)
        indices = torch.floor(scaled + 0.5).long().clamp(0, self.num_codes - 1)
        return self.from_affine_raw(self.sorted_hand_poses[indices]), indices

    def save(self, path: str | Path) -> None:
        values = self.prepare_buffers({name: getattr(self, name) for name in self.BUFFER_NAMES})
        payload = {name: value.cpu().numpy() for name, value in values.items()}
        metadata = json.dumps(self.artifact_metadata, sort_keys=True, allow_nan=False)
        payload.update(
            format_version=np.asarray(self.FORMAT_VERSION),
            pose_space=np.asarray("affine_raw"),
            hand_dim=np.asarray(self.hand_dim),
            num_groups=np.asarray(self.num_groups),
            codebook_size=np.asarray(self.codebook_size),
            metadata_json=np.asarray(metadata),
        )
        # An open file prevents NumPy from silently appending a second suffix.
        with open(path, "wb") as file:
            np.savez(file, **payload)

    def load(self, path: str | Path) -> None:
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            if (
                data["format_version"].item() != self.FORMAT_VERSION
                or data["pose_space"].item() != "affine_raw"
            ):
                raise ValueError(
                    "Hand codebook requires v4 affine_raw format; regenerate or use historical code"
                )
            dimensions = tuple(
                data[name].item() for name in ("hand_dim", "num_groups", "codebook_size")
            )
            if dimensions != (self.hand_dim, self.num_groups, self.codebook_size):
                raise ValueError(f"Hand codebook dimensions do not match Policy: {dimensions}")
            values = {name: torch.from_numpy(data[name]) for name in self.BUFFER_NAMES}
            metadata = json.loads(data["metadata_json"].item())
        self.set_buffers(values)
        self.artifact_metadata = metadata

    def extra_repr(self) -> str:
        return f"hand_dim={self.hand_dim}, num_codes={self.num_codes}, loaded={self.is_loaded}"
