from collections.abc import Iterable

import torch
from torch import nn

from dexmani_policy.common.config import FINGER_VECTOR_FIELDS


class FieldNormalizer(nn.Module):
    def __init__(self, scale, offset, input_stats=None):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))
        self.input_stats = {} if input_stats is None else input_stats

    def normalize(self, value):
        scale, offset = self.scale, self.offset
        value = torch.as_tensor(value, dtype=scale.dtype, device=scale.device)
        return value * scale + offset

    def unnormalize(self, value):
        scale, offset = self.scale, self.offset
        value = torch.as_tensor(value, dtype=scale.dtype, device=scale.device)
        return (value - offset) / scale

    def forward(self, value):
        return self.normalize(value)


class LinearNormalizer(nn.Module):
    def __init__(self, feature_dims=None):
        super().__init__()
        self.fields = nn.ModuleDict()
        for key, width in (feature_dims or {}).items():
            shape = (5, 3) if key in FINGER_VECTOR_FIELDS else (width,)
            self.fields[key] = FieldNormalizer(torch.ones(shape), torch.zeros(shape))

    @torch.no_grad()
    def fit_field(self, key: str, chunks: Iterable, *, mode="limits"):
        # Merge in FP64; store affine parameters in FP32 without clipping outputs.
        if mode not in {"limits", "gaussian"}:
            raise ValueError(f"Unknown normalization mode: {mode}")
        count = 0
        mean: torch.Tensor
        m2: torch.Tensor
        low: torch.Tensor
        high: torch.Tensor
        for index, chunk in enumerate(chunks):
            values = torch.as_tensor(chunk, dtype=torch.float64)
            if key in FINGER_VECTOR_FIELDS:
                if values.shape[-2:] != (5, 3):
                    raise ValueError(f"{key} requires trailing feature shape (5, 3)")
                values = values.flatten(-2)
            if not values.numel():
                continue
            values = values.reshape(-1, values.shape[-1])
            size = len(values)
            chunk_min, chunk_max = torch.aminmax(values, dim=0)
            if not (torch.isfinite(chunk_min).all() and torch.isfinite(chunk_max).all()):
                raise ValueError(f"Nonfinite normalization field {key!r}, chunk {index}")
            variance, chunk_mean = torch.var_mean(values, dim=0, correction=0)
            chunk_m2 = variance * size
            if count == 0:
                low, high, mean, m2 = chunk_min, chunk_max, chunk_mean, chunk_m2
            else:
                low, high = torch.minimum(low, chunk_min), torch.maximum(high, chunk_max)
                delta = chunk_mean - mean
                total = count + size
                m2 = m2 + chunk_m2 + delta.square() * (count * size / total)
                mean = mean + delta * (size / total)
            count += size
            del values, chunk

        if not count or (mode == "gaussian" and count < 2):
            raise ValueError(f"Insufficient samples for {key} with mode={mode}")
        stats = {
            "min": low.float(),
            "max": high.float(),
            "mean": mean.float(),
            "std": (m2 / max(count - 1, 1)).sqrt().float(),
        }
        if key in FINGER_VECTOR_FIELDS:
            stats = {name: value.reshape(5, 3) for name, value in stats.items()}
        if mode == "limits":
            span = stats["max"] - stats["min"]
            constant = span < 1e-4
            scale = 2 / span.masked_fill(constant, 2)
            offset = -1 - scale * stats["min"]
            offset[constant] = -stats["min"][constant]
        else:
            std = stats["std"]
            scale = std.masked_fill(std < 1e-4, 1).reciprocal()
            offset = -stats["mean"] * scale
        if not (torch.isfinite(scale).all() and torch.isfinite(offset).all()) or (scale == 0).any():
            raise ValueError(f"Cannot represent finite, invertible affine parameters for {key!r}")
        self.fields[key] = FieldNormalizer(scale, offset, stats)

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, value):
        self.fields[key] = value


def validate_normalizer_state(normalizer):
    # Strict state loading already checks field names and shapes.
    for key, field in normalizer.fields.items():
        if (
            not torch.isfinite(field.scale).all()
            or not torch.isfinite(field.offset).all()
            or (field.scale == 0).any()
        ):
            raise ValueError(f"Invalid affine parameters for {key}: expected finite, nonzero scale")
