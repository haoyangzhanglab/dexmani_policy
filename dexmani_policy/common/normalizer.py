from collections.abc import Iterable

import torch
from torch import nn

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES


def normalization_feature_shapes(dims):
    return {
        key: FIELD_FEATURE_SHAPES.get(key, (width,))
        for key, width in dims.items()
    }


class FieldNormalizer(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))

    def normalize(self, value) -> torch.Tensor:
        scale, offset = self.scale, self.offset
        value = torch.as_tensor(value, dtype=scale.dtype, device=scale.device)
        return (value * scale).add_(offset)

    def unnormalize(self, value) -> torch.Tensor:
        scale, offset = self.scale, self.offset
        value = torch.as_tensor(value, dtype=scale.dtype, device=scale.device)
        return (value - offset).div_(scale)


class LinearNormalizer(nn.Module):
    def __init__(self, feature_shapes=None):
        super().__init__()
        self.fields = nn.ModuleDict()
        for key, shape in (feature_shapes or {}).items():
            self.fields[key] = FieldNormalizer(torch.ones(shape), torch.zeros(shape))

    @torch.no_grad()
    def fit_field(self, key: str, chunks: Iterable, *, mode="limits", last_n_dims=1):
        """Merge chunk statistics in FP64, preserving the trailing feature axes."""
        if mode not in {"limits", "gaussian", "rms", "max_abs"}:
            raise ValueError(f"Unknown normalization mode: {mode}")

        count = 0
        sum_squares = torch.zeros((), dtype=torch.float64)
        max_abs = torch.zeros((), dtype=torch.float64)
        for chunk in chunks:
            values = torch.as_tensor(chunk, dtype=torch.float64)
            if not values.numel():
                continue
            feature_shape = values.shape[-last_n_dims:]
            values = values.reshape(-1, feature_shape.numel())
            size = len(values)
            if mode in {"rms", "max_abs"}:
                sum_squares += values.square().sum()
                max_abs = torch.maximum(max_abs, values.abs().max())
                size = values.numel()
            elif mode == "limits":
                chunk_min, chunk_max = torch.aminmax(values, dim=0)
                if count == 0:
                    low, high = chunk_min, chunk_max
                else:
                    low, high = torch.minimum(low, chunk_min), torch.maximum(high, chunk_max)
            else:
                variance, chunk_mean = torch.var_mean(values, dim=0, correction=0)
                chunk_m2 = variance * size
                if count == 0:
                    mean, m2 = chunk_mean, chunk_m2
                else:
                    delta = chunk_mean - mean
                    total = count + size
                    m2 = m2 + chunk_m2 + delta.square() * (count * size / total)
                    mean = mean + delta * (size / total)
            count += size
            del values, chunk

        if not count or (mode == "gaussian" and count < 2):
            raise ValueError(f"Insufficient samples for {key} with mode={mode}")

        if mode in {"rms", "max_abs"}:
            rms = (sum_squares / count).sqrt()
            magnitude = rms if mode == "rms" else max_abs
            scalar = torch.ones_like(magnitude) if magnitude == 0 else magnitude.reciprocal()
            scale = scalar.float().expand(feature_shape).clone()
            offset = torch.zeros_like(scale)
        elif mode == "limits":
            low, high = low.float().reshape(feature_shape), high.float().reshape(feature_shape)
            span = high - low
            constant = span < 1e-4
            scale = 2 / span.masked_fill(constant, 2)
            offset = -1 - scale * low
            offset[constant] = -low[constant]
        else:
            mean = mean.float().reshape(feature_shape)
            std = (m2 / (count - 1)).sqrt().float().reshape(feature_shape)
            scale = std.masked_fill(std < 1e-4, 1).reciprocal()
            offset = -mean * scale
        valid = torch.isfinite(scale) & torch.isfinite(offset) & (scale != 0)
        if not valid.all():
            raise ValueError(f"Cannot represent finite, invertible affine parameters for {key!r}")
        self.fields[key] = FieldNormalizer(scale, offset)
        if mode in {"rms", "max_abs"}:
            ratio = "n/a (all zero)" if rms == 0 else f"{max_abs / rms:.6g}"
            print(f"[Normalizer] {key}: {mode}, scale={scalar:.6g}, RMS={rms:.6g}, max/RMS={ratio}")

    def __getitem__(self, key) -> FieldNormalizer:
        return self.fields[key]


def validate_normalizer_state(normalizer: LinearNormalizer) -> None:
    for key, field in normalizer.fields.items():
        valid = torch.isfinite(field.scale) & torch.isfinite(field.offset) & (field.scale != 0)
        if not valid.all():
            raise ValueError(f"Invalid affine parameters for {key}: expected finite, nonzero scale")
