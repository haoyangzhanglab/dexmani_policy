from collections.abc import Iterable

import torch
from torch import nn

from dexmani_policy.common.config import (
    FIXED_OBSERVATION_SHAPES,
    TACTILE_FEATURE_SHAPES,
)


def normalization_feature_shapes(dims, spec, *, required_identity_fields=(), required_limits_fields=()):
    for key in required_limits_fields:
        if spec.get(key) != "limits":
            raise ValueError(f"normalization.{key} must use limits")
    shapes = {}
    identity_fields = {"rgb", *TACTILE_FEATURE_SHAPES, *required_identity_fields}
    for key, width in dims.items():
        mode = spec[key]
        if key == "action" and mode != "limits":
            raise ValueError("action must use limits")
        if mode == "identity":
            continue
        if key in identity_fields:
            raise ValueError(f"{key} must use identity")
        if mode not in {"limits", "mean_std"}:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}")
        shapes[key] = FIXED_OBSERVATION_SHAPES.get(key, (width,))
    return shapes


class FieldNormalizer(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))
        self.validate()

    def validate(self, key="field") -> None:
        if self.scale.shape != self.offset.shape:
            raise ValueError(f"Affine scale and offset for {key} must have identical shapes")
        valid = torch.isfinite(self.scale) & torch.isfinite(self.offset) & (self.scale != 0)
        if not valid.all():
            raise ValueError(f"Invalid affine parameters for {key}: expected finite values and nonzero scale")

    def normalize(self, value) -> torch.Tensor:
        scale, offset = self.scale, self.offset
        value = torch.as_tensor(value, dtype=scale.dtype, device=scale.device)
        return (value * scale).add_(offset)

    def unnormalize(self, value) -> torch.Tensor:
        scale, offset = self.scale, self.offset
        value = torch.as_tensor(value, dtype=scale.dtype, device=scale.device)
        return (value - offset).div_(scale)


def _feature_chunks(key, chunks, check_values):
    feature_shape = None
    for chunk in chunks:
        values = torch.as_tensor(chunk)
        if not values.numel():
            continue
        if check_values and not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite normalization data for {key}")
        if feature_shape is None:
            feature_shape = FIXED_OBSERVATION_SHAPES.get(key, values.shape[-1:])
        if values.shape[-len(feature_shape):] != feature_shape:
            raise ValueError(f"Normalization data for {key} requires trailing shape {tuple(feature_shape)}")
        yield values.reshape(-1, *feature_shape)
    if feature_shape is None:
        raise ValueError(f"Insufficient samples for {key}")


def _fit_limits(chunks):
    extrema = (
        torch.aminmax(values if values.is_floating_point() else values.float(), dim=0)
        for values in chunks
    )
    low, high = next(extrema)
    for chunk_min, chunk_max in extrema:
        low = torch.minimum(low, chunk_min)
        high = torch.maximum(high, chunk_max)
    low, high = low.float(), high.float()
    span = high - low
    constant = span < 1e-4
    scale = 2 / span.masked_fill(constant, 2)
    offset = -1 - scale * low
    offset[constant] = -low[constant]
    return scale, offset


def _fit_mean_std(chunks):
    count = 0
    for values in chunks:
        n = len(values)
        variance, chunk_mean = torch.var_mean(values.double(), dim=0, correction=0)
        if count == 0:
            mean, m2 = chunk_mean, variance * n
        else:
            # Combine central moments, weighting each chunk by its sample count.
            delta = chunk_mean - mean
            m2 += variance * n + delta.square() * (count * n / (count + n))
            mean += delta * (n / (count + n))
        count += n
    std = (m2 / count).sqrt()
    scale = std.masked_fill(std < 1e-4, 1).reciprocal()
    return scale, -mean * scale


class LinearNormalizer(nn.Module):
    def __init__(self, feature_shapes=None):
        super().__init__()
        self.fields = nn.ModuleDict()
        for key, shape in (feature_shapes or {}).items():
            self.fields[key] = FieldNormalizer(torch.ones(shape), torch.zeros(shape))

    @torch.no_grad()
    def fit_field(self, key: str, chunks: Iterable, *, mode: str = "limits", check_values: bool = True) -> None:
        samples = _feature_chunks(key, chunks, check_values)
        if mode == "limits":
            scale, offset = _fit_limits(samples)
        elif mode == "mean_std":
            scale, offset = _fit_mean_std(samples)
        else:
            raise ValueError(f"Cannot fit {key} with mode={mode!r}")
        self.fields[key] = FieldNormalizer(scale, offset)

    def __getitem__(self, key) -> FieldNormalizer:
        return self.fields[key]

    def validate(self) -> None:
        for key, field in self.fields.items():
            field.validate(key)
