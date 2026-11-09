from collections.abc import Iterable
from math import prod

import torch
from torch import nn

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, validate_normalization_spec


def normalization_feature_shapes(dims, spec):
    validate_normalization_spec(dims, spec)
    return {
        key: FIELD_FEATURE_SHAPES.get(key, (width,))
        for key, width in dims.items()
        if spec[key] == "limits"
    }


class FieldNormalizer(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))
        self.validate()

    def validate(self):
        if self.scale.shape != self.offset.shape:
            raise ValueError("Affine scale and offset must have identical shapes")
        valid = torch.isfinite(self.scale) & torch.isfinite(self.offset) & (self.scale != 0)
        if not valid.all():
            raise ValueError("Invalid affine parameters: expected finite values and nonzero scale")

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
    def fit_field(self, key: str, chunks: Iterable) -> None:
        """Fit per-feature limits from streaming training chunks."""
        limits = None
        for chunk in chunks:
            values = torch.as_tensor(chunk)
            if not values.is_floating_point():
                values = values.float()
            if not values.numel():
                continue
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite normalization data for {key}")
            feature_shape = FIELD_FEATURE_SHAPES.get(key, values.shape[-1:])
            values = values.reshape(-1, prod(feature_shape))
            chunk_min, chunk_max = torch.aminmax(values, dim=0)
            if limits is not None:
                chunk_min = torch.minimum(limits[0], chunk_min)
                chunk_max = torch.maximum(limits[1], chunk_max)
            limits = chunk_min, chunk_max
            del values, chunk

        if limits is None:
            raise ValueError(f"Insufficient samples for {key} with mode=limits")
        low, high = (value.float().reshape(feature_shape) for value in limits)
        span = high - low
        constant = span < 1e-4
        scale = 2 / span.masked_fill(constant, 2)
        offset = -1 - scale * low
        offset[constant] = -low[constant]
        self.fields[key] = FieldNormalizer(scale, offset)

    def __getitem__(self, key) -> FieldNormalizer:
        return self.fields[key]


def validate_normalizer_state(normalizer: LinearNormalizer) -> None:
    for key, field in normalizer.fields.items():
        try:
            field.validate()
        except ValueError as error:
            raise ValueError(f"Invalid affine field {key}: {error}") from error
