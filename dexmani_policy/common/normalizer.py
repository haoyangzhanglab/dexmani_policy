from collections.abc import Iterable
from math import prod

import torch
from torch import nn

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, TACTILE_FEATURE_SHAPES


def normalization_feature_shapes(dims, spec):
    """Configuration defines required affine fields, including direct buffer consumers."""
    return {
        key: FIELD_FEATURE_SHAPES.get(key, (width,))
        for key, width in dims.items()
        if spec[key] != "identity" or key == "action" or key in TACTILE_FEATURE_SHAPES
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
    def fit_field(self, key: str, chunks: Iterable) -> None:
        """Fit per-feature limits from streaming training chunks."""
        limits = None
        for chunk in chunks:
            values = torch.as_tensor(chunk)
            if not values.is_floating_point():
                values = values.float()
            if not values.numel():
                continue
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
        valid = torch.isfinite(scale) & torch.isfinite(offset) & (scale != 0)
        if not valid.all():
            raise ValueError(f"Cannot represent finite, invertible affine parameters for {key!r}")
        self.fields[key] = FieldNormalizer(scale, offset)

    def __getitem__(self, key) -> FieldNormalizer:
        return self.fields[key]


def validate_normalizer_state(normalizer: LinearNormalizer) -> None:
    for key, field in normalizer.fields.items():
        valid = torch.isfinite(field.scale) & torch.isfinite(field.offset) & (field.scale != 0)
        if not valid.all():
            raise ValueError(f"Invalid affine parameters for {key}: expected finite, nonzero scale")
