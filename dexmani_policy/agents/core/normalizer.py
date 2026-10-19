"""Affine observation and action normalization with streaming feature statistics."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from dexmani_policy.utils.config import FINGER_VECTOR_FIELDS

logger = logging.getLogger(__name__)


def restore_parameters(state_dict: Mapping[str, torch.Tensor], prefix: str) -> nn.ParameterDict:
    params = nn.ParameterDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            parts = key[len(prefix) :].split(".")
            node = params
            for name in parts[:-1]:
                if name not in node:
                    node[name] = nn.ParameterDict()
                node = node[name]
            node[parts[-1]] = nn.Parameter(value.detach().clone(), requires_grad=False)
    return params


class NormalizerState(nn.Module):
    """Restore named affine parameters without changing checkpoint tensor paths."""

    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__()
        self.params_dict = nn.ParameterDict() if params_dict is None else params_dict
        self.field_views = {}
        self.input_stats = {}

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        previous = set(self.params_dict.state_dict())
        parameter_prefix = prefix + "params_dict."
        self.params_dict = restore_parameters(state_dict, parameter_prefix)
        self.field_views.clear()
        self.input_stats.clear()
        # First load establishes the fields; later loads must preserve their structure.
        if previous:
            incoming = set(self.params_dict.state_dict())
            missing_keys.extend(parameter_prefix + key for key in sorted(previous - incoming))
            unexpected_keys.extend(parameter_prefix + key for key in sorted(incoming - previous))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def affine_parameters(
    stats: dict[str, torch.Tensor],
    *,
    mode: str,
    output_min: float,
    output_max: float,
    range_eps: float,
    fit_offset: bool,
    label: str,
) -> nn.ParameterDict:
    low, high, mean, std = (stats[key] for key in ("min", "max", "mean", "std"))
    if mode == "limits":
        if fit_offset:
            span = high - low
            constant = span < range_eps
            span = span.masked_fill(constant, output_max - output_min)
            scale = (output_max - output_min) / span
            offset = output_min - scale * low
            offset[constant] = -mean[constant]
        else:
            bound = min(abs(output_min), abs(output_max))
            magnitude = torch.maximum(low.abs(), high.abs())
            constant = magnitude < range_eps
            scale = bound / magnitude.masked_fill(constant, bound)
            offset = torch.zeros_like(mean)
    else:
        constant = std < range_eps
        scale = std.masked_fill(constant, 1).reciprocal()
        offset = -mean * scale if fit_offset else torch.zeros_like(mean)

    if logger.isEnabledFor(logging.INFO) and constant.any():
        logger.info(
            "%s: unit scale for near-constant features %s",
            label,
            constant.nonzero().flatten().tolist(),
        )
    return nn.ParameterDict(
        {
            "scale": nn.Parameter(scale, requires_grad=False),
            "offset": nn.Parameter(offset, requires_grad=False),
        }
    )


def affine_transform(
    value: Any, params: nn.ParameterDict, *, inverse: bool = False
) -> torch.Tensor:
    value = torch.as_tensor(value)
    scale, offset = params["scale"], params["offset"]
    if value.shape[-scale.ndim :] != scale.shape:
        raise ValueError(
            f"Expected trailing feature shape {tuple(scale.shape)}, got {tuple(value.shape)}"
        )
    scale, offset = scale.to(value.device), offset.to(value.device)
    value = value.to(scale.dtype)
    return (value - offset) / scale if inverse else value * scale + offset


class FieldNormalizer(NormalizerState):
    """Affine transform for one feature field; outputs are never clipped."""

    @classmethod
    def from_parameters(
        cls, scale: Any, offset: Any, input_stats: dict | None = None
    ) -> FieldNormalizer:
        scale, offset = torch.as_tensor(scale), torch.as_tensor(offset)
        if (
            scale.ndim != 1
            or not scale.numel()
            or scale.shape != offset.shape
            or not torch.isfinite(scale).all()
            or not torch.isfinite(offset).all()
            or (scale == 0).any()
        ):
            raise ValueError("Affine parameters require matching finite vectors and nonzero scale")
        obj = cls(
            nn.ParameterDict(
                {
                    "scale": nn.Parameter(scale, requires_grad=False),
                    "offset": nn.Parameter(offset, requires_grad=False),
                }
            )
        )
        if input_stats is not None:
            obj.input_stats = {key: torch.as_tensor(value) for key, value in input_stats.items()}
        return obj

    def normalize(self, value: Any) -> torch.Tensor:
        return affine_transform(value, self.params_dict)

    def unnormalize(self, value: Any) -> torch.Tensor:
        return affine_transform(value, self.params_dict, inverse=True)

    def forward(self, value: Any) -> torch.Tensor:
        return self.normalize(value)


class AffineNormalizer(NormalizerState):
    """Named affine transforms with independent per-finger statistics."""

    @torch.no_grad()
    def fit_field(
        self,
        key: str,
        chunks: Iterable,
        *,
        dtype: torch.dtype | None = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        """Merge chunk statistics in FP64 using Welford's algorithm."""
        if mode not in {"limits", "gaussian"} or output_max <= output_min:
            raise ValueError("Fitting requires limits/gaussian mode and output_max > output_min")
        dtype = torch.float32 if dtype is None else dtype
        count = 0
        width = None
        mean = m2 = low = high = None
        for chunk in chunks:
            values = torch.as_tensor(chunk).to(torch.float64)
            if key in FINGER_VECTOR_FIELDS:
                if values.shape[-2:] != (5, 3):
                    raise ValueError(f"{key} requires trailing feature shape (5, 3)")
                values = values.flatten(-2)
            if values.ndim == 0:
                raise ValueError("Fitting requires a trailing feature axis")
            if not values.numel():
                continue
            if width is None:
                width = values.shape[-1]
                mean = values.new_zeros(width)
                m2 = torch.zeros_like(mean)
            elif values.shape[-1] != width:
                raise ValueError(f"Inconsistent feature width for {key}: expected {width}")
            values = values.reshape(-1, width)
            size = len(values)
            chunk_min, chunk_max = values.min(0).values, values.max(0).values
            low = chunk_min if low is None else torch.minimum(low, chunk_min)
            high = chunk_max if high is None else torch.maximum(high, chunk_max)
            chunk_mean = values.mean(0)
            chunk_m2 = (values - chunk_mean).square().sum(0)
            delta = chunk_mean - mean
            total = count + size
            m2 = m2 + chunk_m2 + delta.square() * (count * size / total)
            mean = mean + delta * (size / total)
            count = total
            del values, chunk

        if not count or (mode == "gaussian" and count < 2):
            raise ValueError(f"Insufficient samples for {key} with mode={mode}")
        stats = {
            "min": low.to(dtype),
            "max": high.to(dtype),
            "mean": mean.to(dtype),
            "std": (m2 / max(count - 1, 1)).sqrt().to(dtype),
        }
        params = affine_parameters(
            stats,
            mode=mode,
            output_min=output_min,
            output_max=output_max,
            range_eps=range_eps,
            fit_offset=fit_offset,
            label=key,
        )
        if key in FINGER_VECTOR_FIELDS:
            params = nn.ParameterDict(
                {
                    name: nn.Parameter(value.reshape(5, 3), requires_grad=False)
                    for name, value in params.items()
                }
            )
            stats = {name: value.reshape(5, 3) for name, value in stats.items()}
        self.params_dict[key] = params
        self.input_stats[key] = stats
        self.field_views.pop(key, None)

    def __getitem__(self, key: str) -> FieldNormalizer:
        if key not in self.field_views:
            view = FieldNormalizer(self.params_dict[key])
            view.input_stats = self.input_stats.get(key, {})
            self.field_views[key] = view
        return self.field_views[key]

    def __setitem__(self, key: str, value: FieldNormalizer) -> None:
        self.params_dict[key] = value.params_dict
        self.input_stats[key] = value.input_stats
        self.field_views.pop(key, None)


def validate_normalizer_state(
    normalizer: AffineNormalizer, spec: dict[str, str], feature_dims: dict[str, int]
) -> None:
    """Check the saved affine state against declared fields and feature layouts."""
    expected = {key for key, mode in spec.items() if mode != "identity"}
    if set(normalizer.params_dict) != expected:
        raise ValueError(f"Normalizer fields must match non-identity fields: {sorted(expected)}")
    for key in expected:
        params = normalizer.params_dict[key]
        if set(params) != {"scale", "offset"}:
            raise ValueError(f"{key} requires scale and offset")
        scale, offset = params["scale"], params["offset"]
        width = feature_dims[key]
        shape = (5, 3) if key in FINGER_VECTOR_FIELDS else (width,)
        if (
            scale.shape != shape
            or offset.shape != shape
            or scale.numel() != width
            or not scale.numel()
            or not torch.isfinite(scale).all()
            or not torch.isfinite(offset).all()
            or (scale == 0).any()
        ):
            raise ValueError(
                f"Invalid affine parameters for {key}: expected finite {shape} and nonzero scale"
            )
