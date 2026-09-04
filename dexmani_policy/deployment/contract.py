"""Canonical deployment artifact contract shared by export and runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEPLOYMENT_FORMAT = "dexmani.deployment.v2"
DEPLOYMENT_SCHEMA_VERSION = 2
SUPPORTED_OBSERVATION_DTYPES = frozenset({"float32", "uint8"})


class DeploymentContractError(ValueError):
    """Raised when the persisted deployment boundary is malformed."""


@dataclass(frozen=True)
class FrozenMetadata(Mapping[str, Any]):
    """Recursively immutable, pickle-safe field metadata."""

    entries: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class ObservationFieldSpec:
    """One raw model input before Policy-owned preprocessing."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    semantics: Mapping[str, Any]


@dataclass(frozen=True)
class RgbPreprocessingSpec:
    """Deterministic Policy-owned preprocessing for raw RGB frames."""

    input_color_order: str
    input_value_range: tuple[float, float]
    resize_hw: tuple[int, int] | None
    center_crop_hw: tuple[int, int] | None
    interpolation: str
    antialias: bool
    output_layout: str
    output_dtype: str
    scale: float
    normalize_mean: tuple[float, float, float] | None
    normalize_std: tuple[float, float, float] | None


@dataclass(frozen=True)
class DeploymentSpec:
    """Parsed inference and observation contract for one artifact."""

    action_key: str
    action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    denoise_steps: int
    observation_fields: tuple[ObservationFieldSpec, ...]
    control_dt_s: float
    requires_hand: bool
    rgb_preprocessing: RgbPreprocessingSpec | None

    @property
    def control_action_dim(self) -> int:
        return 21 if self.action_key == "action_ee" else 19


def deployment_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sole persisted contract of one canonical v2 artifact."""
    root = _mapping(payload, "deployment payload")
    if root.get("_format") != DEPLOYMENT_FORMAT:
        raise DeploymentContractError(
            f"unsupported deployment format: {root.get('_format')!r}"
        )
    if set(root) != {"_format", "contract", "weights"}:
        raise DeploymentContractError(
            "deployment payload must contain format, contract, and weights"
        )
    contract = _mapping(root.get("contract"), "payload.contract")
    if contract.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION:
        raise DeploymentContractError(
            f"unsupported deployment schema version: {contract.get('schema_version')!r}"
        )
    for name in ("inference_config", "data_contract", "producer"):
        _mapping(contract.get(name), f"contract.{name}")
    if not _mapping(root.get("weights"), "payload.weights"):
        raise DeploymentContractError("deployment weights must not be empty")
    return contract


def parse_deployment_contract(payload: Mapping[str, Any]) -> DeploymentSpec:
    """Parse the model-facing portion of one canonical v2 contract."""
    contract = deployment_contract(payload)
    inference = _mapping(contract.get("inference_config"), "contract.inference_config")
    data = _mapping(contract.get("data_contract"), "contract.data_contract")
    eval_config = _mapping(inference.get("eval"), "inference_config.eval")

    action_key = inference.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise DeploymentContractError("action_key must be 'action' or 'action_ee'")
    action_dim = _positive_int(inference.get("action_dim"), "action_dim")
    horizon = _positive_int(inference.get("horizon"), "horizon")
    n_obs_steps = _positive_int(inference.get("n_obs_steps"), "n_obs_steps")
    n_action_steps = _positive_int(inference.get("n_action_steps"), "n_action_steps")
    if n_obs_steps - 1 + n_action_steps > horizon:
        raise DeploymentContractError("observation/action window exceeds horizon")
    control_dim = 21 if action_key == "action_ee" else 19
    if action_dim < control_dim:
        raise DeploymentContractError("action_dim is smaller than the control action")

    fields = _observation_fields(data.get("observation_fields"))
    preprocessing = _rgb_preprocessing(inference.get("rgb_preprocessing"), fields)
    control_dt_s = _positive_float(data.get("dt"), "data_contract.dt")
    requires_hand = data.get("requires_hand")
    if type(requires_hand) is not bool:
        raise DeploymentContractError("data_contract.requires_hand must be bool")
    return DeploymentSpec(
        action_key=action_key,
        action_dim=action_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        denoise_steps=_positive_int(eval_config.get("denoise_steps"), "denoise_steps"),
        observation_fields=fields,
        control_dt_s=control_dt_s,
        requires_hand=requires_hand,
        rgb_preprocessing=preprocessing,
    )


def _observation_fields(value: Any) -> tuple[ObservationFieldSpec, ...]:
    fields = _mapping(value, "data_contract.observation_fields")
    if not fields:
        raise DeploymentContractError("observation_fields must not be empty")
    result: list[ObservationFieldSpec] = []
    for name, value in fields.items():
        if type(name) is not str or not name:
            raise DeploymentContractError("observation field names must be non-empty")
        field = _mapping(value, f"observation_fields.{name}")
        shape = field.get("shape")
        if (
            type(shape) is not list
            or not shape
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            raise DeploymentContractError(
                f"observation_fields.{name}.shape must contain positive ints"
            )
        dtype = field.get("dtype")
        if dtype not in SUPPORTED_OBSERVATION_DTYPES:
            raise DeploymentContractError(
                f"unsupported observation dtype for {name!r}: {dtype!r}"
            )
        semantics = field.get("semantics", {})
        result.append(
            ObservationFieldSpec(
                name=name,
                shape=tuple(shape),
                dtype=dtype,
                semantics=_freeze_metadata(
                    _mapping(semantics, f"observation_fields.{name}.semantics")
                ),
            )
        )
    return tuple(result)


def _rgb_preprocessing(
    value: Any, fields: tuple[ObservationFieldSpec, ...]
) -> RgbPreprocessingSpec | None:
    has_rgb = any(field.name == "rgb" for field in fields)
    if not has_rgb:
        if value is not None:
            raise DeploymentContractError("rgb_preprocessing requires an rgb field")
        return None
    metadata = _mapping(value, "inference_config.rgb_preprocessing")
    mean = _optional_vector(metadata.get("normalize_mean"), "normalize_mean")
    std = _optional_vector(metadata.get("normalize_std"), "normalize_std")
    if (mean is None) != (std is None):
        raise DeploymentContractError("RGB normalization needs both mean and std")
    return RgbPreprocessingSpec(
        input_color_order=_string(
            metadata.get("input_color_order"), "input_color_order"
        ),
        input_value_range=_numeric_pair(
            metadata.get("input_value_range"), "input_value_range"
        ),
        resize_hw=_optional_hw(metadata.get("resize_hw"), "resize_hw"),
        center_crop_hw=_optional_hw(metadata.get("center_crop_hw"), "center_crop_hw"),
        interpolation=_string(metadata.get("interpolation"), "interpolation"),
        antialias=_bool(metadata.get("antialias"), "antialias"),
        output_layout=_string(metadata.get("output_layout"), "output_layout"),
        output_dtype=_string(metadata.get("output_dtype"), "output_dtype"),
        scale=_positive_float(metadata.get("scale"), "scale"),
        normalize_mean=mean,
        normalize_std=std,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentContractError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DeploymentContractError(f"{label} must be a positive int")
    return value


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise DeploymentContractError(f"{label} must be positive")
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeploymentContractError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise DeploymentContractError(f"{label} must be finite")
    return result


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise DeploymentContractError(f"{label} must be a non-empty string")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise DeploymentContractError(f"{label} must be bool")
    return value


def _optional_hw(value: Any, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise DeploymentContractError(f"{label} must be [H, W] or null")
    return value[0], value[1]


def _numeric_pair(value: Any, label: str) -> tuple[float, float]:
    if type(value) is not list or len(value) != 2:
        raise DeploymentContractError(f"{label} must contain two numbers")
    result = (_finite_float(value[0], label), _finite_float(value[1], label))
    if result[0] >= result[1]:
        raise DeploymentContractError(f"{label} must be increasing")
    return result


def _optional_vector(value: Any, label: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if type(value) is not list or len(value) != 3:
        raise DeploymentContractError(f"{label} must contain three numbers")
    return tuple(_finite_float(item, label) for item in value)  # type: ignore[return-value]


def _freeze_metadata(value: Mapping[str, Any]) -> FrozenMetadata:
    return FrozenMetadata(
        tuple((key, _freeze_value(item)) for key, item in value.items())
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if type(value) is list:
        return tuple(_freeze_value(item) for item in value)
    return value


__all__ = [
    "DEPLOYMENT_FORMAT",
    "DEPLOYMENT_SCHEMA_VERSION",
    "DeploymentContractError",
    "DeploymentSpec",
    "FrozenMetadata",
    "ObservationFieldSpec",
    "RgbPreprocessingSpec",
    "deployment_contract",
    "parse_deployment_contract",
]
