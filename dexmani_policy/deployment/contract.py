"""Plain deployment metadata: public raw tensors and private model restore inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEPLOYMENT_FORMAT = "dexmani.deployment.v3"


class DeploymentContractError(ValueError):
    """Raised when the persisted deployment boundary is malformed."""


@dataclass(frozen=True)
class ObservationFieldSpec:
    """Raw per-step array and concrete axis ordering."""

    name: str
    shape: tuple[int | None, ...]
    dtype: str
    ordering: dict[str, tuple] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySpec:
    """Raw observation and physical action boundary exposed to Real."""

    observation_fields: tuple[ObservationFieldSpec, ...]
    n_obs_steps: int
    n_action_steps: int
    control_dt_s: float
    action_mode: str
    joint_names: tuple[str, ...]
    pointcloud_config: dict[str, Any] | None = None

    def __post_init__(self):
        _positive_int(self.n_obs_steps, "n_obs_steps")
        _positive_int(self.n_action_steps, "n_action_steps")
        _positive_float(self.control_dt_s, "control_dt_s")
        if self.action_mode not in {"joint", "eef"}:
            raise DeploymentContractError("action_mode must be joint or eef")
        if (
            len(self.joint_names) != 19
            or any(type(n) is not str or not n for n in self.joint_names)
            or len(set(self.joint_names)) != 19
        ):
            raise DeploymentContractError(
                "joint_names must contain 19 unique ordered names"
            )
        names = [f.name for f in self.observation_fields]
        if not names or len(set(names)) != len(names) or "joint_state" not in names:
            raise DeploymentContractError(
                "observations must be unique and include joint_state"
            )
        if ("point_cloud" in names) != (self.pointcloud_config is not None):
            raise DeploymentContractError(
                "point_cloud requires its trained algorithm config"
            )
        if self.pointcloud_config is not None:
            config = _mapping(self.pointcloud_config, "pointcloud_config")
            cloud = next(f for f in self.observation_fields if f.name == "point_cloud")
            if (
                type(config.get("remove_table")) is not bool
                or _positive_int(config.get("num_points"), "num_points")
                != cloud.shape[0]
            ):
                raise DeploymentContractError(
                    "pointcloud config conflicts with the raw tensor"
                )

    @classmethod
    def from_dict(cls, value):
        value = _mapping(value, "policy_spec")
        names = value.get("joint_names")
        if not isinstance(names, list):
            raise DeploymentContractError("joint_names must be a list")
        return cls(
            observation_fields=_observation_fields(value.get("observations")),
            n_obs_steps=value.get("n_obs_steps"),
            n_action_steps=value.get("n_action_steps"),
            control_dt_s=value.get("control_dt_s"),
            action_mode=value.get("action_mode"),
            joint_names=tuple(names),
            pointcloud_config=value.get("pointcloud_config"),
        )

    def to_dict(self):
        return {
            "observations": {
                f.name: {
                    "shape": list(f.shape),
                    "dtype": f.dtype,
                    "ordering": {k: list(v) for k, v in f.ordering.items()},
                }
                for f in self.observation_fields
            },
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "control_dt_s": self.control_dt_s,
            "action_mode": self.action_mode,
            "joint_names": list(self.joint_names),
            "pointcloud_config": self.pointcloud_config,
        }


@dataclass(frozen=True)
class RgbPreprocessingSpec:
    """Exact validation-spatial and ImageProcessor chain for raw RGB."""

    input_layout: str
    input_dtype: str
    input_color_order: str
    input_value_range: tuple[float, float]
    execution_device: str
    resize_hw: tuple[int, int] | None
    center_crop_hw: tuple[int, int] | None
    interpolation: str
    antialias: bool
    output_layout: str
    output_dtype: str
    scale: float
    output_value_range: tuple[float, float]
    processor_image_size_hw: tuple[int, int] | None
    processor_interpolation: str
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]


@dataclass(frozen=True)
class DeploymentSpec:
    """Policy-private model restoration and preprocessing inputs."""

    policy_spec: PolicySpec
    action_key: str
    action_dim: int
    horizon: int
    inference_steps: int
    rgb_preprocessing: RgbPreprocessingSpec | None
    normalization: dict[str, Any]
    agent_config: dict[str, Any]
    warmup_rgb_hw: tuple[int, int] | None

    @property
    def control_action_dim(self):
        return 21 if self.policy_spec.action_mode == "eef" else 19

    @property
    def observation_fields(self):
        return self.policy_spec.observation_fields

    @property
    def n_obs_steps(self):
        return self.policy_spec.n_obs_steps

    @property
    def n_action_steps(self):
        return self.policy_spec.n_action_steps


def deployment_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    root = _mapping(payload, "deployment payload")
    if root.get("_format") != DEPLOYMENT_FORMAT:
        raise DeploymentContractError(
            f"unsupported deployment format: {root.get('_format')!r}"
        )
    contract = _mapping(root.get("contract"), "payload.contract")
    for name in ("inference_config", "policy_spec", "producer"):
        _mapping(contract.get(name), f"contract.{name}")
    if not _mapping(root.get("weights"), "payload.weights"):
        raise DeploymentContractError("deployment weights must not be empty")
    return contract


def parse_deployment_contract(payload: Mapping[str, Any]) -> DeploymentSpec:
    contract = deployment_contract(payload)
    public = PolicySpec.from_dict(contract["policy_spec"])
    inference = contract["inference_config"]
    action_key = inference.get("action_key")
    expected_key = "action_ee" if public.action_mode == "eef" else "action"
    if action_key != expected_key:
        raise DeploymentContractError(
            "private action layout conflicts with physical action_mode"
        )
    action_dim = _positive_int(inference.get("action_dim"), "action_dim")
    horizon = _positive_int(inference.get("horizon"), "horizon")
    if public.n_obs_steps - 1 + public.n_action_steps > horizon:
        raise DeploymentContractError("observation/action window exceeds horizon")
    if action_dim < (21 if public.action_mode == "eef" else 19):
        raise DeploymentContractError("action_dim is smaller than physical action")
    fields = public.observation_fields
    preprocessing = _rgb_preprocessing(inference.get("rgb_preprocessing"), fields)
    warmup_hw = _optional_hw(inference.get("warmup_rgb_hw"), "warmup_rgb_hw")
    if preprocessing is not None and warmup_hw is None:
        raise DeploymentContractError("RGB warmup requires sample dimensions")
    return DeploymentSpec(
        policy_spec=public,
        action_key=action_key,
        action_dim=action_dim,
        horizon=horizon,
        inference_steps=_positive_int(
            inference.get("eval", {}).get("inference_steps"), "inference_steps"
        ),
        rgb_preprocessing=preprocessing,
        normalization=_normalization_spec(inference.get("normalization"), fields),
        agent_config=_agent_config(inference.get("agent")),
        warmup_rgb_hw=warmup_hw,
    )


def _normalization_spec(
    value: Any, fields: tuple[ObservationFieldSpec, ...]
) -> dict[str, Any]:
    from dexmani_policy.common.checkpoint_io import parse_normalization_contract

    if type(value) is not dict:
        raise DeploymentContractError(
            "inference_config.normalization must be a plain mapping"
        )
    declared_fields = {field.name for field in fields}
    try:
        spec = parse_normalization_contract(value, observation_fields=declared_fields)
    except ValueError as exc:
        raise DeploymentContractError(f"invalid normalization contract: {exc}") from exc
    return spec


def _agent_config(value: Any) -> dict[str, Any]:
    if type(value) is not dict or not value:
        raise DeploymentContractError(
            "inference_config.agent must be a non-empty plain mapping"
        )
    if type(value.get("_target_")) is not str:
        raise DeploymentContractError("inference_config.agent requires a _target_")
    return value


def _observation_fields(value: Any) -> tuple[ObservationFieldSpec, ...]:
    fields = _mapping(value, "observations")
    result = []
    for name, value in fields.items():
        _string(name, "observation name")
        value = _mapping(value, name)
        shape, dtype = value.get("shape"), value.get("dtype")
        if name == "rgb":
            if shape != [None, None, 3] or dtype != "uint8":
                raise DeploymentContractError(
                    "raw rgb must be uint8 HWC with variable H/W"
                )
        elif (
            not isinstance(shape, list)
            or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
            or dtype != "float32"
        ):
            raise DeploymentContractError(f"invalid fixed numerical observation {name}")
        ordering = _mapping(value.get("ordering", {}), f"{name}.ordering")
        if any(
            not isinstance(v, list)
            or not v
            or any(type(x) not in (str, int) for x in v)
            for v in ordering.values()
        ):
            raise DeploymentContractError(
                f"{name}.ordering must contain concrete lists"
            )
        result.append(
            ObservationFieldSpec(
                name, tuple(shape), dtype, {k: tuple(v) for k, v in ordering.items()}
            )
        )
    return tuple(result)


def _rgb_preprocessing(
    value: Any, fields: tuple[ObservationFieldSpec, ...]
) -> RgbPreprocessingSpec | None:
    rgb_fields = tuple(field for field in fields if field.name == "rgb")
    if not rgb_fields:
        if value is not None:
            raise DeploymentContractError("rgb_preprocessing requires an rgb field")
        return None
    metadata = _mapping(value, "inference_config.rgb_preprocessing")
    required = {
        "input_layout",
        "input_dtype",
        "input_color_order",
        "input_value_range",
        "execution_device",
        "resize_hw",
        "center_crop_hw",
        "interpolation",
        "antialias",
        "output_layout",
        "output_dtype",
        "scale",
        "output_value_range",
        "processor_image_size_hw",
        "processor_interpolation",
        "normalize_mean",
        "normalize_std",
    }
    if set(metadata) != required:
        raise DeploymentContractError(
            "rgb_preprocessing must contain the complete RGB chain"
        )
    result = RgbPreprocessingSpec(
        input_layout=_string(metadata.get("input_layout"), "input_layout"),
        input_dtype=_string(metadata.get("input_dtype"), "input_dtype"),
        input_color_order=_string(
            metadata.get("input_color_order"), "input_color_order"
        ),
        input_value_range=_numeric_pair(
            metadata.get("input_value_range"), "input_value_range"
        ),
        execution_device=_string(metadata.get("execution_device"), "execution_device"),
        resize_hw=_optional_hw(metadata.get("resize_hw"), "resize_hw"),
        center_crop_hw=_optional_hw(metadata.get("center_crop_hw"), "center_crop_hw"),
        interpolation=_string(metadata.get("interpolation"), "interpolation"),
        antialias=_bool(metadata.get("antialias"), "antialias"),
        output_layout=_string(metadata.get("output_layout"), "output_layout"),
        output_dtype=_string(metadata.get("output_dtype"), "output_dtype"),
        scale=_positive_float(metadata.get("scale"), "scale"),
        output_value_range=_numeric_pair(
            metadata.get("output_value_range"), "output_value_range"
        ),
        processor_image_size_hw=_optional_hw(
            metadata.get("processor_image_size_hw"), "processor_image_size_hw"
        ),
        processor_interpolation=_string(
            metadata.get("processor_interpolation"), "processor_interpolation"
        ),
        normalize_mean=_vector(metadata.get("normalize_mean"), "normalize_mean"),
        normalize_std=_vector(metadata.get("normalize_std"), "normalize_std"),
    )
    _validate_rgb_chain(result, rgb_fields[0])
    return result


def _validate_rgb_chain(
    preprocessing: RgbPreprocessingSpec, field: ObservationFieldSpec
) -> None:
    if (
        field.dtype != "uint8"
        or len(field.shape) != 3
        or field.shape[-1] != 3
        or preprocessing.input_layout != "HWC"
        or preprocessing.input_dtype != "uint8"
        or preprocessing.input_color_order != "rgb"
        or preprocessing.input_value_range != (0.0, 255.0)
        or preprocessing.execution_device != "cpu"
    ):
        raise DeploymentContractError(
            "rgb_preprocessing input semantics conflict with the rgb field"
        )
    if preprocessing.resize_hw is None:
        expected_output = ("HWC", "uint8", 1.0, (0.0, 255.0))
        if preprocessing.center_crop_hw is not None:
            raise DeploymentContractError("RGB center crop requires dataset resize")
    else:
        if preprocessing.interpolation != "bilinear" or not preprocessing.antialias:
            raise DeploymentContractError(
                "validation RGB resize must use bilinear antialiasing"
            )
        if preprocessing.output_dtype == "uint8":
            expected_output = ("CHW", "uint8", 1.0, (0.0, 255.0))
        else:
            expected_output = ("CHW", "float32", 1.0 / 255.0, (0.0, 1.0))
    actual_output = (
        preprocessing.output_layout,
        preprocessing.output_dtype,
        preprocessing.scale,
        preprocessing.output_value_range,
    )
    if actual_output != expected_output:
        raise DeploymentContractError(
            "rgb_preprocessing output semantics conflict with validation RGB"
        )
    if preprocessing.processor_interpolation not in {
        "nearest",
        "bilinear",
        "bicubic",
    }:
        raise DeploymentContractError("unsupported ImageProcessor interpolation")


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


def _vector(value: Any, label: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise DeploymentContractError(f"{label} must contain three numbers")
    return tuple(_finite_float(item, label) for item in value)  # type: ignore[return-value]


__all__ = [
    "DEPLOYMENT_FORMAT",
    "DeploymentContractError",
    "DeploymentSpec",
    "PolicySpec",
    "ObservationFieldSpec",
    "RgbPreprocessingSpec",
    "deployment_contract",
    "parse_deployment_contract",
]
