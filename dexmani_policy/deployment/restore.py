"""Strict model and normalizer restore, Policy preprocessing and physical predictions."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.normalizer import validate_normalizer_state
from dexmani_policy.deployment.contract import (
    DeploymentContractError,
    DeploymentSpec,
    ObservationFieldSpec,
    parse_deployment_contract,
)


class DeploymentRestoreError(RuntimeError):
    """Raised when a deployment artifact cannot be restored safely."""


@dataclass(frozen=True)
class RestoredDeployment:
    """A strictly restored inference agent and its artifact dimensions."""

    agent: Any
    spec: DeploymentSpec


def reset_inference_seed(seed: int) -> None:
    """Reset every RNG used by Policy inference, including CUDA when present."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative int")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deployment_spec(payload: Mapping[str, Any]) -> DeploymentSpec:
    """Extract the one canonical deployment contract."""
    try:
        return parse_deployment_contract(payload)
    except DeploymentContractError as exc:
        raise DeploymentRestoreError(f"invalid deployment contract: {exc}") from exc


def deterministic_observation(
    spec: DeploymentSpec,
    *,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a bounded, non-zero synthetic deployment observation.

    The values are arithmetic rather than sampled, so the observation itself
    is reproducible without consuming an RNG stream. Floating-point values lie
    in ``(-1, 1)`` so accidental all-zero handling cannot make verification
    vacuous; raw RGB uses deterministic nonzero ``uint8`` values.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    result: dict[str, torch.Tensor] = {}
    for field in spec.observation_fields:
        raw_shape = (*spec.warmup_rgb_hw, 3) if field.name == "rgb" else field.shape
        shape = (batch_size, spec.n_obs_steps, *raw_shape)
        if field.dtype == "float32":
            result[field.name] = _bounded_nonzero_values(
                shape, device=device, dtype=torch.float32
            )
        elif field.dtype == "uint8":
            values = torch.arange(math.prod(shape), device=device, dtype=torch.int64)
            result[field.name] = (
                values.remainder(251).add(1).to(torch.uint8).reshape(shape)
            )
        else:  # Parsed contracts cannot reach this branch.
            raise DeploymentRestoreError(
                f"unsupported observation dtype for {field.name!r}"
            )
    return result


def restore_deployment_agent(
    payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
) -> RestoredDeployment:
    """Instantiate the saved model and strictly restore its weights and normalizer."""
    spec = deployment_spec(payload)
    agent_config = spec.agent_config
    selected_state = _state_dict(payload.get("weights"), "payload.weights")

    try:
        import hydra

        agent = hydra.utils.instantiate(OmegaConf.create(agent_config))
        agent.action_key = spec.action_key
        agent.load_state_dict(selected_state, strict=True)
        agent.to(device)
        agent.eval()
        _validate_agent_dimensions(agent, spec)
        validate_deployment_normalizer(agent, spec)
        agent.normalization_spec = dict(spec.normalization)
        validate_normalizer_state(agent.normalizer, agent.normalization_spec)
        _validate_consumed_observation_fields(agent, spec)
        _validate_rgb_processor(agent, spec)
    except DeploymentRestoreError:
        raise
    except Exception as exc:
        raise DeploymentRestoreError("deployment agent strict restore failed") from exc
    return RestoredDeployment(agent=agent, spec=spec)


def prepare_deployment_observation(
    observation: Mapping[str, torch.Tensor], spec: DeploymentSpec
) -> dict[str, torch.Tensor]:
    """Validate raw inputs and reproduce validation RGB before ImageProcessor."""
    raw = dict(observation)
    _validate_observation(raw, spec)
    preprocessing = spec.rgb_preprocessing
    if preprocessing is not None:
        from dexmani_policy.datasets.base_dataset import preprocess_validation_rgb

        input_device = raw["rgb"].device
        raw["rgb"] = preprocess_validation_rgb(
            raw["rgb"].cpu(),
            resize_hw=preprocessing.resize_hw,
            center_crop_hw=preprocessing.center_crop_hw,
            keep_uint8=preprocessing.output_dtype == "uint8",
        ).to(input_device)
    return raw


def validate_deployment_normalizer(agent: Any, spec: DeploymentSpec) -> None:
    """Require finite, non-degenerate affine normalizers for all runtime inputs."""
    normalizer = getattr(agent, "normalizer", None)
    if normalizer is None:
        raise DeploymentRestoreError("deployment agent has no normalizer")
    params_dict = getattr(normalizer, "params_dict", None)
    if params_dict is None:
        raise DeploymentRestoreError("normalizer has no params_dict")
    try:
        actual_keys = set(params_dict.keys())
    except (AttributeError, TypeError) as exc:
        raise DeploymentRestoreError("normalizer keys are not inspectable") from exc
    fields = {field.name: field for field in spec.observation_fields}
    if "action" not in actual_keys or actual_keys - {"action", *fields}:
        raise DeploymentRestoreError("normalizer contains fields outside the contract")
    expected_dims = {
        key: spec.action_dim if key == "action" else fields[key].shape[-1]
        for key in actual_keys
    }
    for key, expected_dim in expected_dims.items():
        try:
            params = params_dict[key]
            scale = params.get("scale")
            offset = params.get("offset")
        except Exception as exc:
            raise DeploymentRestoreError(
                f"normalizer {key!r} has invalid deployment state"
            ) from exc
        if (
            not torch.is_tensor(scale)
            or not torch.is_tensor(offset)
            or scale.numel() != expected_dim
            or offset.numel() != expected_dim
            or not bool(torch.isfinite(scale).all())
            or not bool(torch.isfinite(offset).all())
            or bool(torch.any(scale == 0))
        ):
            raise DeploymentRestoreError(
                f"normalizer {key!r} has invalid deployment state"
            )


def validate_prediction(
    result: Mapping[str, Any], spec: DeploymentSpec, *, batch_size: int
) -> torch.Tensor:
    """Validate tensor shape, finiteness, and the canonical control slice."""
    if not isinstance(result, Mapping):
        raise DeploymentRestoreError("predict_action must return a mapping")
    pred = result.get("pred_action")
    control = result.get("control_action")
    if not isinstance(pred, torch.Tensor) or not isinstance(control, torch.Tensor):
        raise DeploymentRestoreError(
            "predict_action must return tensor pred_action and control_action"
        )
    if not pred.is_floating_point() or not control.is_floating_point():
        raise DeploymentRestoreError("predictions must be real floating-point tensors")
    expected_pred = (batch_size, spec.horizon, spec.action_dim)
    expected_control = (
        batch_size,
        spec.n_action_steps,
        spec.control_action_dim,
    )
    if tuple(pred.shape) != expected_pred:
        raise DeploymentRestoreError(
            f"pred_action shape mismatch: got {tuple(pred.shape)}, expected {expected_pred}"
        )
    if tuple(control.shape) != expected_control:
        raise DeploymentRestoreError(
            "control_action shape mismatch: "
            f"got {tuple(control.shape)}, expected {expected_control}"
        )
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(control).all()):
        raise DeploymentRestoreError("prediction contains NaN/Inf")
    start = spec.n_obs_steps - 1
    expected_control_slice = pred[
        :, start : start + spec.n_action_steps, : spec.control_action_dim
    ]
    if not torch.equal(control, expected_control_slice):
        raise DeploymentRestoreError(
            "control_action is not the exact canonical pred_action slice"
        )
    return control.detach()


def _state_dict(value: Any, label: str) -> dict[str, torch.Tensor]:
    if type(value) is not dict or not value:
        raise DeploymentRestoreError(f"{label} must be a non-empty plain state_dict")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if (
            type(key) is not str
            or not key
            or not torch.is_tensor(tensor)
            or key.startswith("module.")
            or "_orig_mod." in key
        ):
            raise DeploymentRestoreError(
                f"{label} must contain canonical string tensor keys"
            )
        result[key] = tensor
    return result


def _validate_agent_dimensions(agent: Any, spec: DeploymentSpec) -> None:
    expected = {
        "action_dim": spec.action_dim,
        "horizon": spec.horizon,
        "n_obs_steps": spec.n_obs_steps,
        "n_action_steps": spec.n_action_steps,
    }
    for name, wanted in expected.items():
        actual = getattr(agent, name, None)
        if actual is not None and actual != wanted:
            raise DeploymentRestoreError(
                f"restored agent.{name}={actual!r} conflicts with artifact={wanted!r}"
            )
    control_dim = getattr(agent, "control_action_dim", None)
    if control_dim is not None and control_dim != spec.control_action_dim:
        raise DeploymentRestoreError(
            "restored agent.control_action_dim="
            f"{control_dim!r} conflicts with artifact={spec.control_action_dim!r}"
        )


def _validate_observation(
    observation: Mapping[str, torch.Tensor], spec: DeploymentSpec
) -> None:
    if not isinstance(observation, Mapping):
        raise DeploymentRestoreError("deployment observation must be a mapping")
    expected_names = {field.name for field in spec.observation_fields}
    if set(observation) != expected_names:
        raise DeploymentRestoreError(
            "deployment observation fields mismatch: "
            f"got {sorted(observation)}, expected {sorted(expected_names)}"
        )
    batch_size: int | None = None
    for field in spec.observation_fields:
        value = observation[field.name]
        if not torch.is_tensor(value):
            raise DeploymentRestoreError(
                f"deployment observation {field.name!r} must be a tensor"
            )
        expected_dtype = _torch_dtype(field)
        if value.dtype != expected_dtype:
            raise DeploymentRestoreError(
                f"deployment observation {field.name!r} dtype mismatch: "
                f"got {value.dtype}, expected {expected_dtype}"
            )
        if value.ndim < 2:
            raise DeploymentRestoreError(
                f"deployment observation {field.name!r} lacks batch/time axes"
            )
        current_batch = int(value.shape[0])
        if current_batch < 1:
            raise DeploymentRestoreError("observation batch size must be positive")
        if batch_size is None:
            batch_size = current_batch
        elif current_batch != batch_size:
            raise DeploymentRestoreError("deployment observation batch sizes differ")
        expected_shape = (current_batch, spec.n_obs_steps, *field.shape)
        if value.ndim != len(expected_shape) or any(
            actual <= 0 or (expected is not None and actual != expected)
            for actual, expected in zip(value.shape, expected_shape)
        ):
            raise DeploymentRestoreError(
                f"deployment observation {field.name!r} shape mismatch: "
                f"got {tuple(value.shape)}, expected {expected_shape}"
            )
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            raise DeploymentRestoreError(
                f"deployment observation {field.name!r} contains NaN/Inf"
            )


def _torch_dtype(field: ObservationFieldSpec) -> torch.dtype:
    if field.dtype == "float32":
        return torch.float32
    if field.dtype == "uint8":
        return torch.uint8
    raise DeploymentRestoreError(
        f"unsupported deployment observation dtype: {field.dtype!r}"
    )


def _validate_consumed_observation_fields(agent: Any, spec: DeploymentSpec) -> None:
    """Require the restored encoder to consume the artifact fields exactly."""
    try:
        consumed = agent.obs_encoder.consumed_observation_fields
    except Exception as exc:
        raise DeploymentRestoreError(
            "deployment requires agent.obs_encoder.consumed_observation_fields"
        ) from exc
    expected = tuple(field.name for field in spec.observation_fields)
    if (
        type(consumed) is not tuple
        or not consumed
        or any(type(name) is not str or not name for name in consumed)
        or consumed != expected
    ):
        raise DeploymentRestoreError(
            "deployment observation_fields do not match "
            "the restored agent consumer contract"
        )


def _validate_rgb_processor(agent: Any, spec: DeploymentSpec) -> None:
    """Require the restored ImageProcessor to match the recorded second stage."""
    preprocessing = spec.rgb_preprocessing
    if preprocessing is None:
        return
    try:
        processor = agent.obs_encoder.image_processor
        image_size = processor.image_size
        interpolation = processor.interpolation
        mean = processor.image_mean
        std = processor.image_std
    except Exception as exc:
        raise DeploymentRestoreError(
            "deployment RGB requires agent.obs_encoder.image_processor"
        ) from exc
    if (
        _optional_hw_tuple(image_size) != preprocessing.processor_image_size_hw
        or interpolation != preprocessing.processor_interpolation
    ):
        raise DeploymentRestoreError(
            "deployment RGB preprocessing conflicts with agent ImageProcessor"
        )
    _validate_rgb_stats(mean, preprocessing.normalize_mean, "image_mean")
    _validate_rgb_stats(std, preprocessing.normalize_std, "image_std")


def _optional_hw_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise DeploymentRestoreError("agent ImageProcessor size is invalid")
    return value


def _validate_rgb_stats(
    value: Any,
    expected: tuple[float, float, float],
    label: str,
) -> None:
    if not torch.is_tensor(value) or tuple(value.shape) != (3,):
        raise DeploymentRestoreError(
            f"deployment RGB preprocessing {label} conflicts with agent ImageProcessor"
        )
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    if not torch.equal(value.detach().cpu().to(dtype=torch.float32), expected_tensor):
        raise DeploymentRestoreError(
            f"deployment RGB preprocessing {label} conflicts with agent ImageProcessor"
        )


def _bounded_nonzero_values(
    shape: tuple[int, ...], *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    # 97 is prime and the half-integer centre cannot be reached by an integer
    # residue; no element is exactly zero, while the full signal is in (-1, 1).
    values = torch.arange(math.prod(shape), device=device, dtype=dtype)
    return (
        values.mul(17).add(1).remainder(97).add(0.5).div(49.0).sub(1.0).reshape(shape)
    )
