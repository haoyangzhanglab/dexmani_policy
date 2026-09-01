"""Strict, deterministic restore and parity helpers for deployment-v2.

This module deliberately knows only the frozen deployment checkpoint payload.
It does not read training checkpoints, datasets, or ``dexmani_real``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

_DEPLOYMENT_FORMAT = "dexmani.deployment.v2"
_REQUIRED_NORMALIZER_KEYS = ("action", "joint_state", "point_cloud")
_JOINT_STATE_DIM = 19
_MAX_PARITY_TOLERANCE = 1e-5


class DeploymentRestoreError(RuntimeError):
    """Raised when a deployment-v2 artifact cannot be restored safely."""


class PredictionParityError(DeploymentRestoreError):
    """Raised when two deterministic deployment predictions differ."""


@dataclass(frozen=True)
class DeploymentSpec:
    """Inference dimensions extracted from one deployment-v2 payload."""

    action_key: str
    action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    denoise_steps: int
    point_cloud_num_points: int
    point_cloud_feature_dim: int

    @property
    def control_action_dim(self) -> int:
        """The currently frozen deployment action-space control dimension."""
        return 21 if self.action_key == "action_ee" else 19


@dataclass(frozen=True)
class RestoredDeployment:
    """A strictly restored inference agent and its artifact dimensions."""

    agent: Any
    spec: DeploymentSpec


@dataclass(frozen=True)
class PredictionSnapshot:
    """Immutable CPU copies of the two deployment contract outputs."""

    pred_action: torch.Tensor
    control_action: torch.Tensor


def reset_inference_seed(seed: int) -> None:
    """Reset every RNG used by Policy inference, including CUDA when present."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative int")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# A concise alias for callers that do not distinguish train and inference seeding.
seed_everything = reset_inference_seed


def deployment_spec(payload: Mapping[str, Any]) -> DeploymentSpec:
    """Extract and validate the subset of deployment-v2 used for restore."""
    if not isinstance(payload, Mapping):
        raise DeploymentRestoreError("deployment payload must be a mapping")
    artifact_format = payload.get("_format")
    if artifact_format != _DEPLOYMENT_FORMAT:
        raise DeploymentRestoreError(
            f"unsupported deployment checkpoint format: {artifact_format!r}"
        )

    state = _mapping(payload.get("state"), "payload.state")
    inference = _mapping(state.get("inference_config"), "state.inference_config")
    data = _mapping(state.get("data_contract"), "state.data_contract")
    eval_config = _mapping(inference.get("eval"), "inference_config.eval")

    action_key = inference.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise DeploymentRestoreError(
            "inference_config.action_key must be 'action' or 'action_ee'"
        )
    action_dim = _positive_int(inference.get("action_dim"), "action_dim")
    horizon = _positive_int(inference.get("horizon"), "horizon")
    n_obs_steps = _positive_int(inference.get("n_obs_steps"), "n_obs_steps")
    n_action_steps = _positive_int(inference.get("n_action_steps"), "n_action_steps")
    if n_obs_steps - 1 + n_action_steps > horizon:
        raise DeploymentRestoreError(
            "n_obs_steps - 1 + n_action_steps exceeds deployment horizon"
        )
    control_dim = 21 if action_key == "action_ee" else 19
    if action_dim < control_dim:
        raise DeploymentRestoreError(
            f"action_dim={action_dim} is smaller than control_action_dim={control_dim}"
        )

    return DeploymentSpec(
        action_key=action_key,
        action_dim=action_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        denoise_steps=_positive_int(eval_config.get("denoise_steps"), "denoise_steps"),
        point_cloud_num_points=_positive_int(
            data.get("point_cloud_num_points"), "point_cloud_num_points"
        ),
        point_cloud_feature_dim=_positive_int(
            data.get("point_cloud_feature_dim"), "point_cloud_feature_dim"
        ),
    )


def deterministic_observation(
    spec: DeploymentSpec,
    *,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Create a bounded, non-zero synthetic point-cloud observation.

    The values are arithmetic rather than sampled, so the observation itself
    is reproducible without consuming an RNG stream.  Every value lies in
    ``(-1, 1)``; a point cloud with accidental all-zero handling therefore
    cannot make exporter verification vacuous.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")

    joint_shape = (batch_size, spec.n_obs_steps, _JOINT_STATE_DIM)
    point_shape = (
        batch_size,
        spec.n_obs_steps,
        spec.point_cloud_num_points,
        spec.point_cloud_feature_dim,
    )
    return {
        "joint_state": _bounded_nonzero_values(joint_shape, device=device, dtype=dtype),
        "point_cloud": _bounded_nonzero_values(point_shape, device=device, dtype=dtype),
    }


def restore_deployment_agent(
    payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
) -> RestoredDeployment:
    """Instantiate a deployment-v2 agent and load the selected weights strictly."""
    spec = deployment_spec(payload)
    state = _mapping(payload.get("state"), "payload.state")
    inference = _mapping(state.get("inference_config"), "state.inference_config")
    agent_config = _mapping(inference.get("agent"), "inference_config.agent")
    weights = _mapping(payload.get("weights"), "payload.weights")
    eval_config = _mapping(inference.get("eval"), "inference_config.eval")
    use_ema = eval_config.get("use_ema")
    if type(use_ema) is not bool:
        raise DeploymentRestoreError("inference_config.eval.use_ema must be bool")

    selected_name = "ema_model" if use_ema else "model"
    selected = weights.get(selected_name)
    if selected is None and use_ema:
        raise DeploymentRestoreError("eval.use_ema requires EMA weights")
    selected_state = _state_dict(selected, f"weights.{selected_name}")

    try:
        import hydra

        agent = hydra.utils.instantiate(OmegaConf.create(dict(agent_config)))
        agent.action_key = spec.action_key
        agent.load_state_dict(selected_state, strict=True)
        agent.to(device)
        agent.eval()
        _validate_agent_dimensions(agent, spec)
        validate_deployment_normalizer(agent, spec)
    except DeploymentRestoreError:
        raise
    except Exception as exc:
        raise DeploymentRestoreError(
            f"deployment agent strict restore failed using weights.{selected_name}"
        ) from exc
    return RestoredDeployment(agent=agent, spec=spec)


# Explicit name for consumers that use the artifact format in their API.
restore_deployment_v2 = restore_deployment_agent


def prediction_snapshot(
    restored: RestoredDeployment,
    *,
    seed: int = 0,
    observation: Mapping[str, torch.Tensor] | None = None,
) -> PredictionSnapshot:
    """Run one seeded prediction and validate the complete output contract."""
    obs = (
        deterministic_observation(restored.spec)
        if observation is None
        else dict(observation)
    )
    _validate_observation(obs, restored.spec)
    reset_inference_seed(seed)
    try:
        with torch.inference_mode():
            result = restored.agent.predict_action(
                obs, denoise_timesteps=restored.spec.denoise_steps
            )
    except Exception as exc:
        raise DeploymentRestoreError("deployment agent prediction failed") from exc
    return validate_prediction(
        result, restored.spec, batch_size=obs["joint_state"].shape[0]
    )


def verify_deployment_prediction(
    payload: Mapping[str, Any], *, seed: int = 0
) -> PredictionSnapshot:
    """Strictly restore a deployment-v2 artifact and run one contract prediction."""
    return prediction_snapshot(restore_deployment_agent(payload), seed=seed)


def validate_deployment_normalizer(agent: Any, spec: DeploymentSpec) -> None:
    """Require finite, non-degenerate affine normalizers for all runtime inputs."""
    normalizer = getattr(agent, "normalizer", None)
    if normalizer is None:
        raise DeploymentRestoreError("deployment agent has no normalizer")
    try:
        fitted = normalizer.is_fitted(required_keys=list(_REQUIRED_NORMALIZER_KEYS))
    except Exception as exc:
        raise DeploymentRestoreError("cannot inspect deployment normalizer") from exc
    if not fitted:
        raise DeploymentRestoreError("normalizer is missing required deployment state")

    expected_dims = {
        "action": spec.action_dim,
        "joint_state": _JOINT_STATE_DIM,
        "point_cloud": spec.point_cloud_feature_dim,
    }
    params_dict = getattr(normalizer, "params_dict", None)
    if params_dict is None:
        raise DeploymentRestoreError("normalizer has no params_dict")
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
) -> PredictionSnapshot:
    """Validate tensor shape, finiteness, and the canonical control slice."""
    if not isinstance(result, Mapping):
        raise DeploymentRestoreError("predict_action must return a mapping")
    pred = result.get("pred_action")
    control = result.get("control_action")
    if not isinstance(pred, torch.Tensor) or not isinstance(control, torch.Tensor):
        raise DeploymentRestoreError(
            "predict_action must return tensor pred_action and control_action"
        )
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
    return PredictionSnapshot(
        pred_action=pred.detach().cpu().clone(),
        control_action=control.detach().cpu().clone(),
    )


def assert_prediction_parity(
    reference: PredictionSnapshot,
    candidate: PredictionSnapshot,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Compare both deployment outputs with exact-by-default narrow tolerance."""
    _validate_tolerance(atol, "atol")
    _validate_tolerance(rtol, "rtol")
    for name in ("pred_action", "control_action"):
        reference_tensor = getattr(reference, name)
        candidate_tensor = getattr(candidate, name)
        if not torch.is_tensor(reference_tensor) or not torch.is_tensor(
            candidate_tensor
        ):
            raise TypeError("prediction snapshots must contain tensors")
        if tuple(reference_tensor.shape) != tuple(candidate_tensor.shape):
            raise PredictionParityError(
                f"{name} shape mismatch: {tuple(reference_tensor.shape)} != "
                f"{tuple(candidate_tensor.shape)}"
            )
        if reference_tensor.dtype != candidate_tensor.dtype:
            raise PredictionParityError(
                f"{name} dtype mismatch: {reference_tensor.dtype} != "
                f"{candidate_tensor.dtype}"
            )
        if not bool(torch.isfinite(reference_tensor).all()) or not bool(
            torch.isfinite(candidate_tensor).all()
        ):
            raise PredictionParityError(f"{name} contains NaN/Inf")
        if atol == 0.0 and rtol == 0.0:
            matches = torch.equal(reference_tensor, candidate_tensor)
        else:
            matches = torch.allclose(
                reference_tensor, candidate_tensor, atol=atol, rtol=rtol
            )
        if not matches:
            max_abs_error = torch.max(
                torch.abs(reference_tensor - candidate_tensor)
            ).item()
            raise PredictionParityError(
                f"{name} parity mismatch (max_abs_error={max_abs_error:.9g}, "
                f"atol={atol}, rtol={rtol})"
            )


def compare_prediction_snapshots(
    reference: PredictionSnapshot,
    candidate: PredictionSnapshot,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Alias retaining a descriptive name for direct/export parity tests."""
    assert_prediction_parity(reference, candidate, atol=atol, rtol=rtol)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentRestoreError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise DeploymentRestoreError(f"{label} must be a positive int")
    return value


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
    try:
        joint_state = observation["joint_state"]
        point_cloud = observation["point_cloud"]
    except KeyError as exc:
        raise DeploymentRestoreError(
            "observation must include joint_state and point_cloud"
        ) from exc
    if not torch.is_tensor(joint_state) or not torch.is_tensor(point_cloud):
        raise DeploymentRestoreError("deployment observation values must be tensors")
    expected_joint = (joint_state.shape[0], spec.n_obs_steps, _JOINT_STATE_DIM)
    expected_point = (
        joint_state.shape[0],
        spec.n_obs_steps,
        spec.point_cloud_num_points,
        spec.point_cloud_feature_dim,
    )
    if tuple(joint_state.shape) != expected_joint:
        raise DeploymentRestoreError(
            "joint_state observation shape mismatch: "
            f"got {tuple(joint_state.shape)}, expected {expected_joint}"
        )
    if tuple(point_cloud.shape) != expected_point:
        raise DeploymentRestoreError(
            "point_cloud observation shape mismatch: "
            f"got {tuple(point_cloud.shape)}, expected {expected_point}"
        )
    if joint_state.shape[0] < 1:
        raise DeploymentRestoreError("observation batch size must be positive")
    if not bool(torch.isfinite(joint_state).all()) or not bool(
        torch.isfinite(point_cloud).all()
    ):
        raise DeploymentRestoreError("deployment observation contains NaN/Inf")


def _bounded_nonzero_values(
    shape: tuple[int, ...], *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    # 97 is prime and the half-integer centre cannot be reached by an integer
    # residue; no element is exactly zero, while the full signal is in (-1, 1).
    values = torch.arange(math.prod(shape), device=device, dtype=dtype)
    return (
        values.mul(17).add(1).remainder(97).add(0.5).div(49.0).sub(1.0).reshape(shape)
    )


def _validate_tolerance(value: float, label: str) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    if value > _MAX_PARITY_TOLERANCE:
        raise ValueError(
            f"{label}={value} exceeds the narrow deployment parity limit "
            f"({_MAX_PARITY_TOLERANCE})"
        )
