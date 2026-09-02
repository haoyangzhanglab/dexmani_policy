"""Stable, NumPy-only public runtime for exported Policy experiments.

The public objects in this module deliberately keep Torch and model details on
the Policy side of the deployment boundary.  Filesystem discovery is also kept
separate from artifact inspection so listing experiments never loads Torch, a
checkpoint, or a model.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import numpy as np

if TYPE_CHECKING:
    from dexmani_policy.deployment.restore import DeploymentSpec, RestoredDeployment


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_ROOT = _REPOSITORY_ROOT / "experiments"
_DEPLOYMENT_SELECTOR = Path("checkpoints/deployment_latest.pt")
_JOINT_STATE_DIM = 19


@dataclass(frozen=True)
class PolicySpec:
    """Policy-owned model and observation contract exposed to runtimes."""

    action_key: str
    action_dim: int
    control_action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    sensor_modalities: tuple[str, ...]
    point_cloud_num_points: int | None
    point_cloud_feature_dim: int | None
    rgb_shape: tuple[int, ...] | None
    rgb_color_order: str | None
    rgb_value_range: tuple[float, float] | None
    control_dt_s: float
    requires_hand: bool

    def __post_init__(self) -> None:
        if self.action_key not in {"action", "action_ee"}:
            raise ValueError("action_key must be 'action' or 'action_ee'")
        for name in (
            "action_dim",
            "control_action_dim",
            "horizon",
            "n_obs_steps",
            "n_action_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive int")
        expected_control_dim = 21 if self.action_key == "action_ee" else 19
        if self.control_action_dim != expected_control_dim:
            raise ValueError(
                "control_action_dim does not match the selected action space"
            )
        if self.action_dim < self.control_action_dim:
            raise ValueError("action_dim must not be smaller than control_action_dim")
        if self.n_obs_steps - 1 + self.n_action_steps > self.horizon:
            raise ValueError("observation/action window exceeds horizon")
        if (
            type(self.sensor_modalities) is not tuple
            or not self.sensor_modalities
            or any(
                type(modality) is not str or not modality
                for modality in self.sensor_modalities
            )
            or len(set(self.sensor_modalities)) != len(self.sensor_modalities)
        ):
            raise ValueError("sensor_modalities must be unique non-empty strings")
        if "joint_state" not in self.sensor_modalities:
            raise ValueError("sensor_modalities must include joint_state")

        point_dimensions = (
            self.point_cloud_num_points,
            self.point_cloud_feature_dim,
        )
        if "point_cloud" in self.sensor_modalities:
            if any(type(value) is not int or value <= 0 for value in point_dimensions):
                raise ValueError(
                    "point-cloud modalities require positive point dimensions"
                )
        elif any(value is not None for value in point_dimensions):
            raise ValueError("point-cloud dimensions require the point_cloud modality")

        rgb_contract = (self.rgb_shape, self.rgb_color_order, self.rgb_value_range)
        if "rgb" in self.sensor_modalities:
            if any(value is None for value in rgb_contract):
                raise ValueError("rgb modality requires the complete RGB contract")
        elif any(value is not None for value in rgb_contract):
            raise ValueError("RGB contract fields require the rgb modality")
        if self.rgb_shape is not None and (
            type(self.rgb_shape) is not tuple
            or not self.rgb_shape
            or any(type(value) is not int or value <= 0 for value in self.rgb_shape)
        ):
            raise ValueError("rgb_shape must contain positive integers")
        if self.rgb_color_order is not None and (
            type(self.rgb_color_order) is not str or not self.rgb_color_order
        ):
            raise ValueError("rgb_color_order must be a non-empty string")
        if self.rgb_value_range is not None:
            if (
                type(self.rgb_value_range) is not tuple
                or len(self.rgb_value_range) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(float(value))
                    for value in self.rgb_value_range
                )
                or self.rgb_value_range[0] >= self.rgb_value_range[1]
            ):
                raise ValueError("rgb_value_range must be an increasing finite pair")
        if (
            isinstance(self.control_dt_s, bool)
            or not isinstance(self.control_dt_s, (int, float))
            or not np.isfinite(float(self.control_dt_s))
            or self.control_dt_s <= 0.0
        ):
            raise ValueError("control_dt_s must be a positive finite number")
        if type(self.requires_hand) is not bool:
            raise ValueError("requires_hand must be bool")


@dataclass(frozen=True)
class ExperimentInfo:
    """Resolved experiment identity and immutable deployment contract."""

    selector: str
    experiment_dir: Path
    policy_name: str
    task_name: str
    checkpoint_path: Path
    checkpoint_name: str
    spec: PolicySpec


def resolve_experiment(selector: str | os.PathLike[str]) -> Path:
    """Resolve an existing directory or a ``policy/task/experiment`` selector.

    Short selectors are always rooted at this repository's ``experiments``
    directory.  There is intentionally no ``latest`` or timestamp selection.
    """
    raw_selector = os.fspath(selector)
    if not raw_selector:
        raise ValueError("experiment selector must not be empty")

    candidate = Path(raw_selector).expanduser()
    if candidate.is_dir():
        return candidate.resolve(strict=True)

    parts = raw_selector.split("/")
    if (
        candidate.is_absolute()
        or len(parts) != 3
        or any(not part or part in {".", ".."} for part in parts)
    ):
        raise ValueError(
            "experiment selector must be an existing directory or "
            "policy/task/experiment"
        )
    experiment_dir = _EXPERIMENTS_ROOT.joinpath(*parts)
    try:
        resolved = experiment_dir.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"experiment not found: {raw_selector}") from exc
    if not resolved.is_dir():
        raise FileNotFoundError(f"experiment is not a directory: {resolved}")
    return resolved


def list_experiments(filter: str | None = None) -> tuple[str, ...]:
    """List deployable short selectors without loading Torch or checkpoints.

    When provided, ``filter`` is a case-insensitive substring matched against
    the complete ``policy/task/experiment`` selector.
    """
    if filter is not None and (not isinstance(filter, str) or not filter):
        raise ValueError("filter must be a non-empty string or None")
    if not _EXPERIMENTS_ROOT.is_dir():
        return ()

    needle = filter.casefold() if filter is not None else None
    selectors: list[str] = []
    for policy_dir in _visible_directories(_EXPERIMENTS_ROOT):
        for task_dir in _visible_directories(policy_dir):
            for experiment_dir in _visible_directories(task_dir):
                selector = "/".join(
                    (policy_dir.name, task_dir.name, experiment_dir.name)
                )
                if needle is not None and needle not in selector.casefold():
                    continue
                if not (experiment_dir / "config.yaml").is_file():
                    continue
                if not (experiment_dir / _DEPLOYMENT_SELECTOR).is_file():
                    continue
                selectors.append(selector)
    return tuple(sorted(selectors))


def inspect_experiment(selector: str | os.PathLike[str]) -> ExperimentInfo:
    """Read one experiment's deployment metadata without constructing a model."""
    experiment_dir = resolve_experiment(selector)
    checkpoint_path = _resolve_deployment_checkpoint(experiment_dir)
    payload = _read_deployment_payload(checkpoint_path, map_location="meta")
    return _experiment_info(experiment_dir, checkpoint_path, payload)


def load_experiment(
    selector: str | os.PathLike[str],
    device: str = "cuda:0",
    seed: int = 0,
) -> LoadedPolicy:
    """Strictly restore the selected deployment artifact as a NumPy runtime."""
    if type(device) is not str or not device:
        raise ValueError("device must be a non-empty string")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative int")

    experiment_dir = resolve_experiment(selector)
    checkpoint_path = _resolve_deployment_checkpoint(experiment_dir)
    payload = _read_deployment_payload(checkpoint_path, map_location="cpu")
    info = _experiment_info(experiment_dir, checkpoint_path, payload)

    from dexmani_policy.deployment.restore import restore_deployment_agent

    restored = restore_deployment_agent(payload, device=device)
    runtime = LoadedPolicy(info, restored, device=device, seed=seed)
    runtime.reset_episode()
    return runtime


@final
class LoadedPolicy:
    """One restored Policy model with a NumPy-only inference surface."""

    def __init__(
        self,
        info: ExperimentInfo,
        restored: RestoredDeployment,
        *,
        device: str,
        seed: int,
    ) -> None:
        self.info = info
        self.spec = info.spec
        self._restored: RestoredDeployment | None = restored
        self._device = device
        self._seed = seed

    def warmup(self, *, samples: int) -> tuple[float, ...]:
        """Run deterministic synthetic samples and return durations in seconds.

        Global Python, NumPy, Torch, and already-initialized CUDA RNG states are
        restored so warmup does not consume the episode's inference stream.
        """
        if type(samples) is not int or samples < 1:
            raise ValueError("samples must be a positive int")
        self._require_open()

        import torch

        from dexmani_policy.deployment.restore import deterministic_observation

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        deployment_spec = self._deployment_spec()
        synthetic_tensors = deterministic_observation(deployment_spec)
        observation = {
            name: value.squeeze(0).numpy() for name, value in synthetic_tensors.items()
        }
        durations: list[float] = []
        try:
            for _ in range(samples):
                started = time.perf_counter()
                self.predict(observation)
                durations.append(time.perf_counter() - started)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return tuple(durations)

    def predict(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Predict one finite float64 ``[N, D]`` control-action chunk."""
        restored = self._require_open()
        tensors = self._observation_tensors(observation)

        import torch

        from dexmani_policy.deployment.restore import validate_prediction

        with torch.inference_mode():
            result = restored.agent.predict_action(
                tensors, denoise_timesteps=restored.spec.denoise_steps
            )
        snapshot = validate_prediction(result, restored.spec, batch_size=1)
        control_action = (
            snapshot.control_action.squeeze(0).to(dtype=torch.float64).numpy().copy()
        )
        expected_shape = (self.spec.n_action_steps, self.spec.control_action_dim)
        if (
            control_action.shape != expected_shape
            or not np.isfinite(control_action).all()
        ):
            raise RuntimeError(
                "Policy prediction is not a finite float64 control-action chunk"
            )
        return control_action

    def reset_episode(self) -> None:
        """Reset Policy-owned stochastic and optional episode-local model state."""
        restored = self._require_open()

        from dexmani_policy.deployment.restore import reset_inference_seed

        reset_inference_seed(self._seed)
        reset_method = getattr(restored.agent, "reset_episode", None)
        if reset_method is not None:
            if not callable(reset_method):
                raise RuntimeError("agent.reset_episode is not callable")
            reset_method()

    def close(self) -> None:
        """Release model resources.  Repeated calls are harmless."""
        restored = self._restored
        if restored is None:
            return
        self._restored = None
        close_method = getattr(restored.agent, "close", None)
        if close_method is not None:
            if not callable(close_method):
                raise RuntimeError("agent.close is not callable")
            close_method()
        restored.agent.to("cpu")
        if self._device.startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _deployment_spec(self) -> DeploymentSpec:
        return self._require_open().spec

    def _require_open(self) -> RestoredDeployment:
        if self._restored is None:
            raise RuntimeError("Policy runtime is closed")
        return self._restored

    def _observation_tensors(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping of NumPy arrays")
        required = set(self.spec.sensor_modalities)
        actual = set(observation)
        if actual != required:
            raise ValueError(
                f"observation modalities mismatch: got {sorted(actual)}, "
                f"expected {sorted(required)}"
            )

        expected_shapes: dict[str, tuple[int, ...]] = {
            "joint_state": (self.spec.n_obs_steps, _JOINT_STATE_DIM)
        }
        if "point_cloud" in required:
            if (
                self.spec.point_cloud_num_points is None
                or self.spec.point_cloud_feature_dim is None
            ):
                raise RuntimeError("point-cloud dimensions are missing from PolicySpec")
            expected_shapes["point_cloud"] = (
                self.spec.n_obs_steps,
                self.spec.point_cloud_num_points,
                self.spec.point_cloud_feature_dim,
            )
        if "rgb" in required:
            if self.spec.rgb_shape is None:
                raise RuntimeError("RGB shape is missing from PolicySpec")
            expected_shapes["rgb"] = (self.spec.n_obs_steps, *self.spec.rgb_shape)

        import torch

        tensors: dict[str, Any] = {}
        for name in self.spec.sensor_modalities:
            value = observation[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"observation[{name!r}] must be a NumPy array")
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"observation[{name!r}] shape mismatch: got {value.shape}, "
                    f"expected {expected_shapes[name]}"
                )
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(
                    f"observation[{name!r}] must contain finite real numbers"
                )
            contiguous = np.ascontiguousarray(value, dtype=np.float32)
            tensors[name] = torch.from_numpy(contiguous).unsqueeze(0).to(self._device)
        return tensors


def _visible_directories(directory: Path) -> tuple[Path, ...]:
    try:
        return tuple(
            child
            for child in directory.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
    except OSError:
        return ()


def _resolve_deployment_checkpoint(experiment_dir: Path) -> Path:
    checkpoint_dir = experiment_dir / "checkpoints"
    selector_path = experiment_dir / _DEPLOYMENT_SELECTOR
    try:
        resolved_dir = checkpoint_dir.resolve(strict=True)
        checkpoint_path = selector_path.resolve(strict=True)
        checkpoint_path.relative_to(resolved_dir)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"valid deployment checkpoint selector not found: {selector_path}"
        ) from exc
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"deployment checkpoint is not a file: {checkpoint_path}"
        )
    return checkpoint_path


def _read_deployment_payload(path: Path, *, map_location: str) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"cannot safely load deployment checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("deployment checkpoint payload must be a mapping")
    return payload


def _experiment_info(
    experiment_dir: Path,
    checkpoint_path: Path,
    payload: Mapping[str, Any],
) -> ExperimentInfo:
    policy_name, configured_task = _experiment_identity(experiment_dir)
    spec, artifact_task = _policy_spec(payload)
    if configured_task != artifact_task:
        raise RuntimeError(
            f"experiment task_name={configured_task!r} conflicts with "
            f"artifact task_name={artifact_task!r}"
        )
    return ExperimentInfo(
        selector=_short_selector(experiment_dir),
        experiment_dir=experiment_dir,
        policy_name=policy_name,
        task_name=artifact_task,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_path.name,
        spec=spec,
    )


def _experiment_identity(experiment_dir: Path) -> tuple[str, str]:
    from omegaconf import OmegaConf

    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config not found: {config_path}")
    try:
        config = OmegaConf.load(config_path)
        policy_name = config.get("policy_name")
        task_name = config.get("task_name")
    except Exception as exc:
        raise RuntimeError(f"cannot read experiment config: {config_path}") from exc
    for label, value in (("policy_name", policy_name), ("task_name", task_name)):
        if type(value) is not str or not value:
            raise RuntimeError(f"experiment config {label} must be a non-empty string")
    return policy_name, task_name


def _policy_spec(payload: Mapping[str, Any]) -> tuple[PolicySpec, str]:
    from dexmani_policy.deployment.restore import deployment_spec

    deployment = deployment_spec(payload)
    try:
        state = payload["state"]
        inference = state["inference_config"]
        data = state["data_contract"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("deployment checkpoint metadata is incomplete") from exc
    if not isinstance(inference, Mapping) or not isinstance(data, Mapping):
        raise RuntimeError("deployment checkpoint metadata must contain mappings")

    task_name = inference.get("task_name")
    if type(task_name) is not str or not task_name:
        raise RuntimeError("inference_config.task_name must be a non-empty string")
    raw_modalities = data.get("sensor_modalities")
    if (
        type(raw_modalities) is not list
        or any(type(item) is not str or not item for item in raw_modalities)
        or len(set(raw_modalities)) != len(raw_modalities)
    ):
        raise RuntimeError("data_contract.sensor_modalities must be unique strings")
    sensor_modalities = tuple(raw_modalities)
    if set(sensor_modalities) != {"joint_state", "point_cloud"}:
        raise RuntimeError(
            "current deployment runtime requires joint_state + point_cloud"
        )

    control_dt_s = _positive_float(data.get("dt"), "data_contract.dt")
    requires_hand = data.get("requires_hand", True)
    if type(requires_hand) is not bool:
        raise RuntimeError("data_contract.requires_hand must be bool when present")
    rgb_shape = _optional_shape(data.get("rgb_shape"), "data_contract.rgb_shape")
    rgb_color_order = _optional_string(
        data.get("rgb_color_order"), "data_contract.rgb_color_order"
    )
    rgb_value_range = _optional_range(
        data.get("rgb_value_range"), "data_contract.rgb_value_range"
    )

    return (
        PolicySpec(
            action_key=deployment.action_key,
            action_dim=deployment.action_dim,
            control_action_dim=deployment.control_action_dim,
            horizon=deployment.horizon,
            n_obs_steps=deployment.n_obs_steps,
            n_action_steps=deployment.n_action_steps,
            sensor_modalities=sensor_modalities,
            point_cloud_num_points=deployment.point_cloud_num_points,
            point_cloud_feature_dim=deployment.point_cloud_feature_dim,
            rgb_shape=rgb_shape,
            rgb_color_order=rgb_color_order,
            rgb_value_range=rgb_value_range,
            control_dt_s=control_dt_s,
            requires_hand=requires_hand,
        ),
        task_name,
    )


def _short_selector(experiment_dir: Path) -> str:
    try:
        relative = experiment_dir.relative_to(_EXPERIMENTS_ROOT)
    except ValueError:
        return str(experiment_dir)
    return relative.as_posix() if len(relative.parts) == 3 else str(experiment_dir)


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{label} must be a positive finite number")
    return result


def _optional_shape(value: Any, label: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if (
        type(value) is not list
        or not value
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise RuntimeError(f"{label} must be a positive integer list or None")
    return tuple(value)


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise RuntimeError(f"{label} must be a non-empty string or None")
    return value


def _optional_range(value: Any, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if type(value) is not list or len(value) != 2:
        raise RuntimeError(f"{label} must be a two-number list or None")
    low = _finite_float(value[0], label)
    high = _finite_float(value[1], label)
    if low >= high:
        raise RuntimeError(f"{label} lower bound must be less than upper bound")
    return low, high


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must contain finite numbers")
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"{label} must contain finite numbers")
    return result
