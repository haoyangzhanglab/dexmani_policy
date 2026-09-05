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
    from dexmani_policy.common.temporal_ensembler import ChunkOverlapBlender
    from dexmani_policy.deployment.contract import (
        DeploymentSpec,
        ObservationFieldSpec,
        RgbPreprocessingSpec,
    )
    from dexmani_policy.deployment.restore import (
        RestoredDeployment,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_ROOT = _REPOSITORY_ROOT / "experiments"
_DEPLOYMENT_SELECTOR = Path("checkpoints/deployment_latest.pt")


@dataclass(frozen=True)
class PolicySpec:
    """Policy-owned model and observation contract exposed to runtimes."""

    action_key: str
    action_dim: int
    control_action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    temporal_ensemble_coeff: float | None
    observation_fields: tuple[ObservationFieldSpec, ...]
    control_dt_s: float
    requires_hand: bool
    rgb_preprocessing: RgbPreprocessingSpec | None = None

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
        if self.temporal_ensemble_coeff is not None and (
            isinstance(self.temporal_ensemble_coeff, bool)
            or not isinstance(self.temporal_ensemble_coeff, (int, float))
            or not np.isfinite(float(self.temporal_ensemble_coeff))
            or self.temporal_ensemble_coeff < 0.0
        ):
            raise ValueError(
                "temporal_ensemble_coeff must be finite and non-negative or None"
            )
        names = tuple(field.name for field in self.observation_fields)
        if not names or len(set(names)) != len(names):
            raise ValueError("observation_fields must be non-empty and unique")
        if (
            isinstance(self.control_dt_s, bool)
            or not isinstance(self.control_dt_s, (int, float))
            or not np.isfinite(float(self.control_dt_s))
            or self.control_dt_s <= 0.0
        ):
            raise ValueError("control_dt_s must be a positive finite number")
        if type(self.requires_hand) is not bool:
            raise ValueError("requires_hand must be bool")
        if ("rgb" in names) != (self.rgb_preprocessing is not None):
            raise ValueError("RGB preprocessing must match the rgb observation field")


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
        self._blender = self._new_blender()

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
        original_blender = self._blender
        self._blender = self._new_blender()
        try:
            for _ in range(samples):
                started = time.perf_counter()
                self.predict(observation)
                durations.append(time.perf_counter() - started)
        finally:
            self._blender = original_blender
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
        if self._blender is None:
            control_tensor = snapshot.control_action
        else:
            full_control_prediction = snapshot.pred_action[
                ..., : self.spec.control_action_dim
            ]
            control_tensor = self._blender.update(
                full_control_prediction, n_action_steps=self.spec.n_action_steps
            )
        control_action = (
            control_tensor.squeeze(0).to(dtype=torch.float64).numpy().copy()
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
        if self._blender is not None:
            self._blender.reset()
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

    def _new_blender(self) -> ChunkOverlapBlender | None:
        coefficient = self.spec.temporal_ensemble_coeff
        if coefficient is None:
            return None
        from dexmani_policy.common.temporal_ensembler import ChunkOverlapBlender

        return ChunkOverlapBlender(
            temporal_ensemble_coeff=coefficient,
            n_obs_steps=self.spec.n_obs_steps,
        )

    def _require_open(self) -> RestoredDeployment:
        if self._restored is None:
            raise RuntimeError("Policy runtime is closed")
        return self._restored

    def _observation_tensors(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping of NumPy arrays")
        required = {field.name for field in self.spec.observation_fields}
        actual = set(observation)
        if actual != required:
            raise ValueError(
                f"observation modalities mismatch: got {sorted(actual)}, "
                f"expected {sorted(required)}"
            )

        import torch

        tensors: dict[str, Any] = {}
        for field in self.spec.observation_fields:
            value = observation[field.name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"observation[{field.name!r}] must be a NumPy array")
            expected_shape = (self.spec.n_obs_steps, *field.shape)
            if value.shape != expected_shape:
                raise ValueError(
                    f"observation[{field.name!r}] shape mismatch: got {value.shape}, "
                    f"expected {expected_shape}"
                )
            expected_dtype = _numpy_dtype(field.dtype)
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"observation[{field.name!r}] dtype mismatch: got {value.dtype}, "
                    f"expected {expected_dtype}"
                )
            if field.dtype == "float32" and not np.isfinite(value).all():
                raise ValueError(
                    f"observation[{field.name!r}] must contain finite real numbers"
                )
            if field.name == "rgb":
                _validate_rgb_value_range(value, field.semantics)
            contiguous_value = np.ascontiguousarray(value)
            if not contiguous_value.flags.writeable:
                contiguous_value = contiguous_value.copy()
            tensor = torch.from_numpy(contiguous_value).unsqueeze(0)
            tensors[field.name] = tensor
        from dexmani_policy.deployment.restore import prepare_deployment_observation

        prepared = prepare_deployment_observation(tensors, self._deployment_spec())
        return {name: value.to(self._device) for name, value in prepared.items()}


def _numpy_dtype(name: str) -> np.dtype[Any]:
    try:
        dtype = np.dtype(name)
    except TypeError as exc:
        raise RuntimeError(f"unsupported deployment dtype: {name!r}") from exc
    if dtype not in {np.dtype(np.float32), np.dtype(np.uint8)}:
        raise RuntimeError(f"unsupported deployment dtype: {name!r}")
    return dtype


def _validate_rgb_value_range(value: np.ndarray, metadata: Mapping[str, Any]) -> None:
    raw_range = metadata.get("value_range")
    if (
        type(raw_range) not in {list, tuple}
        or len(raw_range) != 2
        or tuple(raw_range) != (0, 255)
        or value.min(initial=0) < 0
        or value.max(initial=0) > 255
    ):
        raise ValueError("observation['rgb'] violates the RGB value range")


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
    from dexmani_policy.deployment.contract import deployment_contract
    from dexmani_policy.deployment.restore import deployment_spec

    deployment = deployment_spec(payload)
    contract = deployment_contract(payload)
    inference = contract["inference_config"]

    task_name = inference.get("task_name")
    if type(task_name) is not str or not task_name:
        raise RuntimeError("inference_config.task_name must be a non-empty string")
    return (
        PolicySpec(
            action_key=deployment.action_key,
            action_dim=deployment.action_dim,
            control_action_dim=deployment.control_action_dim,
            horizon=deployment.horizon,
            n_obs_steps=deployment.n_obs_steps,
            n_action_steps=deployment.n_action_steps,
            temporal_ensemble_coeff=deployment.temporal_ensemble_coeff,
            observation_fields=deployment.observation_fields,
            control_dt_s=deployment.control_dt_s,
            requires_hand=deployment.requires_hand,
            rgb_preprocessing=deployment.rgb_preprocessing,
        ),
        task_name,
    )


def _short_selector(experiment_dir: Path) -> str:
    try:
        relative = experiment_dir.relative_to(_EXPERIMENTS_ROOT)
    except ValueError:
        return str(experiment_dir)
    return relative.as_posix() if len(relative.parts) == 3 else str(experiment_dir)
