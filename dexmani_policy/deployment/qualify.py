"""Direct/export inference parity qualification for deployment-v2 artifacts.

The direct branch deliberately constructs only the resolved experiment's
``agent``.  In particular, it never constructs the training dataset, an
environment runner, or anything from ``dexmani_sim``.  This makes the command a
useful deployment boundary check even on a machine with no simulator install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from dexmani_policy.deployment import export as exporter
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    DeploymentSpec,
    PredictionSnapshot,
    assert_prediction_parity,
    deterministic_observation,
    prediction_snapshot,
    reset_inference_seed,
    restore_deployment_agent,
    validate_deployment_normalizer,
    validate_prediction,
)

_MAX_PARITY_TOLERANCE = 1e-5


class PolicyParityError(RuntimeError):
    """Raised when an experiment cannot pass direct/export qualification."""


@dataclass(frozen=True)
class DirectRestoredPolicy:
    """A strict restore from a selected simple.v1 experiment checkpoint."""

    agent: Any
    spec: DeploymentSpec
    experiment_dir: Path
    checkpoint_path: Path
    checkpoint_selector: str
    checkpoint_sha256: str
    use_ema: bool
    selected_weights: str


@dataclass(frozen=True)
class ParityReport:
    """JSON-serializable receipt for a successful direct/export comparison."""

    experiment_dir: str
    selected_checkpoint: str
    checkpoint_selector: str
    source_checkpoint_sha256: str
    deployment_checkpoint: str
    deployment_checkpoint_sha256: str
    use_ema: bool
    selected_weights: str
    action_key: str
    action_dim: int
    control_action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    max_abs_diff: float
    pred_action_max_abs_diff: float
    control_action_max_abs_diff: float
    canonical_slice_max_abs_diff: float
    rtol: float
    atol: float

    def as_dict(self) -> dict[str, Any]:
        """Return only JSON-native values, including paths as strings."""
        return asdict(self)


def restore_direct_policy(
    experiment_dir: Path,
    *,
    checkpoint_selector: str = "best",
    device: torch.device | str = "cpu",
    point_cloud_num_points: int | None = None,
    point_cloud_feature_dim: int | None = None,
) -> DirectRestoredPolicy:
    """Strictly restore the selected simple.v1 model or EMA from an experiment.

    The resolved ``config.yaml`` is the constructor source of truth.  This
    intentionally does *not* use the dataset or environment portions of that
    config; only ``agent`` is handed to Hydra.
    """
    experiment = _resolve_experiment(experiment_dir)
    selected_path = exporter._resolve_checkpoint(experiment, checkpoint_selector)
    cfg_plain = exporter._load_config(experiment)
    checkpoint = exporter._load_training_checkpoint(selected_path)
    train, _, _ = exporter._reconcile_train_params(checkpoint, cfg_plain)
    exporter._validate_resolved_config_contract(cfg_plain, train)

    # Reuse the exporter's resolved-eval checks, but retain the original agent
    # mapping below: direct qualification must represent the experiment itself,
    # not the deployment constructor sanitization.
    inference = exporter._build_inference_config(cfg_plain, cfg_plain["agent"], train)
    agent_config = cfg_plain["agent"]
    exporter._validate_agent_targets(agent_config)
    use_ema = inference["eval"]["use_ema"]
    if type(use_ema) is not bool:
        raise PolicyParityError("resolved eval.use_ema must be bool")
    selected_weights = "ema_model" if use_ema else "model"
    selected_raw = checkpoint.ema_model_state if use_ema else checkpoint.model_state
    if selected_raw is None:
        raise PolicyParityError("eval.use_ema=true requires checkpoint EMA weights")

    # This is the same canonical key normalization the exporter applies before
    # strict loading.  It preserves the selected model/EMA tensors while making
    # ordinary compiled/DDP simple.v1 checkpoints directly restorable.
    selected_state = exporter._canonicalize_state_dict(
        selected_raw, f"weights.{selected_weights}"
    )
    spec = _direct_spec(
        inference,
        agent_config,
        point_cloud_num_points=point_cloud_num_points,
        point_cloud_feature_dim=point_cloud_feature_dim,
    )
    try:
        import hydra

        # Training commands chdir to the repository root before Hydra builds an
        # agent.  Preserve that behaviour for constructor-time relative assets
        # (for example a legacy codebook_path), regardless of the caller's cwd.
        with _repository_cwd():
            agent = hydra.utils.instantiate(OmegaConf.create(dict(agent_config)))
        agent.action_key = spec.action_key
        agent.load_state_dict(selected_state, strict=True)
        agent.to(device)
        agent.eval()
        _validate_direct_agent_dimensions(agent, spec)
        validate_deployment_normalizer(agent, spec)
    except DeploymentRestoreError:
        raise
    except Exception as exc:
        raise PolicyParityError(
            f"direct agent strict restore failed using simple.v1 {selected_weights}"
        ) from exc

    return DirectRestoredPolicy(
        agent=agent,
        spec=spec,
        experiment_dir=experiment,
        checkpoint_path=selected_path,
        checkpoint_selector=checkpoint_selector,
        checkpoint_sha256=_sha256_file(selected_path),
        use_ema=use_ema,
        selected_weights=selected_weights,
    )


def direct_prediction_snapshot(
    restored: DirectRestoredPolicy,
    *,
    seed: int = 0,
    observation: Mapping[str, torch.Tensor] | None = None,
) -> PredictionSnapshot:
    """Predict once from a direct restore after resetting every inference RNG."""
    obs = (
        deterministic_observation(restored.spec)
        if observation is None
        else dict(observation)
    )
    reset_inference_seed(seed)
    try:
        with torch.inference_mode():
            result = restored.agent.predict_action(
                obs, denoise_timesteps=restored.spec.denoise_steps
            )
    except Exception as exc:
        raise PolicyParityError("direct agent prediction failed") from exc
    try:
        return validate_prediction(
            result, restored.spec, batch_size=obs["joint_state"].shape[0]
        )
    except (KeyError, DeploymentRestoreError) as exc:
        raise PolicyParityError(
            "direct agent prediction violates deployment contract"
        ) from exc


def qualify_policy_parity(
    experiment_dir: Path,
    *,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
    zarr_path: Path | None = None,
    verify_export: bool = True,
    seed: int = 0,
    device: torch.device | str = "cpu",
    atol: float = 0.0,
    rtol: float = 0.0,
    tolerance_reason: str | None = None,
) -> ParityReport:
    """Export one checkpoint and prove direct/deployment-v2 prediction parity.

    The exact same synthetic observation is cloned for both predictions, and
    Python, NumPy, Torch, and CUDA RNG state are reset before each prediction.
    The exported payload is reloaded with the exporter's ``weights_only``
    deployment-v2 parser before strict deployment restore.
    """
    _validate_tolerance(atol, "atol")
    _validate_tolerance(rtol, "rtol")
    _require_tolerance_reason(atol, rtol, tolerance_reason)
    experiment = _resolve_experiment(experiment_dir)
    point_contract = _validated_point_cloud_contract(experiment, zarr_path)
    direct = restore_direct_policy(
        experiment,
        checkpoint_selector=checkpoint_selector,
        device=device,
        point_cloud_num_points=point_contract["point_cloud_num_points"],
        point_cloud_feature_dim=point_contract["point_cloud_feature_dim"],
    )

    receipt = exporter.export_deployment_artifact(
        direct.experiment_dir,
        checkpoint_selector=checkpoint_selector,
        output_path=output_path,
        verify=verify_export,
        zarr_path=zarr_path,
    )
    # This is intentionally the safe artifact reload owned by the exporter,
    # rather than retaining the in-memory publication payload.
    payload = exporter._load_deployment_payload(receipt.checkpoint_path)
    deployment = restore_deployment_agent(payload, device=device)
    _require_matching_specs(direct.spec, deployment.spec)

    observation = deterministic_observation(direct.spec, device=device)
    direct_snapshot = direct_prediction_snapshot(
        direct, seed=seed, observation=_clone_observation(observation)
    )
    deployment_snapshot = prediction_snapshot(
        deployment, seed=seed, observation=_clone_observation(observation)
    )
    try:
        assert_prediction_parity(
            direct_snapshot, deployment_snapshot, atol=atol, rtol=rtol
        )
    except DeploymentRestoreError as exc:
        raise PolicyParityError("direct/export prediction parity failed") from exc

    pred_max = _max_abs_diff(
        direct_snapshot.pred_action, deployment_snapshot.pred_action
    )
    control_max = _max_abs_diff(
        direct_snapshot.control_action, deployment_snapshot.control_action
    )
    start = direct.spec.n_obs_steps - 1
    control_dim = direct.spec.control_action_dim
    canonical_max = _max_abs_diff(
        direct_snapshot.pred_action[
            :, start : start + direct.spec.n_action_steps, :control_dim
        ],
        deployment_snapshot.pred_action[
            :, start : start + direct.spec.n_action_steps, :control_dim
        ],
    )
    return ParityReport(
        experiment_dir=str(direct.experiment_dir),
        selected_checkpoint=str(direct.checkpoint_path),
        checkpoint_selector=direct.checkpoint_selector,
        source_checkpoint_sha256=direct.checkpoint_sha256,
        deployment_checkpoint=str(receipt.checkpoint_path),
        deployment_checkpoint_sha256=receipt.checkpoint_sha256,
        use_ema=direct.use_ema,
        selected_weights=direct.selected_weights,
        action_key=direct.spec.action_key,
        action_dim=direct.spec.action_dim,
        control_action_dim=direct.spec.control_action_dim,
        horizon=direct.spec.horizon,
        n_obs_steps=direct.spec.n_obs_steps,
        n_action_steps=direct.spec.n_action_steps,
        max_abs_diff=max(pred_max, control_max, canonical_max),
        pred_action_max_abs_diff=pred_max,
        control_action_max_abs_diff=control_max,
        canonical_slice_max_abs_diff=canonical_max,
        rtol=float(rtol),
        atol=float(atol),
    )


def _resolve_experiment(experiment_dir: Path) -> Path:
    try:
        experiment = Path(experiment_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise PolicyParityError(
            f"experiment directory not found: {experiment_dir}"
        ) from exc
    if not experiment.is_dir():
        raise PolicyParityError(f"experiment path is not a directory: {experiment}")
    return experiment


def _direct_spec(
    inference: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    *,
    point_cloud_num_points: int | None,
    point_cloud_feature_dim: int | None,
) -> DeploymentSpec:
    """Build the shared inference shape contract without opening a Zarr store."""
    try:
        eval_config = inference["eval"]
        if (point_cloud_num_points is None) != (point_cloud_feature_dim is None):
            raise ValueError("point-cloud contract must provide both N and feature dim")
        spec = DeploymentSpec(
            action_key=_required_action_key(inference.get("action_key")),
            action_dim=_positive_int(inference.get("action_dim"), "action_dim"),
            horizon=_positive_int(inference.get("horizon"), "horizon"),
            n_obs_steps=_positive_int(inference.get("n_obs_steps"), "n_obs_steps"),
            n_action_steps=_positive_int(
                inference.get("n_action_steps"), "n_action_steps"
            ),
            denoise_steps=_positive_int(
                eval_config.get("denoise_steps"), "eval.denoise_steps"
            ),
            point_cloud_num_points=_positive_int(
                (
                    point_cloud_num_points
                    if point_cloud_num_points is not None
                    else agent_config.get("num_points")
                ),
                (
                    "point_cloud_num_points"
                    if point_cloud_num_points is not None
                    else "agent.num_points"
                ),
            ),
            point_cloud_feature_dim=_positive_int(
                (
                    point_cloud_feature_dim
                    if point_cloud_feature_dim is not None
                    else agent_config.get("pc_dim")
                ),
                (
                    "point_cloud_feature_dim"
                    if point_cloud_feature_dim is not None
                    else "agent.pc_dim"
                ),
            ),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PolicyParityError(
            "resolved direct agent is missing point-cloud inference dimensions"
        ) from exc
    if spec.n_obs_steps - 1 + spec.n_action_steps > spec.horizon:
        raise PolicyParityError("observation/action window exceeds horizon")
    if spec.action_dim < spec.control_action_dim:
        raise PolicyParityError("action_dim is smaller than control_action_dim")
    return spec


def _validated_point_cloud_contract(
    experiment: Path, zarr_override: Path | None
) -> dict[str, int]:
    """Read only the exporter's validated Zarr shape contract, never a dataset."""
    cfg_plain = exporter._load_config(experiment)
    modalities = exporter._dataset_modalities(cfg_plain)
    repo_root = Path(__file__).resolve().parents[2]
    zarr_path = exporter._resolve_zarr_path(cfg_plain, repo_root, zarr_override)
    contract = exporter._validate_zarr_contract(zarr_path, cfg_plain, modalities)
    return {
        "point_cloud_num_points": _positive_int(
            contract.get("point_cloud_num_points"), "point_cloud_num_points"
        ),
        "point_cloud_feature_dim": _positive_int(
            contract.get("point_cloud_feature_dim"), "point_cloud_feature_dim"
        ),
    }


def _required_action_key(value: Any) -> str:
    if value not in {"action", "action_ee"}:
        raise ValueError("action_key must be 'action' or 'action_ee'")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive int")
    return value


def _require_matching_specs(
    reference: DeploymentSpec, candidate: DeploymentSpec
) -> None:
    for name in (
        "action_key",
        "action_dim",
        "horizon",
        "n_obs_steps",
        "n_action_steps",
        "denoise_steps",
        "point_cloud_num_points",
        "point_cloud_feature_dim",
    ):
        if getattr(reference, name) != getattr(candidate, name):
            raise PolicyParityError(
                f"direct/export inference contract mismatch for {name}: "
                f"{getattr(reference, name)!r} != {getattr(candidate, name)!r}"
            )


def _validate_direct_agent_dimensions(agent: Any, spec: DeploymentSpec) -> None:
    """Keep direct restore strict without depending on restore.py private helpers."""
    expected = {
        "action_dim": spec.action_dim,
        "horizon": spec.horizon,
        "n_obs_steps": spec.n_obs_steps,
        "n_action_steps": spec.n_action_steps,
        "control_action_dim": spec.control_action_dim,
    }
    for name, wanted in expected.items():
        actual = getattr(agent, name, None)
        if actual is not None and actual != wanted:
            raise PolicyParityError(
                f"restored direct agent.{name}={actual!r} conflicts with "
                f"resolved config={wanted!r}"
            )


def _clone_observation(
    observation: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in observation.items()}


def _max_abs_diff(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise PolicyParityError("cannot measure max_abs_diff for different shapes")
    return float(torch.max(torch.abs(reference - candidate)).item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tolerance(value: float, label: str) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    if value > _MAX_PARITY_TOLERANCE:
        raise ValueError(
            f"{label}={value} exceeds the narrow deployment parity limit "
            f"({_MAX_PARITY_TOLERANCE})"
        )


def _require_tolerance_reason(
    atol: float, rtol: float, tolerance_reason: str | None
) -> None:
    if atol == 0.0 and rtol == 0.0:
        return
    if not isinstance(tolerance_reason, str) or not tolerance_reason.strip():
        raise ValueError(
            "non-exact parity tolerance requires a non-empty tolerance_reason"
        )


@contextmanager
def _repository_cwd():
    """Temporarily reproduce the repository-root cwd used by training commands."""
    old_cwd = Path.cwd()
    os.chdir(Path(__file__).resolve().parents[2])
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--zarr-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--tolerance-reason",
        default=None,
        help="required justification when --atol or --rtol is non-zero",
    )
    parser.add_argument(
        "--verify-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the exporter's strict deployment preflight too (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = qualify_policy_parity(
        args.experiment_dir,
        checkpoint_selector=args.checkpoint,
        output_path=args.output,
        zarr_path=args.zarr_path,
        verify_export=args.verify_export,
        seed=args.seed,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
        tolerance_reason=args.tolerance_reason,
    )
    print(json.dumps(report.as_dict(), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
