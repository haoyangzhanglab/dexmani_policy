"""Fresh-process probe for the Real deployment-v2 Policy restore boundary.

It is intentionally a script, not a pytest fixture: the production preflight
uses ``spawn``, and the test needs an import-clean Python interpreter before
the Real package is imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_FORBIDDEN_MODULE_PREFIXES = (
    "dexmani_real.camera",
    "dexmani_real.deployment.coordinator",
    "dexmani_real.deployment.lifecycle",
    "dexmani_real.deployment.worker",
    "dexmani_real.recording",
    "dexmani_real.robot.",
    "dexmani_real.sensor.",
    "dexmani_real.shm.",
    "dexmani_sim",
    "pyrealsense2",
    "xarm",
)


def _assert_policy_not_imported() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name == "dexmani_policy" or name.startswith("dexmani_policy.")
    )
    if loaded:
        raise AssertionError(f"dexmani_policy was imported before Real: {loaded}")


def _assert_no_hardware_modules() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name.startswith(_FORBIDDEN_MODULE_PREFIXES)
    )
    if loaded:
        raise AssertionError(f"restore imported a hardware/simulator module: {loaded}")


def _runtime_source_config() -> SimpleNamespace:
    """The small pure-data subset consumed by ``resolve_policy_runtime_config``."""
    return SimpleNamespace(
        policy=SimpleNamespace(
            control_hz=16.0,
            arm_max_delta_rad_per_tick=0.1,
            endpoint_delta_tolerance_rad=1e-12,
        ),
        hand=SimpleNamespace(hand_max_delta_rad_per_tick=0.3),
        pointcloud=SimpleNamespace(num_points=1024, sha256="a" * 64),
        environment=SimpleNamespace(table=SimpleNamespace(enabled=False)),
    )


def _direct_restore(experiment: Path) -> dict[str, Any]:
    _assert_policy_not_imported()
    from dexmani_real.deployment.artifact import resolve_policy_artifact
    from dexmani_real.deployment.config import resolve_policy_runtime_config
    from dexmani_real.deployment.policy_checkpoint import (
        load_deployment_checkpoint_stream,
    )
    from dexmani_real.integrations.dexmani_policy import (
        DexManiPolicyRuntime,
        precheck_policy_package_provenance,
    )

    artifact = resolve_policy_artifact(experiment)
    projection = resolve_policy_runtime_config(
        artifact=artifact,
        runtime_config=_runtime_source_config(),
        device="cpu",
        inference_seed=0,
    )
    # This gate proves Policy remains absent until Real has resolved the exact
    # artifact and bound its package origin/commit.
    provenance = precheck_policy_package_provenance(projection.runtime)
    with artifact.checkpoint_path.open("rb") as stream:
        checkpoint = load_deployment_checkpoint_stream(stream)
    runtime = DexManiPolicyRuntime(projection.runtime)
    try:
        runtime.load_loaded_checkpoint(checkpoint, package_provenance=provenance)
        manifest = runtime._manifest
        agent = runtime._agent
        if manifest is None or agent is None:
            raise AssertionError("Real runtime did not retain a restored manifest/agent")
        if (
            manifest.action_key != "action"
            or manifest.action_dim != 19
            or manifest.control_action_dim != 19
            or manifest.n_obs_steps != 2
            or manifest.n_action_steps != 8
            or manifest.horizon != 16
            or not manifest.uses_point_cloud
        ):
            raise AssertionError(f"unexpected restored deployment manifest: {manifest}")
        expected_dims = {"action": 19, "joint_state": 19, "point_cloud": 6}
        for key, expected_dim in expected_dims.items():
            params = agent.normalizer.params_dict[key]
            scale, offset = params["scale"], params["offset"]
            if (
                scale.numel() != expected_dim
                or offset.numel() != expected_dim
                or not bool(scale.isfinite().all())
                or not bool(offset.isfinite().all())
                or bool((scale == 0).any())
            ):
                raise AssertionError(f"invalid restored normalizer field {key!r}")
        return {
            "checkpoint_sha256": artifact.checkpoint_sha256_from_index,
            "manifest": {
                "action_dim": manifest.action_dim,
                "control_action_dim": manifest.control_action_dim,
                "n_action_steps": manifest.n_action_steps,
                "n_obs_steps": manifest.n_obs_steps,
                "uses_point_cloud": manifest.uses_point_cloud,
            },
            "normalizer_dims": expected_dims,
            "package_commit": provenance["commit"],
        }
    finally:
        runtime.close()


def _production_preflight(experiment: Path) -> dict[str, Any]:
    _assert_policy_not_imported()
    from dexmani_real.deployment.artifact import resolve_policy_artifact
    from dexmani_real.deployment.config import resolve_policy_runtime_config
    from dexmani_real.deployment.preflight import run_isolated_preflight

    artifact = resolve_policy_artifact(experiment)
    projection = resolve_policy_runtime_config(
        artifact=artifact,
        runtime_config=_runtime_source_config(),
        device="cpu",
        inference_seed=0,
    )
    receipt = run_isolated_preflight(projection.runtime, timeout_s=120.0)
    return {
        "action_dim": receipt.action_dim,
        "action_steps": receipt.action_steps,
        "checkpoint_sha256": receipt.checkpoint_sha256,
        "checkpoint_sha256_verified": receipt.checkpoint_sha256_verified,
        "package_commit": receipt.package_commit,
        "package_dirty": receipt.package_dirty,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--mode", choices=("direct", "preflight"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = (
            _direct_restore(args.experiment)
            if args.mode == "direct"
            else _production_preflight(args.experiment)
        )
        _assert_no_hardware_modules()
    except BaseException as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(json.dumps({"ok": True, "result": result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
