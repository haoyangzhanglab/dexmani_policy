"""Hardware-free qualification of the current Real deployment restore path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

try:
    from .real_restore_fixture import (
        build_tiny_dp3_artifact,
        clone_clean_policy_repository,
        make_policy_source_dirty,
    )
except ImportError:  # unittest discovery may import this module without a package.
    from real_restore_fixture import (
        build_tiny_dp3_artifact,
        clone_clean_policy_repository,
        make_policy_source_dirty,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REAL_ROOT = _REPOSITORY_ROOT.parent / "dexmani_real"
_PROBE = Path(__file__).with_name("real_restore_probe.py")

_FULL_ENABLE_ENV = "DEXMANI_REAL_FULL_RESTORE"
_FULL_EXPERIMENT_ENV = "DEXMANI_REAL_FULL_RESTORE_EXPERIMENT"
_FULL_REAL_ROOT_ENV = "DEXMANI_REAL_FULL_RESTORE_REAL_ROOT"


def _probe(
    *,
    experiment: Path,
    policy_root: Path,
    real_root: Path,
    mode: str,
    synthetic_runtime: bool,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    environment = os.environ.copy()
    import_paths = [str(policy_root), str(real_root)]
    prior = environment.get("PYTHONPATH")
    if prior:
        import_paths.append(prior)
    environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    command = [
        sys.executable,
        str(_PROBE),
        "--experiment",
        str(experiment),
        "--mode",
        mode,
    ]
    if synthetic_runtime:
        command.append("--synthetic-runtime")
    completed = subprocess.run(
        command,
        cwd=str(_REPOSITORY_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(
            "restore probe produced no JSON receipt\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
    try:
        receipt = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "restore probe final stdout line was not JSON\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        ) from exc
    if not isinstance(receipt, dict):
        raise AssertionError(f"restore probe receipt is not an object: {receipt!r}")
    return completed, receipt


@unittest.skipUnless(_DEFAULT_REAL_ROOT.is_dir(), "sibling dexmani_real repository is unavailable")
class TinyDp3RealRestoreTest(unittest.TestCase):
    """Exercise a tiny genuine DP3 checkpoint across the Real boundary."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls._root = Path(cls._temporary_directory.name)
        cls._policy_root = cls._root / "clean-policy"
        cls._commit = clone_clean_policy_repository(_REPOSITORY_ROOT, cls._policy_root)
        cls._experiment = build_tiny_dp3_artifact(
            cls._root, producer_commit=cls._commit
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _run(self, *, experiment: Path | None = None, mode: str) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
        return _probe(
            experiment=experiment or self._experiment,
            policy_root=self._policy_root,
            real_root=_DEFAULT_REAL_ROOT,
            mode=mode,
            synthetic_runtime=True,
        )

    def test_direct_restore_validates_manifest_and_normalizer(self) -> None:
        completed, receipt = self._run(mode="direct")
        self.assertEqual(
            completed.returncode,
            0,
            f"receipt={receipt!r}\nstderr={completed.stderr}",
        )
        self.assertTrue(receipt["ok"], receipt)
        result = receipt["result"]
        self.assertEqual(result["package_commit"], self._commit)
        self.assertEqual(
            result["manifest"],
            {
                "action_dim": 19,
                "control_action_dim": 19,
                "n_action_steps": 8,
                "n_obs_steps": 2,
                "uses_point_cloud": True,
            },
        )
        self.assertEqual(
            result["normalizer_dims"],
            {"action": 19, "joint_state": 19, "point_cloud": 6},
        )

    def test_production_spawn_preflight_restores_and_predicts(self) -> None:
        completed, receipt = self._run(mode="preflight")
        self.assertEqual(
            completed.returncode,
            0,
            f"receipt={receipt!r}\nstderr={completed.stderr}",
        )
        self.assertTrue(receipt["ok"], receipt)
        result = receipt["result"]
        self.assertEqual(result["package_commit"], self._commit)
        self.assertEqual(result["package_dirty"], "false")
        self.assertEqual(result["action_dim"], 19)
        self.assertEqual(result["action_steps"], 15)
        self.assertTrue(result["checkpoint_sha256_verified"])

    def test_producer_commit_mismatch_is_rejected_in_a_temp_repository(self) -> None:
        mismatched = build_tiny_dp3_artifact(
            self._root / "producer-mismatch",
            producer_commit="f" * 40 if self._commit != "f" * 40 else "e" * 40,
        )
        completed, receipt = self._run(experiment=mismatched, mode="direct")
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(receipt["ok"])
        self.assertIn("commit does not match artifact producer", receipt["error"])

    def test_dirty_policy_source_is_rejected_in_a_temp_repository(self) -> None:
        dirty_root = self._root / "dirty-policy"
        dirty_commit = clone_clean_policy_repository(_REPOSITORY_ROOT, dirty_root)
        dirty_artifact = build_tiny_dp3_artifact(
            self._root / "dirty-artifact", producer_commit=dirty_commit
        )
        make_policy_source_dirty(dirty_root)
        completed, receipt = _probe(
            experiment=dirty_artifact,
            policy_root=dirty_root,
            real_root=_DEFAULT_REAL_ROOT,
            mode="direct",
            synthetic_runtime=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(receipt["ok"])
        self.assertIn("source tree must be clean", receipt["error"])

    def test_wrong_normalizer_width_is_rejected_before_prediction(self) -> None:
        malformed = build_tiny_dp3_artifact(
            self._root / "wrong-normalizer",
            producer_commit=self._commit,
            normalizer_point_cloud_dim=5,
        )
        completed, receipt = self._run(experiment=malformed, mode="direct")
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(receipt["ok"])
        self.assertIn("normalizer 'point_cloud'", receipt["error"])


class FullCleanTreeDp3RestoreTest(unittest.TestCase):
    """Opt-in, unmocked qualification of an operator-supplied exported DP3 run."""

    def test_full_clean_tree_dp3_preflight(self) -> None:
        missing = [
            name
            for name in (
                _FULL_ENABLE_ENV,
                _FULL_EXPERIMENT_ENV,
                _FULL_REAL_ROOT_ENV,
            )
            if not os.environ.get(name)
        ]
        if missing:
            self.skipTest(
                "opt-in full Real qualification requires " + ", ".join(missing)
            )
        if os.environ[_FULL_ENABLE_ENV] != "1":
            self.skipTest(f"{_FULL_ENABLE_ENV}=1 is required")
        experiment = Path(os.environ[_FULL_EXPERIMENT_ENV]).expanduser().resolve()
        real_root = Path(os.environ[_FULL_REAL_ROOT_ENV]).expanduser().resolve()
        if not experiment.is_dir():
            self.fail(f"{_FULL_EXPERIMENT_ENV} is not an experiment directory: {experiment}")
        if not (real_root / "dexmani_real").is_dir():
            self.fail(f"{_FULL_REAL_ROOT_ENV} is not a dexmani_real checkout: {real_root}")
        completed, receipt = _probe(
            experiment=experiment,
            policy_root=_REPOSITORY_ROOT,
            real_root=real_root,
            mode="preflight",
            synthetic_runtime=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"receipt={receipt!r}\nstderr={completed.stderr}",
        )
        self.assertTrue(receipt["ok"], receipt)
        result = receipt["result"]
        self.assertEqual(result["action_dim"], 19)
        self.assertEqual(result["action_steps"], 15)
        self.assertTrue(result["checkpoint_sha256_verified"])
        self.assertEqual(result["package_dirty"], "false")


if __name__ == "__main__":
    unittest.main()
