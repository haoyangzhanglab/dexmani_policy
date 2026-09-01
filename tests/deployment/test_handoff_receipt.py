from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from dexmani_policy.deployment.qualification_matrix import QUALIFICATION_MATRIX

try:
    from .real_restore_fixture import build_tiny_dp3_artifact
except ImportError:
    from real_restore_fixture import build_tiny_dp3_artifact


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RECEIPT_PATH = _REPOSITORY_ROOT / "docs" / "policy_to_real_handoff.json"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HandoffReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(_RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_frozen_repositories_and_artifact_identity(self) -> None:
        policy = self.receipt["policy_producer"]
        real = self.receipt["real_consumer"]
        artifact = self.receipt["artifact"]

        self.assertEqual(policy["repository"], "haoyangzhanglab/dexmani_policy")
        self.assertEqual(real["repository"], "haoyangzhanglab/dexmani_real")
        self.assertRegex(policy["commit"], _COMMIT_RE)
        self.assertRegex(real["commit"], _COMMIT_RE)
        self.assertIs(policy["clean"], True)
        self.assertIs(real["clean"], True)
        self.assertEqual(artifact["format"], "dexmani.deployment.v2")
        self.assertEqual(artifact["sidecar_schema_version"], 2)
        for name in (
            "checkpoint_sha256",
            "sidecar_sha256",
            "embedded_contract_sha256",
        ):
            self.assertRegex(artifact[name], _SHA256_RE)

    def test_prediction_future_is_not_executable_control(self) -> None:
        contract = self.receipt["policy_contract"]
        self.assertEqual(contract["horizon"], 16)
        self.assertEqual(contract["n_obs_steps"], 2)
        self.assertEqual(contract["n_action_steps"], 8)
        self.assertEqual(contract["required_action_steps"], 15)
        self.assertEqual(contract["prediction_future_steps"], 15)
        self.assertEqual(contract["executable_control_steps"], 8)
        self.assertNotEqual(
            contract["prediction_future_steps"],
            contract["executable_control_steps"],
        )
        self.assertEqual(contract["default_executable_output"], "control_action")

    def test_support_matrix_snapshot_matches_code_owned_source(self) -> None:
        snapshot = self.receipt["support_matrix"]
        source = _REPOSITORY_ROOT / snapshot["source"]
        self.assertTrue(source.is_file())
        for status in ("qualified", "conditional", "deferred", "rejected"):
            expected = sorted(
                policy
                for policy, entry in QUALIFICATION_MATRIX.items()
                if entry["status"] == status
            )
            self.assertEqual(snapshot[status], expected)

    def test_fixture_generator_is_byte_reproducible(self) -> None:
        producer_commit = self.receipt["policy_producer"]["commit"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_tiny_dp3_artifact(
                root / "first", producer_commit=producer_commit
            )
            second = build_tiny_dp3_artifact(
                root / "second", producer_commit=producer_commit
            )
            relative_paths = (
                Path("checkpoints/tiny-dp3-deployment-v2.pt"),
                Path("checkpoints/tiny-dp3-deployment-v2.pt.deployment.json"),
            )
            for relative_path in relative_paths:
                with self.subTest(path=relative_path):
                    self.assertEqual(
                        _sha256_file(first / relative_path),
                        _sha256_file(second / relative_path),
                    )


if __name__ == "__main__":
    unittest.main()
