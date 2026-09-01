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

    def test_representative_temporal_values_follow_artifact_metadata(self) -> None:
        representative = self.receipt["representative_qualification"]
        contract = self.receipt["representative_policy_contract"]
        runtime = self.receipt["runtime_temporal_invariants"]

        for name in ("horizon", "n_obs_steps", "n_action_steps"):
            self.assertEqual(contract[name], representative[name])
        expected_required = contract["horizon"] - (contract["n_obs_steps"] - 1)
        self.assertEqual(contract["required_action_steps"], expected_required)
        self.assertEqual(
            contract["prediction_future_steps"], contract["required_action_steps"]
        )
        self.assertEqual(
            contract["executable_control_steps"], contract["n_action_steps"]
        )
        self.assertNotEqual(
            contract["prediction_future_steps"],
            contract["executable_control_steps"],
        )
        self.assertEqual(contract["default_executable_output"], "control_action")
        self.assertIs(runtime["artifact_driven"], True)
        self.assertEqual(runtime["control_start_expression"], "n_obs_steps - 1")
        self.assertEqual(
            runtime["required_action_steps_expression"],
            "horizon - (n_obs_steps - 1)",
        )
        self.assertEqual(
            runtime["prediction_future_steps_expression"], "required_action_steps"
        )
        self.assertEqual(
            runtime["executable_control_steps_expression"], "n_action_steps"
        )

        legacy = self.receipt["legacy_real_observation"]
        self.assertEqual(legacy["artifact_scope"], "representative_qualified_dp3")
        self.assertEqual(
            legacy["consumer_commit"], self.receipt["real_consumer"]["commit"]
        )
        self.assertEqual(
            legacy["preflight_action_steps"], contract["required_action_steps"]
        )

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

    def test_fixture_is_reproducible_and_matches_frozen_receipt(self) -> None:
        producer_commit = self.receipt["policy_producer"]["commit"]
        fixture_receipt = self.receipt["deterministic_fixture"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_tiny_dp3_artifact(
                root / "first", producer_commit=producer_commit
            )
            second = build_tiny_dp3_artifact(
                root / "second", producer_commit=producer_commit
            )
            checkpoint_relative = Path("checkpoints/tiny-dp3-deployment-v2.pt")
            sidecar_relative = Path(
                "checkpoints/tiny-dp3-deployment-v2.pt.deployment.json"
            )
            for relative_path in (checkpoint_relative, sidecar_relative):
                with self.subTest(path=relative_path):
                    self.assertEqual(
                        (first / relative_path).read_bytes(),
                        (second / relative_path).read_bytes(),
                    )

            checkpoint_sha256 = _sha256_file(first / checkpoint_relative)
            sidecar_sha256 = _sha256_file(first / sidecar_relative)
            self.assertEqual(
                checkpoint_sha256, fixture_receipt["checkpoint_sha256"]
            )
            self.assertEqual(sidecar_sha256, fixture_receipt["sidecar_sha256"])

            sidecar = json.loads((first / sidecar_relative).read_text(encoding="utf-8"))
            self.assertEqual(
                sidecar["schema_version"], fixture_receipt["sidecar_schema_version"]
            )
            self.assertEqual(
                sidecar["embedded_contract_sha256"],
                fixture_receipt["embedded_contract_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
