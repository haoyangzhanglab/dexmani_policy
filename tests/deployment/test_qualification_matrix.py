from __future__ import annotations

import unittest

from dexmani_policy.deployment.qualification_matrix import QUALIFICATION_MATRIX


class QualificationMatrixTest(unittest.TestCase):
    def test_matrix_is_explicit_and_evidence_backed(self) -> None:
        self.assertEqual(
            set(QUALIFICATION_MATRIX),
            {
                "dp3",
                "action_flow",
                "dqrise",
                "r3d",
                "maniflow",
                "sat",
                "dp_rgb",
                "moe_dp_rgb",
                "multitask_dit",
            },
        )
        valid_statuses = {"qualified", "conditional", "rejected", "deferred"}
        for policy, record in QUALIFICATION_MATRIX.items():
            with self.subTest(policy=policy):
                self.assertIn(record["status"], valid_statuses)
                self.assertIsInstance(record["evidence"], list)
                self.assertIsInstance(record["condition"], str)
                self.assertTrue(record["condition"])
                if record["status"] == "qualified":
                    self.assertTrue(record["evidence"])


if __name__ == "__main__":
    unittest.main()
