"""融合位置的控制变量与缺失传感器语义；采样使用明确的测试替身。"""

import unittest
from unittest.mock import patch

import torch

from dexmani_policy.agents.obs_encoder.interaction.encoder import InteractionObsEncoder
from dexmani_policy.agents.obs_encoder.interaction.finger_evidence import FingerEvidenceEncoder
from test_interaction_flow import ReferencePointOps


class InteractionEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = InteractionObsEncoder(
            n_obs_steps=2, token_dim=24, num_patches=4, group_size=4,
            patch_depth=1, num_scene_tokens=2, use_tactile_valid=True,
            fps_random_config={"use_random": False},
        ).eval()
        self.obs = {
            "joint_state": torch.randn(4, 19),
            "point_cloud": torch.rand(4, 16, 6) * 0.06,
            "eef_pose": torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.]).repeat(4, 1),
            "fingertip_points": torch.rand(4, 5, 3) * 0.04,
            "contact_force": torch.randn(4, 5, 3),
            "tactile_valid": torch.ones(4, 5, dtype=torch.bool),
        }
        self.ops = patch("dexmani_policy.agents.obs_encoder.pointcloud.ops._pointcloud_ops", return_value=ReferencePointOps)
        self.ops.start()
        self.addCleanup(self.ops.stop)

    def test_query_late_share_direct_residual_and_only_change_read_query(self):
        read_queries, norm_inputs = [], []
        q_hook = self.encoder.relation.register_forward_pre_hook(
            lambda module, args, kwargs: read_queries.append(kwargs["query_context"].detach().clone()),
            with_kwargs=True,
        )
        n_hook = self.encoder.hand_input_norm.register_forward_pre_hook(
            lambda module, args: norm_inputs.append(args[0].detach().clone())
        )
        self.addCleanup(q_hook.remove)
        self.addCleanup(n_hook.remove)
        params_before = sum(p.numel() for p in self.encoder.parameters())
        query_cond, query_aux = self.encoder(self.obs, return_intermediate=True)
        self.encoder.tactile_fusion = "late"
        late_cond, late_aux = self.encoder(self.obs, return_intermediate=True)
        self.assertEqual(params_before, sum(p.numel() for p in self.encoder.parameters()))
        torch.testing.assert_close(read_queries[0] - read_queries[1], query_aux["finger_evidence"])
        torch.testing.assert_close(
            norm_inputs[0] - query_aux["relation_update"],
            norm_inputs[1] - late_aux["relation_update"],
        )
        self.assertGreater((query_aux["relation_update"] - late_aux["relation_update"]).abs().max().item(), 1e-6)
        self.assertEqual(query_cond.shape, (2, 18, 24))
        torch.testing.assert_close(query_cond, query_aux["frame_tokens"].reshape(2, 18, 24))
        self.assertTrue(torch.isfinite(late_cond).all())

    def test_missing_touch_masks_nan_and_query_equals_late(self):
        self.obs["tactile_valid"].fill_(False)
        self.obs["contact_force"].fill_(float("nan"))
        query_cond, aux = self.encoder(self.obs, return_intermediate=True)
        self.encoder.tactile_fusion = "late"
        late_cond, _ = self.encoder(self.obs)
        torch.testing.assert_close(query_cond, late_cond)
        self.assertEqual(aux["finger_evidence"].count_nonzero().item(), 0)

    def test_zero_valid_signal_differs_from_missing_and_wrist_has_no_touch(self):
        evidence = FingerEvidenceEncoder(frame_dim=64, token_dim=24)
        frame = torch.zeros(2, 5, 64)
        force = torch.zeros(2, 5, 3)
        valid = torch.ones(2, 5, dtype=torch.bool)
        present = evidence(frame, force, valid)
        missing = evidence(frame, force, ~valid)
        self.assertEqual(present[:, 0].count_nonzero().item(), 0)
        self.assertGreater(present[:, 1:].abs().sum().item(), 0)
        self.assertEqual(missing.count_nonzero().item(), 0)

    def test_dropout_masks_sensor_evidence_without_changing_geometry(self):
        self.encoder.train()
        self.encoder.tactile_dropout_prob = 1.0
        pointcloud = self.obs["point_cloud"].clone()
        _, aux = self.encoder(self.obs, return_intermediate=True)
        self.assertFalse(aux["tactile_valid"].any())
        self.assertEqual(aux["finger_evidence"].count_nonzero().item(), 0)
        torch.testing.assert_close(self.obs["point_cloud"], pointcloud)

    def test_nonfinite_visible_points_are_rejected_before_sampling(self):
        self.obs["point_cloud"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite observed points"):
            self.encoder(self.obs)


if __name__ == "__main__":
    unittest.main()
