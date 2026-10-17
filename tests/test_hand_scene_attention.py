"""CPU-only contract tests; no PyTorch3D, data, weights or simulator required."""

import unittest

import torch

from dexmani_policy.agents.obs_encoder.interaction.hand_scene_attention import (
    HandSceneAttention,
    SurfaceSetEncoder,
    gather_surface_support,
)


class HandSceneAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = HandSceneAttention(token_channels=24, surface_k=8)
        self.anchors = torch.randn(2, 6, 3) * 0.025
        self.points = torch.randn(2, 40, 3) * 0.035
        self.tokens = torch.randn(2, 6, 24)
        self.patches = {
            "tokens": torch.randn(2, 9, 24),
            "xyz": torch.randn(2, 9, 3) * 0.08,
            "mask": torch.ones(2, 9, dtype=torch.bool),
        }

    def run_model(self, **kwargs):
        return self.model(self.points, self.patches, self.anchors, self.tokens, **kwargs)

    def test_contract_and_backward(self):
        self.patches["tokens"].requires_grad_()
        self.tokens.requires_grad_()
        result = self.run_model(return_intermediate=True)
        self.assertEqual(result["relation_update"].shape, (2, 6, 24))
        self.assertEqual(result["geometry_descriptor"].shape, (2, 6, 7))
        self.assertEqual(self.model.out_shape, (6, 24))
        self.assertIs(result["hand_query"], self.tokens)
        torch.testing.assert_close(result["wrist_relation"], result["relation_update"][:, :1])
        torch.testing.assert_close(result["finger_relation"], result["relation_update"][:, 1:])
        torch.testing.assert_close(result["context_attention"].sum(-1), torch.ones(2, 6, 4))
        result["relation_update"].square().mean().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(self.patches["tokens"].grad.abs().sum() > 0)
        self.assertTrue(self.tokens.grad.abs().sum() > 0)

    def test_masked_nonfinite_equals_removed(self):
        self.patches["tokens"][:, 2] = torch.nan
        self.patches["xyz"][:, 2] = torch.inf
        self.patches["mask"][:, 2] = False
        self.points[:, 3] = torch.nan
        mask = torch.ones(2, 40, dtype=torch.bool)
        mask[:, 3] = False
        masked = self.run_model(point_mask=mask)["relation_update"]
        keep_patch = torch.arange(9) != 2
        keep_point = torch.arange(40) != 3
        clean_patches = {key: value[:, keep_patch] for key, value in self.patches.items()}
        clean = self.model(self.points[:, keep_point], clean_patches, self.anchors, self.tokens)
        torch.testing.assert_close(masked, clean["relation_update"])
        self.assertTrue(torch.isfinite(masked).all())

    def test_nonfinite_surface_points_are_missing(self):
        self.points[:, :2] = torch.nan
        automatic = self.run_model()
        clean = self.model(self.points[:, 2:], self.patches, self.anchors, self.tokens)
        torch.testing.assert_close(automatic["relation_update"], clean["relation_update"])

    def test_all_missing_forward_and_backward(self):
        self.points.fill_(torch.nan)
        self.patches["tokens"].fill_(torch.nan)
        self.patches["xyz"].fill_(torch.nan)
        self.patches["mask"].fill_(False)
        result = self.run_model(return_intermediate=True)
        self.assertTrue((result["relation_update"] == 0).all())
        self.assertTrue((result["geometry_descriptor"] == 0).all())
        self.assertTrue(torch.isinf(result["min_observed_distance"]).all())
        self.assertFalse(result["has_reference"].any())
        self.assertTrue((result["context_null_mass"] == 1).all())
        result["relation_update"].sum().backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue((parameter.grad == 0).all())

    def test_empty_tensor_dimensions(self):
        patches = {key: value[:, :0] for key, value in self.patches.items()}
        result = self.model(self.points[:, :0], patches, self.anchors, self.tokens, return_intermediate=True)
        self.assertTrue((result["relation_update"] == 0).all())
        self.assertEqual(result["context_attention"].shape, (2, 6, 4, 1))

    def test_empty_raw_surface_keeps_scene_context(self):
        result = self.model(self.points[:, :0], self.patches, self.anchors, self.tokens, return_intermediate=True)
        self.assertTrue(torch.isfinite(result["relation_update"]).all())
        self.assertGreater(result["relation_update"].abs().sum().item(), 0)
        self.assertTrue((result["finger_local"] == 0).all())
        self.assertFalse(result["has_reference"].any())

    def test_point_mask_does_not_mask_context(self):
        result = self.run_model(point_mask=torch.zeros(2, 40, dtype=torch.bool), return_intermediate=True)
        self.assertFalse(result["has_reference"].any())
        self.assertGreater(result["context_update"].abs().sum().item(), 0)

    def test_raw_surface_independent_of_patch_members(self):
        # The exact reference point need not be any patch center/member.
        self.points[:, 0] = self.anchors[:, 1]
        self.patches["neighbor_indices"] = torch.full((2, 9, 2), 10, dtype=torch.long)
        result = self.run_model()
        torch.testing.assert_close(result["min_observed_distance"][:, 1], torch.zeros(2))

    def test_near_far_missing_are_distinct(self):
        anchor = torch.zeros(1, 6, 3)
        near = gather_surface_support(torch.tensor([[[0.01, 0., 0.]]]), anchor, 32, .04, .1)
        far = gather_surface_support(torch.tensor([[[0.4, 0., 0.]]]), anchor, 32, .04, .1)
        missing = gather_surface_support(torch.empty(1, 0, 3), anchor, 32, .04, .1)
        self.assertTrue(near["near_supported"].all())
        self.assertTrue(far["has_reference"].all())
        self.assertFalse(far["near_supported"].any())
        self.assertFalse(missing["has_reference"].any())
        self.assertTrue(torch.isfinite(missing["descriptor"]).all())
        self.assertTrue((missing["descriptor"] == 0).all())
        self.assertGreater(near["descriptor"][0, 0, 4], far["descriptor"][0, 0, 4])
        torch.testing.assert_close(near["nearest_distance"], torch.full((1, 6), .01))

    def test_far_points_do_not_create_local_features(self):
        self.points += 5.
        result = self.run_model(return_intermediate=True)
        self.assertTrue(result["has_reference"].all())
        self.assertFalse(result["near_has_support"].any())
        self.assertTrue((result["finger_local"] == 0).all())

    def test_wrist_depends_on_hand_shape(self):
        baseline = self.run_model()["wrist_relation"]
        anchors = self.anchors.clone()
        anchors[:, 2, 1] += .03
        changed = self.model(self.points, self.patches, anchors, self.tokens)["wrist_relation"]
        self.assertGreater((changed - baseline).abs().max().item(), 1e-7)

    def test_local_surface_shape_not_only_nearest_point(self):
        encoder = SurfaceSetEncoder(.04)
        a = torch.tensor([[[[.005, 0., 0.], [.012, 0., 0.], [.025, 0., 0.]]]])
        b = a.clone()
        b[..., 1, :] = torch.tensor([0., .012, 0.])
        b[..., 2, :] = torch.tensor([0., 0., .025])
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        self.assertGreater((encoder(a, mask) - encoder(b, mask)).abs().max().item(), 1e-6)

    def test_point_and_patch_permutation(self):
        baseline = self.run_model()["relation_update"]
        point_order = torch.randperm(40)
        patch_order = torch.randperm(9)
        patches = {key: value[:, patch_order] for key, value in self.patches.items()}
        result = self.model(self.points[:, point_order], patches, self.anchors, self.tokens)
        torch.testing.assert_close(baseline, result["relation_update"], atol=1e-6, rtol=1e-5)

    def test_fingers_share_local_reader(self):
        order = torch.tensor([0, 5, 3, 1, 4, 2])
        baseline = self.run_model()["finger_relation"]
        result = self.model(self.points, self.patches, self.anchors[:, order], self.tokens[:, order])
        torch.testing.assert_close(result["finger_relation"], baseline[:, order[1:] - 1])
        # Wrist intentionally uses the anatomical order, so it is not tested as a set.

    def test_translation_with_fixed_hand_tokens(self):
        baseline = self.run_model()["relation_update"]
        offset = torch.tensor([.3, -.2, .5])
        patches = dict(self.patches, xyz=self.patches["xyz"] + offset)
        shifted = self.model(self.points + offset, patches, self.anchors + offset, self.tokens)
        torch.testing.assert_close(baseline, shifted["relation_update"], atol=2e-6, rtol=1e-5)
        # This does not assert invariance of HandSpatialEncoder's absolute features.

    def test_query_context_broadcast(self):
        context = torch.randn(2, 24)
        a = self.run_model(query_context=context)["relation_update"]
        b = self.run_model(query_context=context[:, None])["relation_update"]
        c = self.run_model(query_context=context[:, None].expand(-1, 6, -1))["relation_update"]
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, c)

    def test_bfloat16_autocast_backward(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = self.run_model()
            loss = result["relation_update"].float().square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_inputs_fail(self):
        for kwargs in ({"surface_k": 0}, {"surface_radius": 0.}, {"token_channels": 25}, {"metric_scale": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HandSceneAttention(**kwargs)
        with self.assertRaises(ValueError):
            self.run_model(point_mask=torch.ones(2, 40))
        with self.assertRaises(ValueError):
            self.run_model(query_context=torch.zeros(2, 2, 24))
        self.anchors[0, 0, 0] = torch.nan
        with self.assertRaises(ValueError):
            self.run_model()


if __name__ == "__main__":
    unittest.main()
