"""手部相对几何的坐标、空邻域与检索条件契约；不依赖点云采样扩展。"""

import unittest

import torch

from dexmani_policy.agents.obs_encoder.interaction.hand_scene_relation import (
    HandSceneRelationEncoder,
    nearest_patch_evidence,
)


class HandSceneRelationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.channels = 24
        self.pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0.]])
        self.fingertips = torch.zeros(1, 5, 3)

    def patches(self, pointcloud):
        count = pointcloud.shape[1]
        return {
            "patch_token": torch.randn(1, count, self.channels),
            "local_token": torch.randn(1, count, self.channels),
            "patch_center": pointcloud[..., :3].clone(),
            "neighbor_idx": torch.arange(count).reshape(1, count, 1),
        }

    def test_empty_near_support_blocks_null_and_projection_bias(self):
        model = HandSceneRelationEncoder(token_channels=self.channels).eval()
        with torch.no_grad():
            model.near.null_value.fill_(2.)
            model.near.out.bias.fill_(3.)
        cloud = torch.tensor([[[2., 0., 0.], [3., 0., 0.]]])
        outputs = model(cloud, self.patches(cloud), self.pose, self.fingertips,
                        return_intermediate=True)
        self.assertFalse(outputs["near_has_support"].any().item())
        torch.testing.assert_close(outputs["near_update"], torch.zeros(1, 6, self.channels))
        self.assertTrue(torch.isfinite(outputs["relation_update"]).all().item())

    def test_all_masked_members_keep_finite_updates_and_no_near_evidence(self):
        model = HandSceneRelationEncoder(token_channels=self.channels).eval()
        cloud = torch.tensor([[[0.01, 0., 0.], [0., 0.02, 0.]]])
        outputs = model(cloud, self.patches(cloud), self.pose, self.fingertips,
                        member_valid=torch.zeros(1, 2, 1, dtype=torch.bool),
                        return_intermediate=True)
        self.assertTrue(torch.isinf(outputs["min_observed_distance"]).all().item())
        self.assertFalse(outputs["surface_valid"].any().item())
        torch.testing.assert_close(outputs["near_update"], torch.zeros(1, 6, self.channels))
        torch.testing.assert_close(outputs["near_null_mass"], torch.ones(1, 6, 4))
        self.assertTrue(torch.isfinite(outputs["relation_update"]).all().item())

    def test_nearest_member_honors_validity(self):
        anchors = torch.zeros(1, 1, 3)
        members = torch.tensor([[[[0.01, 0., 0.], [0., 0.02, 0.]]]])
        relative, distance, observed = nearest_patch_evidence(
            anchors, members, torch.tensor([[[False, True]]])
        )
        torch.testing.assert_close(relative, torch.tensor([[[[0., 0.02, 0.]]]]))
        torch.testing.assert_close(distance, torch.tensor([[[0.02]]]))
        self.assertTrue(observed.all().item())

    def test_wrist_edges_rotate_vectors_but_keep_metric_distance(self):
        cloud = torch.tensor([[[0.1, 0., 0.]]])
        patches = self.patches(cloud)
        # Rz(+90 deg): columns x=(0,1,0), y=(-1,0,0).
        pose = torch.tensor([[0., 0., 0., 0., 1., 0., -1., 0., 0.]])
        wrist = HandSceneRelationEncoder(token_channels=self.channels, edge_frame="wrist").eval()
        base = HandSceneRelationEncoder(token_channels=self.channels, edge_frame="base").eval()
        wrist_outputs = wrist(cloud, patches, pose, self.fingertips, return_intermediate=True)
        base_outputs = base(cloud, patches, pose, self.fingertips, return_intermediate=True)
        expected_wrist = torch.tensor([0., -0.1, 0.]).expand(1, 6, 1, 3)
        expected_base = torch.tensor([0.1, 0., 0.]).expand(1, 6, 1, 3)
        torch.testing.assert_close(wrist_outputs["surface_relative"], expected_wrist)
        torch.testing.assert_close(wrist_outputs["context_relative"], expected_wrist)
        torch.testing.assert_close(base_outputs["surface_relative"], expected_base)
        torch.testing.assert_close(wrist_outputs["surface_distance"], base_outputs["surface_distance"])
        torch.testing.assert_close(wrist_outputs["interaction_center"], base_outputs["interaction_center"])

    def test_query_condition_is_separate_from_residual_hand_query(self):
        model = HandSceneRelationEncoder(token_channels=self.channels).eval()
        cloud = torch.tensor([[[0.01, 0., 0.], [0., 0.02, 0.]]])
        patches = self.patches(cloud)
        plain = model(cloud, patches, self.pose, self.fingertips)
        conditioned = model(cloud, patches, self.pose, self.fingertips,
                            query_context=torch.randn(1, 6, self.channels))
        torch.testing.assert_close(plain["hand_query"], conditioned["hand_query"])
        self.assertFalse(torch.allclose(plain["relation_update"], conditioned["relation_update"]))
        shared_context = torch.randn(1, self.channels)
        shared = model(cloud, patches, self.pose, self.fingertips,
                       query_context=shared_context)
        expanded = model(cloud, patches, self.pose, self.fingertips,
                         query_context=shared_context[:, None].expand(-1, 6, -1))
        torch.testing.assert_close(shared["relation_update"], expanded["relation_update"])

    def test_rejects_degenerate_wrist_rotation(self):
        model = HandSceneRelationEncoder(token_channels=self.channels)
        cloud = torch.tensor([[[0.01, 0., 0.]]])
        with self.assertRaisesRegex(ValueError, "rot6d"):
            model(cloud, self.patches(cloud), torch.zeros(1, 9), self.fingertips)


if __name__ == "__main__":
    unittest.main()
