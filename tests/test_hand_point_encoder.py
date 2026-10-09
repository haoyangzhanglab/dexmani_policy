"""定向检查；在仓库根目录运行 python -m unittest discover -s tests。"""

import unittest

import torch

from dexmani_policy.agents.obs_encoder.pointcloud.interaction_encoder import (
    PointPatchInteractionEncoder,
    nearest_patch_evidence,
)
from dexmani_policy.agents.obs_encoder.proprio.hand_point_encoder import WristFingertipEncoder


def hand_inputs():
    wrist = torch.tensor([[0.3, -0.1, 0.45, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    offsets = torch.tensor([[[0.05, -0.03, 0.02], [0.10, -0.01, 0.01],
                             [0.11, 0.01, 0.00], [0.09, 0.03, 0.01], [0.07, 0.05, 0.02]]])
    return wrist, wrist[:, None, :3] + offsets


class HandPointEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = WristFingertipEncoder(token_channels=24).eval()
        self.wrist, self.fingertips = hand_inputs()

    def test_world_centers_and_real_sim_layouts(self):
        output = self.encoder(self.wrist, self.fingertips)
        flat = self.encoder(self.wrist, self.fingertips.flatten(1))
        self.assertEqual(output["hand_token"].shape, (1, 6, 24))
        self.assertEqual(self.encoder.out_shape, (6, 24))
        torch.testing.assert_close(output["hand_center"][:, 0], self.wrist[:, :3])
        torch.testing.assert_close(output["hand_center"][:, 1:], self.fingertips)
        torch.testing.assert_close(output["hand_token"], flat["hand_token"])

    def test_absolute_world_position_survives_fourier_period(self):
        translated = self.wrist.clone()
        shift = torch.tensor([0.64, 0.0, 0.0])  # 所有默认 Fourier 波长的整数周期。
        translated[:, :3] += shift
        before = self.encoder(self.wrist, self.fingertips)
        after = self.encoder(translated, self.fingertips + shift)
        torch.testing.assert_close(after["hand_center"], before["hand_center"] + shift)
        self.assertFalse(torch.allclose(before["hand_token"], after["hand_token"], atol=1e-4))

    def test_orientation_changes_features_without_rotating_centers(self):
        rotated = self.wrist.clone()
        rotated[:, 3:] = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])  # 绕世界 Z 轴 90°。
        before = self.encoder(self.wrist, self.fingertips)
        after = self.encoder(rotated, self.fingertips)
        torch.testing.assert_close(before["hand_center"], after["hand_center"])
        self.assertFalse(torch.allclose(before["hand_token"], after["hand_token"]))

    def test_coincident_points_keep_body_identity(self):
        coincident = self.wrist[:, None, :3].expand(-1, 5, -1)
        tokens = self.encoder(self.wrist, coincident)["hand_token"]
        for i in range(1, 6):
            self.assertFalse(torch.allclose(tokens[:, 0], tokens[:, i]))

    def test_pose_and_fingertips_receive_finite_gradients(self):
        wrist = self.wrist.clone().requires_grad_()
        tips = self.fingertips.clone().requires_grad_()
        tokens = self.encoder(wrist, tips)["hand_token"]
        (tokens * torch.randn_like(tokens)).sum().backward()
        for gradient in (wrist.grad[:, :3], wrist.grad[:, 3:], tips.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        for parameter in self.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_geometry_is_rejected(self):
        for rotation in (torch.zeros(6), torch.tensor([1., 0., 0., 2., 0., 0.])):
            wrist = self.wrist.clone()
            wrist[:, 3:] = rotation
            with self.assertRaises(ValueError):
                self.encoder(wrist, self.fingertips)
        invalid_tips = self.fingertips.clone()
        invalid_tips[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.encoder(self.wrist, invalid_tips)
        with self.assertRaisesRegex(ValueError, "fingertip_points"):
            self.encoder(self.wrist, self.fingertips[:, :4])
        with self.assertRaisesRegex(ValueError, "positive"):
            WristFingertipEncoder(max_wavelength=0.0)


class WorldInteractionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.encoder = PointPatchInteractionEncoder(token_channels=24).eval()
        self.wrist, self.fingertips = hand_inputs()
        offsets = torch.tensor([[[0.02, 0.00, 0.00], [0.05, 0.01, 0.00],
                                 [0.00, 0.04, 0.00], [0.00, 0.08, 0.02]]])
        self.cloud = self.wrist[:, None, :3] + offsets
        self.patches = {
            "patch_token": torch.randn(1, 2, 24),
            "local_token": torch.randn(1, 2, 24),
            "patch_center": self.cloud.reshape(1, 2, 2, 3).mean(2),
            "neighbor_idx": torch.tensor([[[0, 1], [2, 3]]]),
        }

    def test_world_geometry_does_not_follow_wrist_rotation(self):
        before = self.encoder(self.cloud, self.patches, self.wrist, self.fingertips,
                              return_intermediate=True)
        rotated = self.wrist.clone()
        rotated[:, 3:] = torch.tensor([0., 1., 0., -1., 0., 0.])
        after = self.encoder(self.cloud, self.patches, rotated, self.fingertips.flatten(1),
                             return_intermediate=True)
        expected = torch.tensor([[0.02, 0., 0.], [0., 0.04, 0.]])
        torch.testing.assert_close(after["surface_relative"][0, 0], expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(after["context_relative"][0, 0],
                                   self.patches["patch_center"][0] - self.wrist[0, :3])
        for name in ("surface_relative", "surface_distance", "context_relative", "interaction_center"):
            torch.testing.assert_close(before[name], after[name])
        self.assertFalse(torch.allclose(before["interaction_token"], after["interaction_token"]))

    def test_nearest_valid_member_and_no_observation(self):
        anchors = self.wrist[:, None, :3]
        members = self.cloud.reshape(1, 2, 2, 3)
        mask = torch.tensor([[[False, True], [False, False]]])
        delta, distance, observed = nearest_patch_evidence(anchors, members, mask)
        torch.testing.assert_close(delta[0, 0, 0], torch.tensor([0.05, 0.01, 0.]), atol=1e-6, rtol=0)
        self.assertFalse(observed[0, 0, 1])
        self.assertTrue(torch.isinf(distance[0, 0, 1]))
        torch.testing.assert_close(delta[0, 0, 1], torch.zeros(3))

        output = self.encoder(self.cloud, self.patches, self.wrist, self.fingertips,
                              member_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
                              return_intermediate=True)
        self.assertTrue(torch.isfinite(output["interaction_token"]).all())
        torch.testing.assert_close(output["near_gate"], torch.zeros(1, 6))
        torch.testing.assert_close(output["near_null_mass"], torch.ones(1, 6, 4))
        torch.testing.assert_close(output["context_null_mass"], torch.ones(1, 6, 4))

    def test_cpu_autocast_backward(self):
        for name in ("patch_token", "local_token"):
            self.patches[name].requires_grad_()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            tokens = self.encoder(self.cloud, self.patches, self.wrist, self.fingertips)["interaction_token"]
            loss = (tokens.float() * torch.randn_like(tokens.float())).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(tokens).all())
        for name in ("patch_token", "local_token"):
            gradient = self.patches[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        for parameter in self.encoder.hand_encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
