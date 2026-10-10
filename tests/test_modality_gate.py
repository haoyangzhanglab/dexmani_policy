"""独立逐指 gain 的状态条件性、缺测语义和非竞争性。"""

import unittest

import torch

from dexmani_policy.agents.obs_encoder.interaction.modality_gate import FingerModalityGate


class ModalityGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.gate = FingerModalityGate(24)
        self.hand = torch.randn(2, 6, 24)
        self.touch = torch.randn(2, 6, 24)
        self.scene = torch.randn(2, 24)
        self.force = torch.zeros(2, 5, 3)
        self.valid = torch.ones(2, 5, dtype=torch.bool)

    def call(self):
        return self.gate(self.hand, self.touch, self.scene, self.force, self.valid)

    def test_invalid_nan_cannot_pollute_geometry_gain(self):
        self.valid[:, 2] = False
        self.force[:, 2] = float("nan")
        self.touch[:, 3] = float("nan")
        gains = self.call()
        self.assertTrue(all(torch.isfinite(x).all() for x in gains.values()))
        self.assertEqual(gains["tactile_gain"][:, 3].count_nonzero().item(), 0)
        torch.testing.assert_close(gains["geometry_gain"][:, 3], torch.ones(2, 1))

    def test_two_modalities_can_both_be_amplified(self):
        with torch.no_grad():
            self.gate.gate[-1].bias.fill_(1.0)
        gains = self.call()
        self.assertTrue((gains["geometry_gain"][:, 1:] > 1).all())
        self.assertTrue((gains["tactile_gain"][:, 1:] > 1).all())
        torch.testing.assert_close(gains["geometry_gain"][:, 0], torch.ones(2, 1))
        self.assertEqual(gains["tactile_gain"][:, 0].count_nonzero().item(), 0)

    def test_bfloat16_preserves_small_updates_around_unit_gain(self):
        # BF16 sigmoid rounds these learned nonzero logits to gain=1 even
        # though backward remains nonzero. Finite-loss tests cannot catch it.
        with torch.no_grad():
            self.gate.gate[-1].bias.copy_(torch.tensor([0.002, -0.002]))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            gains = self.call()
        geometry = gains["geometry_gain"][:, 1:]
        tactile = gains["tactile_gain"][:, 1:]
        self.assertTrue((geometry > 1).all())
        self.assertTrue((tactile < 1).all())
        expected = 2 * self.gate.gate[-1].bias.to(torch.bfloat16).float().sigmoid()
        torch.testing.assert_close(geometry, expected[0].expand_as(geometry))
        torch.testing.assert_close(tactile, expected[1].expand_as(tactile))
        self.assertEqual(geometry.dtype, torch.float32)
        (geometry.sum() + tactile.sum()).backward()
        grad = self.gate.gate[-1].bias.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad > 0).all())

    def test_gate_can_use_sensor_amplitude_per_finger(self):
        with torch.no_grad():
            self.gate.gate[0].weight.zero_()
            self.gate.gate[0].bias.zero_()
            self.gate.gate[0].weight[0, -4] = 1.0  # 保留的规范化 force-x 通道。
            self.gate.gate[-1].weight.zero_()
            self.gate.gate[-1].weight[:, 0] = torch.tensor([1.0, -1.0])
        self.force[:, 0, 0] = 2.0
        self.force[:, 1, 0] = -1.0
        gains = self.call()
        self.assertTrue((gains["geometry_gain"][:, 1] > gains["geometry_gain"][:, 2]).all())
        self.assertTrue((gains["tactile_gain"][:, 1] < gains["tactile_gain"][:, 2]).all())


if __name__ == "__main__":
    unittest.main()
