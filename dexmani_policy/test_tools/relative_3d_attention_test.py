"""Small-tensor checks for zero initialization and linear relative geometry values."""

import math
import unittest

import torch
from torch.nn import functional as F

from dexmani_policy.agents.obs_encoder.interaction.relative_3d_attention import Relative3DCrossAttention


def reference(model, body, scene, anchors, xyz, *, pairwise):
    b, n, _ = scene.shape
    q = model.q(model.query_norm(body)).reshape(b, 6, 4, 48).transpose(1, 2)
    k, v = model.kv(model.memory_norm(scene)).reshape(b, n, 2, 4, 48).permute(2, 0, 3, 1, 4)
    bias = model.relative_bias(anchors, xyz).to(q.dtype)
    if pairwise:
        weights = (q @ k.transpose(-2, -1) / math.sqrt(48) + bias).softmax(-1)
        delta = (xyz[:, None] - anchors[:, :, None]) / 0.10
        geometry = model.relative_value(delta).reshape(b, 6, n, 4, 48).permute(0, 3, 1, 2, 4)
        context = (weights[..., None] * (v[:, :, None] + geometry)).sum(-2)
    else:
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
    context = context.transpose(1, 2).reshape(b, 6, 192)
    return model.norm(body.float() + model.out(context).float())


class RelativeAttentionTest(unittest.TestCase):
    def test_zero_initialization_and_pairwise_equivalence(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            torch.manual_seed(42)
            model = Relative3DCrossAttention().to(device)
            inputs = [torch.randn(shape, device=device, requires_grad=True) for shape in
                      ((2, 6, 192), (2, 7, 192), (2, 6, 3), (2, 7, 3))]
            probe = torch.randn_like(inputs[0])
            self.assertEqual(model.relative_value.weight.numel(), 576)
            for bf16 in (False, True):
                if bf16 and device == "cuda" and not torch.cuda.is_bf16_supported():
                    continue
                with self.subTest(device=device, bf16=bf16):
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                        output = model(*inputs)
                        baseline = reference(model, *inputs, pairwise=False)
                    self.assertEqual(output.shape, (2, 6, 192))
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(output, baseline, rtol=1e-5, atol=1e-6)
                    grad, = torch.autograd.grad((output * probe).sum(), model.relative_value.weight)
                    self.assertTrue(torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0)
            with torch.no_grad():
                model.relative_value.weight.normal_(std=0.01)
            output = model(*inputs)
            expected = reference(model, *inputs, pairwise=True)
            torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
            targets = (model.relative_value.weight, *inputs[2:])
            actual_grads = torch.autograd.grad((output * probe).sum(), targets)
            expected_grads = torch.autograd.grad((expected * probe).sum(), targets)
            for actual, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=5e-5)
            order = torch.randperm(7, device=device)
            with torch.no_grad():
                permuted = model(inputs[0], inputs[1][:, order], inputs[2], inputs[3][:, order])
            torch.testing.assert_close(permuted, output, rtol=2e-5, atol=2e-6)
            print(f"PASS {device}: zero-init FP32/BF16, gradients, pairwise equivalence and scene permutation")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
