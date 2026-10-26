"""Explicit CUDA checks: python -m tests.cuda_checks (no CPU fallback or skips)."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection

from dexmani_policy.agents.hand_prior.vqvae import EMAVectorQuantizer
from dexmani_policy.agents.obs_encoder.text import CLIPTextEncoder
from dexmani_policy.deployment.runtime import load_policy
from tests.helpers import checkpoint_fixture
from tests.test_numerics_runtime import NumericsRuntimeTests, host_state
from tests.test_policy_contracts import PolicyContractsTests, hand_artifacts
from tests.test_training import TrainingTests


class CudaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA checks require an actual CUDA device')
        torch.cuda.init()

    def test_nine_policies(self):
        checks = PolicyContractsTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_artifacts(root)
            for name in ('dp', 'dp3', 'sat', 'r3d', 'maniflow', 'multitask_dit',
                         'dqrise', 'lamp', 'latent_flow'):
                with self.subTest(policy=name):
                    checks.check_policy(name, root, device='cuda:0')

    def test_runtime_load_interleaving_warmup_and_rtc(self):
        checks = NumericsRuntimeTests()
        with tempfile.TemporaryDirectory() as tmp:
            _, _, info, obs = checkpoint_fixture(tmp)
            before, cuda_before = host_state(), torch.cuda.get_rng_state_all()
            a = load_policy(info, device='cuda:0', seed=5)
            try:
                first, second = a.predict(obs), a.predict(obs)
                a.reset_episode()
                np.testing.assert_array_equal(a.predict(obs), first)
                b = load_policy(info, device='cuda:0', seed=9)
                try:
                    b.predict(obs)
                    b.reset_episode()
                    b.warmup(samples=1)
                    np.testing.assert_array_equal(a.predict(obs), second)
                finally:
                    b.close()
                a.reset_episode()
                a.predict(obs)
                a.warmup(samples=1)
                np.testing.assert_array_equal(a.predict(obs), second)
                prefix = np.full((3, 19), .2)
                a.reset_episode()
                guided = a.predict(obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=1.)
                a.reset_episode()
                np.testing.assert_array_equal(guided, a.predict(
                    obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=1.))
                a.reset_episode()
                self.assertFalse(np.array_equal(guided, a.predict(obs)))
            finally:
                a.close()
            checks.assert_host_equal(before)
            for old, current in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True):
                self.assertTrue(torch.equal(old, current))

    def test_training_and_resume(self):
        TrainingTests().check_training('cuda:0')

    def test_clip_explicit_device_and_vq(self):
        model = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=8, intermediate_size=16,
                    num_hidden_layers=1, num_attention_heads=2, projection_dim=4, vocab_size=16))
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
             patch('transformers.CLIPTextModelWithProjection.from_pretrained', return_value=model):
            self.assertEqual(CLIPTextEncoder().device, torch.device('cpu'))
            self.assertEqual(CLIPTextEncoder(device='cuda:0').device, torch.device('cuda:0'))
        q = EMAVectorQuantizer(8, 4).cuda()
        initial = q.embed.clone()
        x = initial[0].repeat(8, 1).requires_grad_()
        for _ in range(20):
            y, _, _ = q(x)
            self.assertTrue(torch.isfinite(q.embed).all())
            self.assertLess(q.embed.norm(dim=1).max(), initial.norm(dim=1).max() * 2)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CudaChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
