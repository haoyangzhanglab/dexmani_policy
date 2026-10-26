import copy
import random
import tempfile
import unittest

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.vqvae import EMAVectorQuantizer
from dexmani_policy.deployment.runtime import PolicyRuntime, load_policy
from dexmani_policy.smoke_test import load_smoke_config
from dexmani_policy.training.validation import validate_config
from tests.helpers import checkpoint_fixture, tiny_policy


def host_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


class NumericsRuntimeTests(unittest.TestCase):
    def assert_host_equal(self, before):
        after = host_state()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1][0], after[1][0])
        np.testing.assert_array_equal(before[1][1], after[1][1])
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_vq_unhit_codes_stay_bounded_and_straight_through(self):
        torch.manual_seed(7)
        q = EMAVectorQuantizer(8, 4)
        initial = q.embed.clone()
        x = initial[0].repeat(8, 1).requires_grad_()
        for _ in range(20):
            y, _, loss = q(x)
            self.assertTrue(torch.isfinite(q.embed).all())
            self.assertLess(float(q.embed.norm(dim=1).max()), float(initial.norm(dim=1).max()) * 2)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
        before = copy.deepcopy(q.state_dict())
        q.eval()(x)
        torch.testing.assert_close(before, q.state_dict(), rtol=0, atol=0)

    def test_vq_kmeans_dead_codes_all_hits_and_restore(self):
        for kmeans in (False, True):
            for dead in (0, 2):
                q = EMAVectorQuantizer(4, 4, kmeans_init=kmeans, threshold_ema_dead_code=dead)
                q(torch.ones(8, 4) if kmeans else q.embed.clone())
                for _ in range(5):
                    q(torch.ones(8, 4))
                self.assertTrue(all(torch.isfinite(v).all() for v in q.state_dict().values()))
                other = EMAVectorQuantizer(4, 4, kmeans_init=not kmeans)
                other.load_state_dict(q.state_dict(), strict=True)
                torch.testing.assert_close(other.state_dict(), q.state_dict(), rtol=0, atol=0)
        # Loading historical zero counts must not rewrite their saved statistics.
        state = q.state_dict()
        state['cluster_size'].zero_()
        other.load_state_dict(state, strict=True)
        self.assertEqual(int(other.cluster_size.count_nonzero()), 0)

    def test_runtime_interleaving_reset_and_warmup(self):
        agent, _, info, obs = tiny_policy()
        before = host_state()
        a = PolicyRuntime(agent, info, device='cpu', seed=5)
        self.assert_host_equal(before)
        first, second = a.predict(obs), a.predict(obs)
        a.reset_episode()
        np.testing.assert_array_equal(a.predict(obs), first)
        b = PolicyRuntime(copy.deepcopy(agent), info, device='cpu', seed=9)
        b.predict(obs)
        b.reset_episode()
        b.warmup(samples=1)
        np.testing.assert_array_equal(a.predict(obs), second)
        a.reset_episode()
        a.predict(obs)
        a.warmup(samples=1)
        np.testing.assert_array_equal(a.predict(obs), second)
        self.assert_host_equal(before)
        b.reset_episode()
        self.assertFalse(np.array_equal(first, b.predict(obs)))

    def test_load_preserves_host_rng(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, info, obs = checkpoint_fixture(root)
            before = host_state()
            runtime = load_policy(info, device='cpu', seed=8)
            runtime.predict(obs)
            runtime.close()
            self.assert_host_equal(before)

    def test_rtc_keeps_autograd_under_runtime_inference_mode(self):
        agent, _, info, obs = tiny_policy()
        a = PolicyRuntime(agent, info, device='cpu', seed=5)
        prefix = np.full((3, 19), .2)
        first = a.predict(obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=1.)
        a.reset_episode()
        np.testing.assert_array_equal(first, a.predict(obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=1.))
        self.assertEqual(first.shape, (7, 19))
        a.reset_episode()
        self.assertFalse(np.array_equal(first, a.predict(obs)))

    def test_r3d_rejects_incompatible_normalization(self):
        cfg = load_smoke_config('r3d')
        for mode in ('identity', 'gaussian'):
            cfg.normalization.point_cloud = mode
            with self.assertRaisesRegex(ValueError, 'limits'):
                validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
