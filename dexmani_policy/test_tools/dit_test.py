"""CPU regressions for the standalone SmolVLA-style action expert."""

import copy
import unittest
from unittest.mock import patch

import torch

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow


class DiTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = DiT(
            horizon=5, action_dim=3, n_obs_steps=2, obs_token_dim=7,
            hidden_dim=24, n_layers=4, n_head=4, n_kv_head=2,
            head_dim=8, timestep_embed_dim=16,
        )
        self.actions = torch.randn(2, 5, 3)
        self.context = torch.randn(2, 6, 7)
        self.time = torch.tensor([0.2, 0.7])

    def test_conditioning_and_flow_gradients(self):
        actions = self.actions.clone().requires_grad_()
        context = self.context.clone().requires_grad_()
        output = self.model(actions, self.time, context)
        self.assertEqual(output.shape, actions.shape)
        self.assertFalse(torch.allclose(output, self.model(actions, self.time + 0.1, context)))
        changed_context = torch.randn_like(context)
        self.assertFalse(torch.allclose(output, self.model(actions, self.time, changed_context)))

        decoder = RectifiedFlow(self.model)
        loss, metrics = decoder.compute_loss(context, self.actions, training_pair=(
            actions, self.time, torch.randn_like(actions),
        ))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_action", metrics)
        loss.backward()
        for name, gradient in (("actions", actions.grad), ("context", context.grad)):
            with self.subTest(gradient=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    @torch.no_grad()
    def test_causal_actions_and_prefix_features(self):
        self.model.eval()
        output = self.model(self.actions, self.time, self.context)
        perturbed = self.actions.clone()
        perturbed[:, 3:] += 20 * torch.randn_like(perturbed[:, 3:])
        changed = self.model(perturbed, self.time, self.context)
        torch.testing.assert_close(output[:, :3], changed[:, :3], rtol=0, atol=1e-7)
        self.assertFalse(torch.allclose(output[:, 3:], changed[:, 3:]))
        full = self.model.forward_prefix_features(self.actions, self.time, self.context)
        prefix = self.model.forward_prefix_features(self.actions[:, :3], self.time, self.context)
        torch.testing.assert_close(full[:, :3], prefix, rtol=1e-5, atol=1e-7)

    @torch.no_grad()
    def test_context_cache_and_euler_agree(self):
        self.assertIsNone(self.model.prepare_context_kv(self.context))
        self.model.eval()
        cache = self.model.prepare_context_kv(self.context)
        original = tuple(tuple(tensor.clone() for tensor in pair) for pair in cache)
        for time in (0.0, 0.4, 0.9):
            actions = torch.randn_like(self.actions)
            expected = self.model(actions, time, self.context)
            actual = self.model(actions, time, self.context, context_kv=cache)
            torch.testing.assert_close(actual, expected)
        for pair, expected_pair in zip(cache, original):
            for tensor, expected in zip(pair, expected_pair):
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        decoder = RectifiedFlow(self.model, num_inference_steps=3).eval()
        expected = decoder.predict_action(
            self.context, self.actions.shape, generator=torch.Generator().manual_seed(9),
        )
        with patch.object(self.model, "prepare_context_kv", return_value=None):
            actual = decoder.predict_action(
                self.context, self.actions.shape, generator=torch.Generator().manual_seed(9),
            )
        torch.testing.assert_close(actual, expected)

    def test_bfloat16_paths(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.actions, self.time, self.context)
        attention = torch.nn.functional.scaled_dot_product_attention

        def check_attention_precision(query, key, value, **kwargs):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            for tensor in (query, key, value):
                self.assertEqual(tensor.dtype, torch.float32)
            return attention(query, key, value, **kwargs)

        with patch(
            "dexmani_policy.agents.action_decoders.backbones.dit.F.scaled_dot_product_attention",
            side_effect=check_attention_precision,
        ) as attention_spy, torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.model(self.actions, self.time, self.context)
            loss = output.float().square().mean()
            with torch.no_grad():
                cache = self.model.prepare_context_kv(self.context)
                cached = self.model(self.actions, self.time, self.context, context_kv=cache)
            torch.testing.assert_close(cached, output, rtol=0, atol=0)
        self.assertGreater(attention_spy.call_count, 0)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.float(), expected, rtol=0.05, atol=0.005)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.input_embedder.weight.grad).all())

        reduced = copy.deepcopy(self.model).to(torch.bfloat16)
        with torch.no_grad():
            context = self.context.bfloat16()
            cache = reduced.prepare_context_kv(context)
            output = reduced(self.actions.bfloat16(), self.time, context, context_kv=cache)
            uncached = reduced(self.actions.bfloat16(), self.time, context)
        torch.testing.assert_close(output, uncached, rtol=0, atol=0)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.float(), expected, rtol=0.05, atol=0.005)

    def test_rtc_guidance_uses_action_gradients(self):
        decoder = RectifiedFlow(self.model, num_inference_steps=3).eval()
        kwargs = dict(cond=self.context, action_shape=self.actions.shape)
        unguided = decoder.predict_action(
            **kwargs, generator=torch.Generator().manual_seed(11),
        )
        prefix = torch.full((2, 2, 3), 5.0)
        guided = decoder.predict_action(
            **kwargs, generator=torch.Generator().manual_seed(11),
            rtc_prefix=prefix, delay_steps=0, rtc_guidance_cap=0.5,
        )
        self.assertTrue(torch.isfinite(guided).all())
        self.assertLess(
            (guided[:, :2] - prefix).square().mean().item(),
            (unguided[:, :2] - prefix).square().mean().item(),
        )


if __name__ == "__main__":
    unittest.main()
