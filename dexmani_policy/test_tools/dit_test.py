"""CPU regressions for the standalone SmolVLA-style action expert."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hydra
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.action_io import SATActionIO
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.training.common.config import load_policy_config


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

    def test_backbones_reject_condition_batch_broadcast(self):
        for cls in (DiT, DiTX):
            model = cls(horizon=5, action_dim=3, n_obs_steps=2, obs_token_dim=7,
                        hidden_dim=24, n_head=4, n_layers=2)
            with self.assertRaisesRegex(ValueError, 'same batch size'):
                model(self.actions, self.time, self.context[:1])

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
    def test_causal_actions(self):
        self.model.eval()
        output = self.model(self.actions, self.time, self.context)
        perturbed = self.actions.clone()
        perturbed[:, 3:] += 20 * torch.randn_like(perturbed[:, 3:])
        changed = self.model(perturbed, self.time, self.context)
        torch.testing.assert_close(output[:, :3], changed[:, :3], rtol=0, atol=1e-7)
        self.assertFalse(torch.allclose(output[:, 3:], changed[:, 3:]))

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
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            for tensor in (query, key, value):
                self.assertEqual(tensor.dtype, torch.bfloat16)
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
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
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

    def make_sat_model(self):
        return DiT(
            horizon=5, action_dim=19, n_obs_steps=2, obs_token_dim=7,
            hidden_dim=24, n_layers=4, n_head=4, n_kv_head=2,
            head_dim=8, timestep_embed_dim=16, qk_norm=True,
            action_io=SATActionIO(horizon=5, hidden_dim=24),
        )

    def test_sat_initialization_and_training(self):
        action_io = SATActionIO(horizon=5, hidden_dim=24)
        original = copy.deepcopy(action_io.state_dict())
        model = DiT(
            horizon=5, action_dim=19, n_obs_steps=2, obs_token_dim=7,
            hidden_dim=24, n_layers=4, n_head=8, n_kv_head=2,
            timestep_embed_dim=16, action_io=action_io,
        )
        # SAT has no temporal RoPE, so its head width (24 / 8 = 3) may be odd.
        with self.assertRaisesRegex(ValueError, "even for RoPE"):
            DiT(horizon=5, action_dim=3, n_obs_steps=2, obs_token_dim=7,
                hidden_dim=24, n_head=8)
        for name, value in action_io.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertFalse(hasattr(model, "input_embedder"))
        self.assertFalse(hasattr(model, "output_proj"))
        actions = torch.randn(2, 5, 19)
        output = model(actions, self.time, self.context)
        self.assertEqual(output.shape, actions.shape)
        torch.testing.assert_close(output, torch.zeros_like(actions), rtol=0, atol=0)
        decoder = RectifiedFlow(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # The first update opens SAT's zero head; the second must train upstream.
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = decoder.compute_loss(self.context, actions)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                with self.subTest(step=step, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    if step:
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_sat_joint_communication_and_permutation(self):
        model = self.make_sat_model().eval()
        actions = torch.randn(2, 5, 19, requires_grad=True)
        hidden = model._forward_hidden(actions, self.time, self.context)
        self.assertEqual(hidden.shape, (2, 19, 24))
        for output_joint, input_joint in ((0, 18), (18, 0)):
            gradient = torch.autograd.grad(
                hidden[:, output_joint].square().sum(), actions, retain_graph=True,
            )[0]
            self.assertGreater(gradient[:, :, input_joint].abs().sum().item(), 0)
        # Joint identities move with the trajectory, so token order is immaterial.
        permutation = torch.randperm(19)
        permuted = copy.deepcopy(model)
        permuted.action_io.robot_ids = model.action_io.robot_ids[permutation]
        permuted.action_io.joint_ids = model.action_io.joint_ids[permutation]
        changed = permuted._forward_hidden(actions[:, :, permutation], self.time, self.context)
        torch.testing.assert_close(changed, hidden[:, permutation], rtol=1e-5, atol=1e-7)

    @torch.no_grad()
    def test_observation_set_order_and_frame_identity(self):
        for model, actions in (
            (self.model.eval(), self.actions),
            (self.make_sat_model().eval(), torch.randn(2, 5, 19)),
        ):
            # Learned frame identities may differ after training.
            model.context_frame_pos_embed.normal_(std=0.2)
            original = model._forward_hidden(actions, self.time, self.context)
            within_frame = self.context[:, [1, 0, 2, 4, 3, 5]]
            actual = model._forward_hidden(actions, self.time, within_frame)
            torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-7)
            swapped_frames = self.context[:, [3, 4, 5, 0, 1, 2]]
            self.assertFalse(torch.allclose(
                original, model._forward_hidden(actions, self.time, swapped_frames),
            ))

    def test_sat_cache_gradients_bfloat16_and_rtc(self):
        model = self.make_sat_model().eval()
        # Exercise a nonzero learned velocity, not only the initial identity flow.
        torch.nn.init.normal_(model.action_io.final_layer.ffn_final.fc2.weight, std=0.02)
        actions = torch.randn(2, 5, 19, requires_grad=True)
        with torch.no_grad():
            cache = model.prepare_context_kv(self.context)
        expected = model(actions, self.time, self.context)
        actual = model(actions, self.time, self.context, context_kv=cache)
        torch.testing.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected.square().sum(), actions)[0]
        actual_grad = torch.autograd.grad(actual.square().sum(), actions)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertGreater(actual_grad.abs().sum().item(), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cache_bf16 = model.prepare_context_kv(self.context)
            bf16 = model(actions, self.time, self.context, context_kv=cache_bf16)
            torch.testing.assert_close(bf16, model(actions, self.time, self.context))
            bf16.float().square().mean().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        torch.testing.assert_close(bf16.float(), expected, rtol=0.1, atol=0.01)
        decoder = RectifiedFlow(model, num_inference_steps=3).eval()
        kwargs = dict(cond=self.context, action_shape=actions.shape)
        unguided = decoder.predict_action(**kwargs, generator=torch.Generator().manual_seed(11))
        with patch.object(model, "prepare_context_kv", return_value=None):
            uncached = decoder.predict_action(**kwargs, generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(unguided, uncached)
        prefix = torch.full((2, 2, 19), 5.0)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            guided = decoder.predict_action(
                **kwargs, generator=torch.Generator().manual_seed(11),
                rtc_prefix=prefix, delay_steps=0, rtc_guidance_cap=0.5,
            )
        self.assertTrue(torch.isfinite(guided).all())
        self.assertLess(
            (guided[:, :2] - prefix).square().mean().item(),
            (unguided[:, :2] - prefix).square().mean().item(),
        )

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

    @torch.no_grad()
    def test_sat_config_and_checkpoint_round_trip(self):
        for kind, backbone_class in (("ditx", DiTX), ("dit", DiT)):
            with self.subTest(backbone=kind):
                overrides = [] if kind == "ditx" else ["agent.backbone_type=dit"]
                cfg = load_policy_config("sat", overrides)
                self.assertEqual(cfg.agent.backbone_type, kind)
                config = {"agent": OmegaConf.to_container(cfg.agent, resolve=True)}
                # Real encoder construction with a tiny, randomly initialized ViT.
                config["agent"].update(
                    horizon=5, n_action_steps=2, hidden_dim=24, n_layers=2,
                    n_head=4, timestep_embed_dim=16, p_drop_attn=0.0,
                    pc_encoder_config=dict(channels=24, num_patches=2, group_size=2,
                                           depth=1, num_heads=4),
                    rgb_backbone_name="dino",
                    rgb_backbone_config=dict(
                        image_size=[32, 32], tune_mode="full",
                        architecture=dict(model_type="dinov2", hidden_size=24,
                                          num_hidden_layers=1, num_attention_heads=4,
                                          image_size=32, patch_size=16),
                    ),
                )
                policy = hydra.utils.instantiate(config["agent"], _convert_="all").eval()
                self.assertIsInstance(policy.action_decoder.model, backbone_class)
                self.assertEqual(policy.action_decoder.time_shift_alpha, 3.0)
                config["agent"].update(policy.effective_config)
                self.assertEqual(config["agent"]["backbone_type"], kind)
                config["normalization_dims"] = {"action": 19}
                config["normalization"] = {"action": "limits"}
                policy.configure_normalizer({"action": 19}, {"action": (19,)})
                torch.nn.init.normal_(
                    policy.action_decoder.model.action_io.final_layer.ffn_final.fc2.weight,
                    std=0.02,
                )
                actions = torch.randn(2, 5, 19)
                context = torch.randn(2, 6, policy.obs_encoder.out_dim)
                expected = policy.action_decoder.model(actions, self.time, context)
                self.assertGreater(expected.abs().sum().item(), 0)
                checkpoint = Checkpoint(
                    global_step=0, model_state=policy.state_dict(), ema_model_state=None,
                    optimizer_state={}, scheduler_state={}, rng_state={}, effective_config=config,
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = save_checkpoint(Path(directory) / "policy.pt", checkpoint)
                    saved = load_checkpoint(path)
                restored = restore_policy(saved.effective_config, saved.model_state, "cpu")
                self.assertIsInstance(restored.action_decoder.model, backbone_class)
                torch.testing.assert_close(restored.state_dict(), policy.state_dict())
                actual = restored.action_decoder.model(actions, self.time, context)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
