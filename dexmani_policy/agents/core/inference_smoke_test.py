"""Synthetic shared inference contracts; no external data or pretrained weights.

Run: python -m dexmani_policy.agents.core.inference_smoke_test
"""

from pathlib import Path
import unittest
from unittest.mock import patch

import hydra
import torch
from torch import nn

from dexmani_policy.agents.action_decoders.consistency_flow import ConsistencyFlowMatch
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.smoke_test import load_config, _prepare_dqrise_codebook


class Velocity(nn.Module):
    def forward(self, x, timestep, context, target_t=None):
        return torch.ones_like(x) * 0.25


def small_policy(name):
    """Scale only test fixtures; exercise each policy's real observation/decoder path."""
    cfg = load_config(name)
    agent_cfg = cfg.agent
    agent_cfg.num_inference_steps = 10
    if "num_points" in agent_cfg:
        agent_cfg.num_points = 64
    if name in ("dp", "dp3", "dqrise"):
        agent_cfg.down_dims = [16, 32]
        agent_cfg.diffusion_step_embed_dim = 16
        agent_cfg.n_groups = 4
        agent_cfg.num_training_steps = 20
    if name == "dp":
        agent_cfg.rgb_backbone_name = "resnet"
        agent_cfg.rgb_backbone_config = dict(
            model_name="resnet18", weights=None, tune_mode="full",
            norm_mode="group_norm", image_size=[32, 32],
        )
    if name in ("maniflow", "sat"):
        agent_cfg.hidden_dim = 32
        agent_cfg.n_layers = 2
        agent_cfg.n_head = 4
    if name == "sat":
        agent_cfg.pc_encoder_config = dict(
            stem_channels=16, token_channels=32, num_patches=8,
            patch_neighbors=[4, 8], use_patch_self_attn=True,
            patch_attn_layers=1, patch_attn_heads=4,
        )
    if name == "r3d":
        agent_cfg.pc_encoder_config = dict(
            pc_model="eva02_tiny_patch14_224", embed_dim=32, num_group=8,
            group_size=8, pc_in_channels=6, use_pretrained_weights=False,
        )
        agent_cfg.embedding_dim = 32
        agent_cfg.depth = 1
        agent_cfg.num_heads = 4
        agent_cfg.mlp_dim = 64
        agent_cfg.timestep_embed_dim = 16
    obs = {"joint_state": torch.rand(2, cfg.n_obs_steps, 19)}
    if name == "dp":
        obs["rgb"] = torch.rand(2, cfg.n_obs_steps, 3, 32, 32)
    else:
        obs["point_cloud"] = torch.rand(2, cfg.n_obs_steps, 64, 6)
    normalizer = LinearNormalizer()
    normalizer.fit({
        **{k: v for k, v in obs.items() if k != "rgb"},
        "action": torch.rand(2, cfg.horizon, agent_cfg.action_dim),
    }, mode="limits")
    codebook = _prepare_dqrise_codebook(cfg, normalizer)
    if codebook is not None:
        agent_cfg.codebook_path = codebook
    try:
        agent = hydra.utils.instantiate(agent_cfg).eval()
        agent.load_normalizer_from_dataset(normalizer)
    finally:
        if codebook is not None:
            Path(codebook).unlink()
    return cfg, agent, obs


class DecoderInferenceTest(unittest.TestCase):
    def test_override_validation_and_persistent_defaults(self):
        for cls in (ConsistencyFlowMatch, RectifiedFlow, Diffusion):
            with self.subTest(decoder=cls.__name__):
                decoder = cls(Velocity(), num_inference_steps=10)
                cond, template = torch.zeros(2, 8, 16), torch.zeros(2, 16, 19)
                for override, calls in ((4, 4), (None, 10), (1, 1)):
                    with patch.object(decoder.model, "forward", wraps=decoder.model.forward) as forward:
                        result = decoder.predict_action(cond, template, inference_steps=override)
                        self.assertEqual(forward.call_count, calls)
                        self.assertEqual(result.shape, template.shape)
                        self.assertTrue(torch.isfinite(result).all())
                    self.assertEqual(decoder.num_inference_steps, 10)
                for invalid in (0, -1, 1.5, True, "4"):
                    with self.assertRaisesRegex(ValueError, "inference_steps"):
                        decoder.predict_action(cond, template, inference_steps=invalid)
                if isinstance(decoder, ConsistencyFlowMatch):
                    self.assertEqual(decoder.denoise_timesteps, 10)
                    self.assertEqual(decoder.time_sampler.num_steps, 10)

    def test_flow_training_grid_independent_of_default_and_runtime_nfe(self):
        expected = None
        for nfe in (1, 4, 10):
            decoder = RectifiedFlow(Velocity(), num_inference_steps=nfe, num_flow_train_timesteps=7)
            torch.manual_seed(9)
            samples = decoder.time_sampler.sample(512, "discrete", "cpu")
            if expected is None:
                expected = samples
            torch.testing.assert_close(samples, expected, rtol=0, atol=0)
            decoder.predict_action(torch.zeros(2, 8), torch.zeros(2, 16, 19), inference_steps=2)
            self.assertEqual(decoder.time_sampler.num_steps, 7)
            self.assertEqual(decoder.num_flow_train_timesteps, 7)
            self.assertEqual(decoder.num_inference_steps, nfe)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "num_flow_train_timesteps"):
                RectifiedFlow(Velocity(), num_flow_train_timesteps=invalid)


class PolicyInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _check_real_policy_predict_and_execution(self, device):
        for name in ("maniflow", "dp", "dp3", "r3d", "sat", "dqrise"):
            with self.subTest(policy=name):
                cfg, agent, obs = small_policy(name)
                agent.to(device)
                obs = {key: value.to(device) for key, value in obs.items()}
                with patch.object(agent.action_decoder.model, "forward", wraps=agent.action_decoder.model.forward) as forward:
                    result = agent.predict_action(obs, inference_steps=4)
                    self.assertEqual(forward.call_count, 4)
                self.assertEqual(result["control_action"].shape, (2, cfg.n_action_steps, agent.control_action_dim))
                self.assertTrue(torch.isfinite(result["pred_action"]).all())
                start = cfg.n_obs_steps - 1
                torch.testing.assert_close(
                    result["control_action"],
                    result["pred_action"][:, start:start + cfg.n_action_steps, :agent.control_action_dim],
                    rtol=0, atol=0,
                )
                self.assertEqual(agent.action_decoder.num_inference_steps, 10)
                with torch.no_grad():
                    cond, _ = agent._build_cond(obs)
                    self.assertEqual(agent.predict_action_from_cond(cond, inference_steps=1)["control_action"].shape,
                                     result["control_action"].shape)

    def test_real_policy_predict_cpu(self):
        self._check_real_policy_predict_and_execution("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA policy inference NOT VERIFIED")
    def test_real_policy_predict_cuda(self):
        self._check_real_policy_predict_and_execution("cuda")

    def test_multitask_decoder_branches(self):
        from dexmani_policy.agents.core.multi_task import MultiTaskAgent

        class TextFixture(nn.Module):
            embed_dim = 8

            def forward(self, task_texts):
                return torch.ones(len(task_texts), 1, self.embed_dim)

        for decoder_type in ("diffusion", "rectified_flow"):
            with self.subTest(decoder=decoder_type), patch(
                "dexmani_policy.agents.core.multi_task.CLIPTextEncoder", return_value=TextFixture(),
            ):
                agent = MultiTaskAgent(
                    rgb_backbone_name="resnet",
                    rgb_backbone_config=dict(model_name="resnet18", weights=None,
                                             norm_mode="group_norm", image_size=[32, 32]),
                    n_emb=32, num_heads=4, n_layers=1, action_decoder_type=decoder_type,
                    num_training_steps=20, num_inference_steps=10,
                    flow_num_inference_steps=10, num_flow_train_timesteps=7,
                ).eval()
                obs = {"rgb": torch.rand(2, 2, 3, 32, 32),
                       "joint_state": torch.rand(2, 2, 19), "task_text": ["one", "two"]}
                normalizer = LinearNormalizer()
                normalizer.fit({"joint_state": obs["joint_state"], "action": torch.rand(2, 16, 19)}, mode="limits")
                agent.load_normalizer_from_dataset(normalizer)
                with patch.object(agent.action_decoder.model, "forward", wraps=agent.action_decoder.model.forward) as forward:
                    result = agent.predict_action(obs, inference_steps=4)
                    self.assertEqual(forward.call_count, 4)
                self.assertEqual(result["control_action"].shape, (2, 8, 19))
                self.assertEqual(agent.action_decoder.num_inference_steps, 10)
                if decoder_type == "rectified_flow":
                    self.assertEqual(agent.action_decoder.time_sampler.num_steps, 7)

    def test_config_contract(self):
        for name in ("maniflow", "dp", "dp3", "r3d", "sat", "dqrise", "multitask_dit"):
            cfg = load_config(name)
            self.assertIn("inference_steps", cfg.eval)
            self.assertIn("inference_steps_list", cfg.eval)
            self.assertNotIn("denoise_steps", cfg.eval)
            self.assertNotIn("denoise_timesteps_list", cfg.eval)


if __name__ == "__main__":
    unittest.main()
