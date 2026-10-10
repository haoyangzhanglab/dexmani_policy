"""Bounded CPU integration checks; reference sampling does not validate PyTorch3D."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dexmani_policy.agents.core.interaction_flow import InteractionFlowAgent
from dexmani_policy.agents.normalization import validate_normalizer_state
from dexmani_policy.utils.validation import validate_observation_fields


class ReferencePointOps:
    """Small deterministic FPS/KNN used only to exercise the policy on CPU."""

    @staticmethod
    def sample_farthest_points(points, K):
        batch_size, count, _ = points.shape
        indices = torch.zeros(batch_size, K, dtype=torch.long, device=points.device)
        distance = torch.full((batch_size, count), float("inf"), device=points.device)
        rows = torch.arange(batch_size, device=points.device)
        farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
        for index in range(K):
            indices[:, index] = farthest
            delta = points - points[rows, farthest, None]
            distance = torch.minimum(distance, delta.square().sum(-1))
            farthest = distance.argmax(-1)
        return points[rows[:, None], indices], indices

    @staticmethod
    def knn_points(p1, p2, K, return_sorted=False):
        distances, indices = torch.cdist(p1, p2).square().topk(
            K, dim=-1, largest=False, sorted=return_sorted
        )
        return SimpleNamespace(idx=indices, dists=distances)


def small_agent(**overrides):
    config = dict(
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        action_dim=7,
        token_dim=24,
        num_patches=4,
        group_size=4,
        patch_depth=1,
        num_scene_tokens=2,
        hidden_dim=32,
        n_head=4,
        timestep_embed_dim=16,
        mlp_ratio=2.0,
        p_drop_attn=0.0,
        wrist_pose_key="split",
        fps_random_config={"use_random": False},
        num_inference_steps=2,
    )
    config.update(overrides)
    return InteractionFlowAgent(**config)


def observation(*, real=False, dense=False):
    batch_size, history = 2, 2
    rotation = torch.tensor([1., 0., 0., 0., 1., 0.]).expand(batch_size, history, 6).clone()
    position = torch.zeros(batch_size, history, 3)
    fingers = torch.randn(batch_size, history, 5, 3) * 0.01
    force = torch.randn(batch_size, history, 5, 3)
    result = {
        "joint_state": torch.randn(batch_size, history, 19),
        "point_cloud": torch.cat(
            (torch.randn(batch_size, history, 16, 3) * 0.025,
             torch.rand(batch_size, history, 16, 3)), dim=-1
        ),
        "fingertip_points": fingers if real else fingers.flatten(-2),
        "contact_force": force if real else force.flatten(-2),
    }
    if real:
        result["eef_pose"] = torch.cat((position, rotation), dim=-1)
    else:
        result.update(eef_pos=position, eef_rot6d=rotation)
    if dense:
        result["tactile_force"] = torch.randn(batch_size, history, 5, 120, 3)
    return result


def install_normalization(agent, obs):
    spec = {key: "identity" for key in obs}
    spec.update(joint_state="limits", contact_force="gaussian", action="limits")
    if "tactile_force" in obs:
        spec["tactile_force"] = "gaussian"
    agent.set_normalization_spec(spec)
    agent.normalizer.fit_field("action", torch.randn(16, agent.action_dim), mode="limits")
    for field, values in obs.items():
        if spec[field] != "identity":
            agent.normalizer.fit_field(field, values, mode=spec[field])
    validate_normalizer_state(agent.normalizer, spec)


class InteractionFlowTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        # Exercise existing production wrappers with a small reference backend;
        # deployment still requires its real PyTorch3D build.
        self.sampling = patch(
            "dexmani_policy.agents.obs_encoder.pointcloud.ops._pointcloud_ops",
            return_value=ReferencePointOps,
        )
        self.sampling.start()
        self.addCleanup(self.sampling.stop)

    def test_sim_flow_training_and_action_window(self):
        agent = small_agent(time_shift_alpha=2.0)
        obs = observation()
        install_normalization(agent, obs)
        validate_observation_fields(agent, tuple(obs))
        self.assertEqual(len(agent.action_decoder.model.ditx_blocks), 8)
        self.assertFalse(agent.requires_ema_for_loss)
        self.assertEqual(agent.action_decoder.time_shift_alpha, 2.0)
        cond, _ = agent._build_cond(obs)
        self.assertEqual(cond.shape, (2, 2 * (2 + 6 + 1), 24))
        # Identity geometry survives the shared preprocessing / time flattening.
        prepared = agent.preprocess(obs)
        for field in agent.metric_observation_fields:
            torch.testing.assert_close(prepared[field], obs[field].flatten(0, 1))

        optimizer = agent.configure_optimizer(lr=1e-3, weight_decay=1e-6)
        batch = {"obs": obs, "action": torch.randn(2, 4, 7)}
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = agent.compute_loss(batch)
            self.assertTrue(torch.isfinite(loss).item())
            self.assertIn("loss_action", metrics)
            loss.backward()
            self.assertTrue(all(
                torch.isfinite(p.grad).all().item()
                for p in agent.parameters() if p.grad is not None
            ))
            optimizer.step()
        gradient = sum(
            p.grad.abs().sum().item() for p in agent.obs_encoder.parameters()
            if p.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        gate_grad = agent.obs_encoder.modality_gate.gate[-1].weight.grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(gate_grad.abs().sum().item(), 0.0)

        agent.eval()
        prediction = agent.predict_action(obs)
        self.assertEqual(prediction["pred_action"].shape, (2, 4, 7))
        self.assertEqual(prediction["control_action"].shape, (2, 2, 7))
        self.assertEqual(prediction["tail"].shape, (2, 1, 7))
        self.assertTrue(torch.isfinite(prediction["pred_action"]).all().item())
        torch.testing.assert_close(prediction["control_action"], prediction["pred_action"][:, 1:3])
        torch.testing.assert_close(prediction["tail"], prediction["pred_action"][:, 3:])

    def test_real_pose_and_dense_tactile_contract(self):
        agent = small_agent(
            wrist_pose_key="eef_pose", tactile_input_key="tactile_force", use_tactile_valid=True
        )
        obs = observation(real=True, dense=True)
        obs["tactile_valid"] = torch.ones(2, 2, 5, dtype=torch.bool)
        obs["tactile_valid"][..., 2] = False
        install_normalization(agent, obs)
        validate_observation_fields(agent, tuple(obs))
        agent.eval()
        with torch.no_grad():
            cond, _ = agent._build_cond(obs)
        self.assertEqual(cond.shape, (2, 18, 24))
        self.assertTrue(torch.isfinite(cond).all().item())
        with self.assertRaisesRegex(ValueError, "boolean mask"):
            agent.set_normalization_spec(dict(agent.normalization_spec, tactile_valid="limits"))

    def test_bfloat16_autocast_keeps_geometry_and_loss_finite(self):
        agent = small_agent()
        obs = observation()
        install_normalization(agent, obs)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            cond, aux = agent.obs_encoder(agent.preprocess(obs), return_intermediate=True)
            loss, _ = agent.compute_loss({"obs": obs, "action": torch.randn(2, 4, 7)})
        self.assertTrue(torch.isfinite(cond).all())
        self.assertEqual(aux["surface_distance"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None))

    def test_rejects_affine_geometry_and_blind_dropout(self):
        agent = small_agent()
        obs = observation()
        install_normalization(agent, obs)
        for field in agent.metric_observation_fields:
            spec = dict(agent.normalization_spec, **{field: "limits"})
            with self.assertRaisesRegex(ValueError, "identity"):
                agent.set_normalization_spec(spec)
        for field in ("point_cloud", "fingertip_points", "eef_pos", "eef_rot6d", "contact_force"):
            with self.assertRaisesRegex(ValueError, "modality_dropout_probs"):
                small_agent(modality_dropout_probs={field: 0.1})
        with self.assertRaisesRegex(ValueError, "8 DiT-X"):
            small_agent(n_layers=4)


if __name__ == "__main__":
    unittest.main()
