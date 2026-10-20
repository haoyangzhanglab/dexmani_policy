"""CPU regression checks for inspected configuration and physical inference outputs."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from dexmani_policy.agents.core.base_policy import BasePolicy
from dexmani_policy.agents.core.normalizer import FieldNormalizer
from dexmani_policy.deployment import PolicyInfo, PolicyRuntime, inspect_policy, load_policy


class ToyPolicy(BasePolicy):
    def __init__(
        self, horizon, n_obs_steps, n_action_steps, action_dim, observation_fields, use_aux_ee=False
    ):
        encoder = torch.nn.Identity()
        encoder.consumed_observation_fields = tuple(observation_fields)
        super().__init__(
            encoder, torch.nn.Identity(), horizon, n_obs_steps, n_action_steps, action_dim
        )
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.use_aux_ee = use_aux_ee
        self.ee_dim = 9
        self.observations = None
        self.arguments = None

    @property
    def control_action_dim(self):
        return self.action_dim - (9 if self.use_aux_ee else 0)

    def predict_action(self, obs_dict, inference_steps=None, **kwargs):
        self.observations = obs_dict
        self.arguments = kwargs
        values = torch.arange(self.horizon * self.action_dim, device=self.offset.device).float()
        return self.format_action_output(
            values.reshape(1, self.horizon, self.action_dim) + self.offset
        )


class ToyCondition(torch.nn.Module):
    consumed_observation_fields = ("joint_state",)

    def forward(self, obs):
        return obs["joint_state"].reshape(1, -1)


class ToyDenoiser(torch.nn.Module):
    def forward(self, x, timestep, context):
        return x * 0.1 + context.mean() * 0.01


def make_experiment(root, *, action_key="action", auxiliary=False, fields=None):
    fields = fields or ["joint_state"]
    dimension = (19 if action_key == "action" else 21) + (9 if auxiliary else 0)
    config = {
        "policy_name": "toy",
        "task_name": "task",
        "action_key": action_key,
        "agent": {
            "_target_": f"{__name__}.ToyPolicy",
            "horizon": 4,
            "n_obs_steps": 2,
            "n_action_steps": 2,
            "action_dim": dimension,
            "observation_fields": fields,
            "use_aux_ee": auxiliary,
        },
        "dataset": {"sensor_modalities": fields, "pc_dim": 3},
        "normalization": {**dict.fromkeys(fields, "identity"), "action": "limits"},
        "eval": {"use_ema": False, "inference_steps": 3},
        "real_runtime": {"dt": 0.1, "pointcloud": {"num_points": 8}},
    }
    args = dict(config["agent"])
    args.pop("_target_")
    agent = ToyPolicy(**args)
    agent.normalizer["action"] = FieldNormalizer.from_parameters(
        torch.ones(dimension), torch.zeros(dimension)
    )
    raw = agent.state_dict()
    ema = copy.deepcopy(raw)
    ema["offset"].fill_(2)
    checkpoint = root / "checkpoints" / "epoch=0001-step=2-milestone=20pct.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "_format": "simple.v4",
            "effective_config": config,
            "state": {"global_step": 2},
            "weights": {"model": raw, "ema_model": ema},
        },
        checkpoint,
    )
    OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
    selection = root / "eval_ckpt_selector/toy/selection_result.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "status": "success",
                "selection_id": "toy",
                "ckpt_relpath": str(checkpoint.relative_to(root)),
                "global_step": 2,
                "inference": {
                    "use_ema": True,
                    "inference_steps": 5,
                    "policy_seed_mode": "episode_seed",
                },
            }
        )
    )
    (root / "best_ckpt.json").write_text(
        json.dumps(
            {
                "selection_result": str(selection.relative_to(root)),
            }
        )
    )
    return config, checkpoint


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_inspection_is_light_and_config_is_a_snapshot(self):
        config, _ = make_experiment(self.root)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from dexmani_policy.deployment import inspect_policy; "
                "inspect_policy(sys.argv[1]); assert 'torch' not in sys.modules",
                str(self.root),
            ],
            check=True,
        )
        info = inspect_policy(self.root)
        self.assertEqual((info.weights, info.inference_steps), ("ema", 5))
        summary = info.to_dict()
        self.assertNotIn("saved_config", summary)
        self.assertEqual(summary["checkpoint_path"], str(info.checkpoint_path))
        self.assertEqual(summary["observation_fields"], ["joint_state"])
        json.dumps(summary)
        with self.assertRaises(ReadonlyConfigError):
            info.saved_config.agent.horizon = 99
        recipe = info.pointcloud_config
        recipe["num_points"] = 999
        self.assertEqual(info.pointcloud_config["num_points"], 8)
        config["real_runtime"]["dt"] = 9
        OmegaConf.save(OmegaConf.create(config), self.root / "config.yaml")
        policy = load_policy(info, device="cpu")
        self.addCleanup(policy.close)
        self.assertEqual(policy.info.control_dt_s, 0.1)

    def test_raw_ema_future_and_auxiliary_dimensions(self):
        for action_key, auxiliary in (("action", False), ("action_ee", False), ("action", True)):
            for weights, offset in (("raw", 0), ("ema", 2)):
                with self.subTest(action_key=action_key, auxiliary=auxiliary, weights=weights):
                    root = self.root / f"{action_key}-{auxiliary}-{weights}"
                    config, _ = make_experiment(root, action_key=action_key, auxiliary=auxiliary)
                    info = inspect_policy(root, weights=weights, inference_steps=7)
                    policy = load_policy(info, device="cpu", seed=7)
                    self.addCleanup(policy.close)
                    obs = np.arange(5 * 19, dtype=np.float32).reshape(5, 19)
                    prediction = policy.predict({"joint_state": obs})
                    full = np.arange(4 * info.action_dim).reshape(4, info.action_dim) + offset
                    np.testing.assert_equal(prediction, full[1:, : info.control_action_dim])
                    np.testing.assert_equal(
                        policy.agent.observations["joint_state"], obs[-2:][None]
                    )
                    self.assertEqual(prediction.dtype, np.float64)
                    self.assertEqual(info.inference_steps, 7)
                    policy.close()
                    policy.close()
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        policy.predict({"joint_state": obs})

    def test_boundary_errors_and_rgb_recipe(self):
        config, _ = make_experiment(self.root, fields=["joint_state", "rgb", "point_cloud"])
        config["dataset"].update(
            rgb_preprocess_size=[6, 8], rgb_random_crop_size=[4, 6], rgb_keep_uint8=True
        )
        OmegaConf.save(OmegaConf.create(config), self.root / "config.yaml")
        info = inspect_policy(self.root)
        agent_args = dict(config["agent"])
        agent_args.pop("_target_")
        policy = PolicyRuntime(ToyPolicy(**agent_args), info, device="cpu", seed=0)
        observation = {
            "joint_state": np.zeros((3, 19), np.float32),
            "rgb": np.arange(3 * 10 * 12 * 3, dtype=np.uint8).reshape(3, 10, 12, 3),
            "point_cloud": np.ones((3, 8, 6), np.float32),
        }
        policy.predict(observation)
        self.assertEqual(tuple(policy.agent.observations["point_cloud"].shape), (1, 2, 8, 3))
        from dexmani_policy.datasets.common.rgb_preprocessing import preprocess_validation_rgb

        expected = preprocess_validation_rgb(observation["rgb"][-2:], **info.rgb_preprocessing)
        torch.testing.assert_close(policy.agent.observations["rgb"], expected.unsqueeze(0))
        invalid = [
            {"joint_state": observation["joint_state"]},
            {**observation, "joint_state": np.zeros((2, 19), np.float32)},
            {**observation, "joint_state": np.full((3, 19), np.nan)},
            {**observation, "point_cloud": np.ones((3, 8, 2))},
        ]
        for obs in invalid:
            with self.assertRaises(ValueError):
                policy.predict(obs)
        for prefix in (np.zeros((4, 19)), np.zeros((2, 21)), np.full((2, 19), np.nan)):
            with self.assertRaises(ValueError):
                policy.predict(observation, rtc_prefix=prefix)
        with self.assertRaises(ValueError):
            policy.predict(observation, delay_steps=1)
        with self.assertRaises(ValueError):
            policy.predict(observation, action_context={"hand_history": np.zeros((2, 12))})
        with patch.object(
            policy.agent,
            "predict_action",
            return_value={"pred_action": torch.full((1, 4, 19), float("nan"))},
        ):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                policy.predict(observation)
        with patch.object(
            policy.agent, "predict_action", return_value={"pred_action": torch.zeros(1, 3, 19)}
        ):
            with self.assertRaisesRegex(ValueError, "pred_action"):
                policy.predict(observation)

    def test_checkpoint_recipe_and_missing_ema(self):
        _, checkpoint = make_experiment(self.root)
        info = inspect_policy(self.root)
        payload = torch.load(checkpoint, weights_only=False)
        payload["effective_config"]["agent"]["n_obs_steps"] = 1
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            load_policy(info, device="cpu")
        payload["effective_config"]["agent"]["n_obs_steps"] = 2
        payload["weights"]["ema_model"] = None
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "EMA"):
            load_policy(info, device="cpu")

    def test_inference_ignores_training_settings_but_checks_inputs(self):
        config, _ = make_experiment(self.root, fields=["joint_state", "rgb"])
        config.update(
            optimizer={"lr": 0.5},
            dataloader={"batch_size": 1},
            training={"device": "cpu", "seed": 999},
            eval={"episodes": 99},
        )
        config["dataset"]["zarr_path"] = "/moved/data.zarr"
        OmegaConf.save(OmegaConf.create(config), self.root / "config.yaml")
        policy = load_policy(inspect_policy(self.root), device="cpu")
        policy.close()
        for key, value in (("rgb_preprocess_size", [32, 32]), ("pc_dim", 6)):
            changed = copy.deepcopy(config)
            changed["dataset"][key] = value
            OmegaConf.save(OmegaConf.create(changed), self.root / "config.yaml")
            with self.assertRaisesRegex(ValueError, "inference recipe mismatch"):
                load_policy(inspect_policy(self.root), device="cpu")
        config["real_runtime"]["dt"] = 0.2
        OmegaConf.save(OmegaConf.create(config), self.root / "config.yaml")
        with self.assertRaisesRegex(ValueError, "inference recipe mismatch"):
            load_policy(inspect_policy(self.root), device="cpu")

    def test_warmup_rtc_and_cleanup(self):
        make_experiment(self.root)
        policy = load_policy(inspect_policy(self.root), device="cpu")
        with self.assertRaises(NotImplementedError):
            policy.configure_execution("rtc", 1.0)
        policy.configure_execution("sync")
        durations = policy.warmup(samples=2)
        self.assertEqual(len(durations["bootstrap"]), 2)
        self.assertEqual(durations["bootstrap"], durations["steady"])
        with patch.object(policy, "reset_episode", wraps=policy.reset_episode) as reset:
            with patch.object(policy, "predict", side_effect=ValueError("failed inference")):
                with self.assertRaises(ValueError):
                    policy.warmup(samples=1)
            self.assertEqual(reset.call_count, 2)
        agent = policy.agent
        with patch(
            "dexmani_policy.training.common.checkpoint.restore_policy_agent", return_value=agent
        ):
            with patch(
                "dexmani_policy.deployment.runtime.PolicyRuntime",
                side_effect=ValueError("invalid layout"),
            ):
                with patch.object(agent, "to", wraps=agent.to) as move:
                    with self.assertRaisesRegex(ValueError, "invalid layout"):
                        load_policy(policy.info, device="cpu")
                    move.assert_called_once_with("cpu")
        with patch.object(agent, "close", create=True, side_effect=RuntimeError("close failed")):
            with patch.object(agent, "to", wraps=agent.to) as move:
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    policy.close()
                move.assert_called_once_with("cpu")
        self.assertIsNone(policy.agent)
        policy.close()

    def test_continuous_rtc_inference_and_warmup(self):
        from dexmani_policy.agents.action_decoders.diffusion import Diffusion

        make_experiment(self.root)
        info = inspect_policy(self.root, inference_steps=3)
        agent = BasePolicy(
            ToyCondition(), Diffusion(ToyDenoiser(), num_training_steps=20), 4, 2, 2, 19
        )
        agent.set_normalization_spec(info.normalization)
        agent.normalizer["action"] = FieldNormalizer.from_parameters(
            torch.ones(19), torch.zeros(19)
        )
        policy = PolicyRuntime(agent, info, device="cpu", seed=3)
        observation = {"joint_state": np.ones((2, 19), np.float32)}
        policy.configure_execution("rtc", 1.0)
        prediction = policy.predict(observation, rtc_prefix=np.zeros((2, 19)), delay_steps=1)
        self.assertEqual(prediction.shape, (3, 19))
        self.assertTrue(np.isfinite(prediction).all())
        durations = policy.warmup(samples=1, rtc_delay=1)
        self.assertEqual(len(durations["bootstrap"]), 1)
        self.assertEqual(len(durations["steady"]), 1)
        policy.close()

    def test_lamp_history_and_synchronous_policy_boundaries(self):
        from dexmani_policy.agents.core.lamp_policy import LAMPPolicy
        from dexmani_policy.agents.core.temporal_latent_flow_policy import TemporalLatentFlowPolicy

        config, checkpoint = make_experiment(self.root)
        config["agent"].update(
            _target_="dexmani_policy.agents.core.lamp_policy.LAMPPolicy",
            horizon=1,
            n_obs_steps=1,
            n_action_steps=1,
            history_steps=2,
        )
        config["dataset"]["history_steps"] = 2
        info = PolicyInfo(self.root, checkpoint, config, "raw", 1)
        agent = LAMPPolicy.__new__(LAMPPolicy)
        BasePolicy.__init__(agent, ToyCondition(), torch.nn.Identity(), 1, 1, 1, 19)
        agent.history_steps, agent.hand_dim = 2, 12
        policy = PolicyRuntime(agent, info, device="cpu", seed=0)
        observation = {"joint_state": np.zeros((1, 19), np.float32)}
        history = np.arange(24, dtype=np.float64).reshape(2, 12)
        for context in (
            None,
            {"hand_history": np.zeros((1, 12))},
            {"hand_history": np.full((2, 12), np.nan)},
        ):
            with self.assertRaises(ValueError):
                policy.predict(observation, action_context=context)
        with patch.object(agent, "encode_condition", return_value=(None, None)) as encode:
            with patch.object(
                agent,
                "predict_action_from_cond",
                return_value={"pred_action": torch.zeros(1, 1, 19)},
            ):
                policy.predict(observation, action_context={"hand_history": history})
            converted = encode.call_args.args[1]["hand_history"]
            torch.testing.assert_close(converted, torch.from_numpy(history).float().unsqueeze(0))
        with self.assertRaises(NotImplementedError):
            policy.predict(
                observation, action_context={"hand_history": history}, rtc_prefix=np.zeros((1, 19))
            )
        for mode in ("async", "rtc"):
            with self.assertRaises(NotImplementedError):
                policy.configure_execution(mode, 1.0)
        policy.close()

        config["agent"][
            "_target_"
        ] = "dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy"
        info = PolicyInfo(self.root, checkpoint, config, "raw", 1)
        agent = TemporalLatentFlowPolicy.__new__(TemporalLatentFlowPolicy)
        BasePolicy.__init__(agent, ToyCondition(), torch.nn.Identity(), 1, 1, 1, 19)
        policy = PolicyRuntime(agent, info, device="cpu", seed=0)
        for mode in ("async", "rtc"):
            with self.assertRaises(NotImplementedError):
                policy.configure_execution(mode, 1.0)
        with self.assertRaises(NotImplementedError):
            policy.predict(observation, rtc_prefix=np.zeros((1, 19)))
        policy.close()


if __name__ == "__main__":
    unittest.main()
