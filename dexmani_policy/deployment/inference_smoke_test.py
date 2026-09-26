"""Policy-only deployment NFE/schema regression using a synthetic training snapshot.

Run: python -m dexmani_policy.deployment.inference_smoke_test
The separate deployment.smoke_test covers the Real producer/consumer boundary.
"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.core.inference_smoke_test import small_policy
from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint, build_agent_contract
from dexmani_policy.deployment import export_deployment_artifact, inspect_experiment, load_experiment
from dexmani_policy.deployment.contract import DEPLOYMENT_FORMAT, DeploymentContractError, parse_deployment_contract
from dexmani_policy.deployment.export import _resolve_selected_inference_settings, UnsupportedPolicyError
from dexmani_policy.deployment.restore import DeploymentRestoreError, restore_deployment_agent


def synthetic_experiment(directory):
    """Explicit synthetic checkpoint-owned facts, never historical calibration guesses."""
    cfg, agent, obs = small_policy("maniflow")
    cfg.task_name = "synthetic"
    cfg.eval.inference_steps = 4
    cfg.eval.use_ema = False
    agent.action_key = "action"
    agent.normalization_spec = dict(joint_state="limits", point_cloud="limits", action="limits")
    resume_contract = dict(
        agent=build_agent_contract(agent),
        agent_config=OmegaConf.to_container(cfg.agent, resolve=True),
        dataset={"sensor_modalities": ["joint_state", "point_cloud"]},
        deployment_data_semantics={
            "task_name": "synthetic", "dt": 0.05,
            "joint_names": [f"synthetic_joint_{index}" for index in range(19)],
            "pointcloud_config": {"num_points": 64, "remove_table": False},
            "observation_fields": {
                "joint_state": {"shape": [19], "dtype": "float32"},
                "point_cloud": {"shape": [64, 6], "dtype": "float32"},
            },
        },
    )
    store = CheckpointStore(directory / "checkpoints")
    store.save("train.pt", TrainCheckpoint(
        epoch=0, global_step=0, next_micro_step=0, model_state=agent.state_dict(),
        ema_model_state=None, optimizer_state={}, scheduler_state={}, monitor={},
        resume_contract=resume_contract, ema_updater_step=None, ema_decay=None, rng_states=[{}],
    ))
    OmegaConf.save(cfg, directory / "config.yaml")
    with patch("zarr.open_group", side_effect=AssertionError("export must not reopen data")):
        receipt = export_deployment_artifact(directory, "train.pt")
    return cfg, agent, obs, receipt


class DeploymentInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.cfg, cls.agent, cls.obs, cls.receipt = synthetic_experiment(cls.directory)
        cls.payload = torch.load(cls.receipt.checkpoint_path, weights_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_new_schema_and_strict_restore(self):
        self.assertEqual(DEPLOYMENT_FORMAT, "dexmani.deployment.v3")
        self.assertEqual(self.payload["_format"], DEPLOYMENT_FORMAT)
        self.assertEqual(self.payload["contract"]["inference_config"]["eval"], {"inference_steps": 4})
        self.assertTrue(self.receipt.selector_path.is_symlink())
        info = inspect_experiment(self.directory)
        self.assertEqual(info.default_inference_steps, 4)
        restored = restore_deployment_agent(self.payload)
        for key, tensor in self.agent.state_dict().items():
            torch.testing.assert_close(restored.agent.state_dict()[key], tensor, rtol=0, atol=0)
        torch.manual_seed(17)
        expected = self.agent.predict_action(self.obs, inference_steps=4)
        torch.manual_seed(17)
        actual = restored.agent.predict_action(self.obs, inference_steps=4)
        torch.testing.assert_close(expected["pred_action"], actual["pred_action"], rtol=0, atol=0)
        broken = copy.deepcopy(self.payload)
        broken["weights"].pop("action_decoder.model.input_embedder.weight")
        with self.assertRaises(DeploymentRestoreError):
            restore_deployment_agent(broken)

    def test_runtime_override_and_warmup(self):
        observation = {key: value[0].numpy() for key, value in self.obs.items()}
        for override, expected_calls in ((None, 4), (2, 2)):
            policy = load_experiment(self.directory, device="cpu", inference_steps=override)
            try:
                agent = policy._restored.agent
                with patch.object(agent.action_decoder.model, "forward", wraps=agent.action_decoder.model.forward) as forward:
                    result = policy.predict(observation)
                    self.assertEqual(forward.call_count, expected_calls)
                    forward.reset_mock()
                    rng = torch.get_rng_state()
                    policy.warmup(samples=2)
                    self.assertEqual(forward.call_count, expected_calls * 2)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assertEqual(result.shape, (8, 19))
                self.assertEqual(result.dtype, np.float64)
                self.assertEqual(agent.action_decoder.num_inference_steps, 10)
                self.assertEqual(agent.action_decoder.time_sampler.num_steps, 10)
                self.assertEqual(policy.info.default_inference_steps, 4)
            finally:
                policy.close()
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                load_experiment(self.directory, device="cpu", inference_steps=invalid)

    def test_old_artifact_rejected(self):
        old = copy.deepcopy(self.payload)
        old["_format"] = "dexmani.deployment.v2"
        old["contract"]["inference_config"]["eval"] = {"denoise_steps": 4}
        with self.assertRaisesRegex(DeploymentContractError, "unsupported deployment format"):
            parse_deployment_contract(old)

    def test_legacy_config_ingress_and_sweep_rejection(self):
        for field in ("inference_steps", "denoise_steps"):
            selected = _resolve_selected_inference_settings(
                self.directory, "train.pt", {"eval": {field: 3, "use_ema": False}},
            )
            self.assertEqual(selected.inference_steps, 3)
        for field in ("inference_steps_list", "denoise_timesteps_list"):
            with self.assertRaises(UnsupportedPolicyError):
                _resolve_selected_inference_settings(
                    self.directory, "train.pt", {"eval": {field: [1, 4]}},
                )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            _resolve_selected_inference_settings(
                self.directory, "train.pt",
                {"eval": {"inference_steps": 3, "denoise_steps": 4, "use_ema": False}},
            )


if __name__ == "__main__":
    unittest.main()
