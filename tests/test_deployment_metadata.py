"""训练数据身份与公开部署配方的无设备回归。"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.deployment import PolicyInfo, inspect_policy, load_policy
from dexmani_policy.training.common.config import assemble_effective_config
from dexmani_policy.training.common.data import build_dataset
from dexmani_policy.training.common.logging import ExperimentLogger


class DatasetFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def create_data(self, name="data.zarr", *, source_id="source-a", canonical=True):
        path = self.directory / name
        root = zarr.open_group(str(path), mode="w")
        if canonical:
            root.attrs["format"] = "dexmani.real.canonical"
        if source_id is not None:
            root.attrs["source_id"] = source_id
        root.attrs["episode_ids"] = ["episode_001"]
        data = root.create_group("data")
        for key in ("joint_state", "action"):
            data.create_dataset(key, data=np.zeros((4, 19), dtype=np.float32))
        root.create_group("meta").create_dataset("episode_ends", data=np.array([4]))
        return path

    def config(self, path, *, resume=False, manifest=None):
        return OmegaConf.create({
            "resume_from": "checkpoint.pt" if resume else None,
            "normalization": {"joint_state": "limits", "action": "auto"},
            "agent": {"action_dim": 19},
            "dataset": {
                "_target_": "dexmani_policy.datasets.common.robot_dataset.RobotDataset",
                "zarr_path": str(path), "sensor_modalities": ["joint_state"],
                "horizon": 1, "obs_horizon": 1, "split_manifest": manifest,
            },
        })

class CanonicalResumeTest(DatasetFixture, unittest.TestCase):
    def test_same_identity_moves_but_reexport_does_not_resume(self):
        path = self.create_data()
        _, description = build_dataset(self.config(path), verbose=False)
        manifest = description["split_manifest"]
        moved = self.directory / "moved.zarr"
        shutil.copytree(path, moved)
        build_dataset(self.config(moved, resume=True, manifest=manifest), verbose=False)
        replacement = self.create_data("replacement.zarr", source_id="source-b")
        with self.assertRaisesRegex(InstantiationException, "source identity mismatch"):
            build_dataset(self.config(replacement, resume=True, manifest=manifest), verbose=False)

    def test_missing_identity_rejects_resume_even_at_the_same_path(self):
        path = self.create_data(source_id=None)
        _, description = build_dataset(self.config(path), verbose=False)
        manifest = description["split_manifest"]
        # A supplied legacy split remains valid for fresh training and evaluation.
        build_dataset(self.config(path, manifest=manifest), verbose=False)
        with self.assertRaisesRegex(ValueError, "Canonical checkpoint resume requires source_id"):
            build_dataset(self.config(path, resume=True, manifest=manifest), verbose=False)
        for source_id, saved_id in (("source-a", None), (None, "source-a")):
            root = zarr.open_group(str(path), mode="a")
            if source_id is None:
                root.attrs.pop("source_id", None)
            else:
                root.attrs["source_id"] = source_id
            manifest["source_id"] = saved_id
            with self.assertRaisesRegex(InstantiationException, "source identity mismatch"):
                build_dataset(self.config(path, resume=True, manifest=manifest), verbose=False)

    def test_other_formats_keep_their_existing_identity_contract(self):
        path = self.create_data(source_id=None, canonical=False)
        _, description = build_dataset(self.config(path), verbose=False)
        build_dataset(
            self.config(path, resume=True, manifest=description["split_manifest"]), verbose=False,
        )


class CpuPolicy(BasePolicy):
    """保留真实公开元信息和 runtime，仅用 CPU 动作代替模型推理。"""

    def __init__(self, *, action_dim, n_obs_steps, hand_history=False):
        super().__init__(torch.nn.Identity(), torch.nn.Identity(), n_obs_steps, n_obs_steps, 1, action_dim)
        self.hand_history = hand_history
        self.seen_context = None

    @property
    def action_context_shapes(self):
        return {"hand_history": [8, 12]} if self.hand_history else {}

    def predict_action(self, observation, *, generator, **kwargs):
        self.seen_context = kwargs.get("action_context")
        if self.hand_history and self.seen_context is None:
            raise ValueError("hand_history is required")
        return {"future_actions": torch.randn(1, self.future_steps, self.action_dim, generator=generator)}


class DeploymentMetadataTest(DatasetFixture, unittest.TestCase):
    def artifact(self, path, modalities, *, action_key="action", hand_history=False):
        cfg = self.config(path)
        cfg.dataset.sensor_modalities = modalities
        cfg.dataset.action_key = action_key
        cfg.dataset.obs_horizon = cfg.dataset.horizon = 1 if hand_history else 2
        cfg.normalization = {name: "identity" for name in modalities}
        cfg.normalization.action = "auto"
        cfg.agent.action_dim = 21 if action_key == "action_ee" else 19
        cfg.agent.n_obs_steps = cfg.agent.horizon = cfg.dataset.obs_horizon
        cfg.agent.n_action_steps = 1
        # Encoder history alone must not become physical hand action history.
        cfg.agent.history_steps = 99
        cfg.policy_name, cfg.task_name, cfg.action_key = "cpu_test", "test_task", action_key
        cfg.tactile_source = "real" if "finger_tactile_force" in modalities else None
        cfg.dataset.tactile_source = cfg.tactile_source
        cfg.training = {"lr_scheduler": "constant"}
        cfg.eval = {"use_ema": False, "inference_steps": 1}
        _, description = build_dataset(cfg, verbose=False)
        model = CpuPolicy(action_dim=cfg.agent.action_dim, n_obs_steps=cfg.agent.n_obs_steps,
                          hand_history=hand_history)
        effective = assemble_effective_config(cfg, description, model)
        directory = self.directory / ("experiment_" + action_key + str(hand_history))
        logger = ExperimentLogger(str(directory))
        try:
            logger.save_config(effective)
        finally:
            logger.close()
        payload = OmegaConf.to_container(effective, resolve=True)
        checkpoint_path = save_checkpoint(directory / "checkpoints/latest.pt", Checkpoint(
            epoch=0, global_step=0, model_state={}, ema_model_state=None,
            optimizer_state={}, scheduler_state={}, ema_updater_step=None, rng_state={},
            effective_config=payload,
        ))
        self.assertEqual(load_checkpoint(checkpoint_path).effective_config["deployment_metadata"],
                         payload["deployment_metadata"])
        info = inspect_policy(directory, checkpoint="latest")
        self.assertIsInstance(info, PolicyInfo)
        return info, model, payload

    def test_real_recipe_survives_training_save_and_public_inspection(self):
        path = self.create_data()
        root = zarr.open_group(str(path), mode="a")
        attrs = dict(
            dt=0.05, depth_scale_m_per_unit=0.001,
            pointcloud_config={"remove_table": True, "num_points": 4,
                               "table_margin_m": 0.01, "workspace": [0, -0.5, 0, 0.8, 0.5, 0.8]},
            hand_pointcloud_config={"num_points": 3},
            fingertip_link_names=[f"tip_{index}" for index in range(5)],
            processing_source={"repository": "dexmani_real", "hand_template": "assets/hand_3.npz"},
        )
        root.attrs.update(attrs)
        shapes = {"depth": (4, 2, 3), "point_cloud": (4, 4, 6), "hand_pointcloud": (4, 3, 6),
                  "fingertip_xyz": (4, 5, 3), "finger_tactile_force": (4, 5, 3)}
        for key, shape in shapes.items():
            root["data"].create_dataset(key, data=np.ones(shape, dtype=np.float32))
        root["data"].create_dataset("action_ee", data=np.zeros((4, 21), dtype=np.float32))
        modalities = ["joint_state", *shapes]
        for action_key in ("action", "action_ee"):
            with self.subTest(action_key=action_key):
                info, _, saved = self.artifact(path, modalities, action_key=action_key)
                self.assertEqual(info.control_dt_s, 0.05)
                self.assertEqual(info.depth_config, {"shape": [2, 3], "depth_scale_m": 0.001})
                self.assertEqual(info.pointcloud_config, attrs["pointcloud_config"])
                self.assertEqual(info.hand_pointcloud_config, attrs["hand_pointcloud_config"])
                self.assertEqual(info.fingertip_link_names, tuple(attrs["fingertip_link_names"]))
                self.assertEqual(info.tactile_source, "real")
                self.assertIsNone(info.hand_history_steps)
                self.assertEqual(saved["deployment_metadata"]["processing_source"], attrs["processing_source"])
                self.assertEqual(info.action_mode, "eef" if action_key == "action_ee" else "joint")
                observation = {name: np.asarray(root["data"][name][:info.n_obs_steps]) for name in modalities}
                with patch("dexmani_policy.deployment.runtime.restore_policy", return_value=CpuPolicy(
                    action_dim=info.control_action_dim, n_obs_steps=info.n_obs_steps,
                )):
                    runtime = load_policy(info, device="cpu")
                runtime.warmup(observation, samples=1)
                self.assertEqual(runtime.predict(observation).shape, (info.future_steps, info.control_action_dim))
                runtime.close()

    def test_public_load_warmup_and_predict_keep_context_and_rng(self):
        path = self.create_data()
        info, model, _ = self.artifact(path, ["joint_state"], hand_history=True)
        self.assertEqual(info.hand_history_steps, 8)
        self.assertEqual(info.to_dict()["hand_history_steps"], 8)
        observation = {"joint_state": np.zeros((1, 19), dtype=np.float32)}
        context = {"hand_history": np.full((8, 12), 0.2, dtype=np.float32)}
        with patch("dexmani_policy.deployment.runtime.restore_policy", return_value=model):
            runtime = load_policy(info, device="cpu", seed=42)
        state = runtime.generator.get_state().clone()
        durations = runtime.warmup(observation, samples=1, action_context=context)
        self.assertTrue(torch.equal(state, runtime.generator.get_state()))
        self.assertEqual(len(durations["bootstrap"]), 1)
        self.assertEqual(tuple(model.seen_context["hand_history"].shape), (1, 8, 12))
        action = runtime.predict(observation, action_context=context)
        self.assertEqual(action.shape, (1, 19))
        runtime.reset_episode()
        np.testing.assert_array_equal(action, runtime.predict(observation, action_context=context))
        runtime.close()

    def test_inspection_without_canonical_recipe_stays_available(self):
        path = self.create_data(canonical=False)
        info, _, saved = self.artifact(path, ["joint_state"])
        self.assertEqual(saved["deployment_metadata"], {})
        for field in ("control_dt_s", "depth_config", "pointcloud_config", "hand_pointcloud_config",
                      "fingertip_link_names", "tactile_source", "hand_history_steps"):
            self.assertIsNone(getattr(info, field))
        # Existing non-Real artifacts without the new optional section still inspect.
        saved.pop("deployment_metadata")
        OmegaConf.save(OmegaConf.create(saved), info.experiment_dir / "config.yaml")
        self.assertIsNone(inspect_policy(info.experiment_dir, checkpoint="latest").control_dt_s)


if __name__ == "__main__":
    unittest.main()
