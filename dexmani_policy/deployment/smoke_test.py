"""Offline cross-repository deployment regressions; requires sibling Real installed.

Run with python -m dexmani_policy.deployment.smoke_test.
"""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.deployment import (
    export_deployment_artifact,
    inspect_experiment,
    load_experiment,
)
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    restore_deployment_agent,
    validate_prediction,
)
from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
)
from dexmani_policy.training.resume import build_resume_contract, build_train_loader


def tiny_config(name, zarr_path, directory):
    cfg = load_config(name)
    cfg.task_name = "smoke"
    cfg.dataset.zarr_path = str(zarr_path)
    cfg.training.device = "cpu"
    cfg.training.use_compile = False
    cfg.workspace.output_dir = str(directory)
    cfg.dataloader.num_workers = 0
    cfg.dataloader.persistent_workers = False
    cfg.dataloader.batch_size = 2
    cfg.dataloader.drop_last = False
    cfg.agent.down_dims = [16, 32]
    cfg.agent.diffusion_step_embed_dim = 16
    cfg.agent.n_groups = 4
    cfg.agent.num_training_steps = 4
    cfg.agent.num_inference_steps = 1
    cfg.eval.denoise_steps = 1
    cfg.eval.use_ema = False
    if name == "dp":
        cfg.agent.rgb_backbone_name = "resnet"
        cfg.agent.rgb_backbone_config = {
            "model_name": "resnet18",
            "weights": None,
            "tune_mode": "full",
            "norm_mode": "group_norm",
            "image_size": [32, 32],
        }
        cfg.dataset.rgb_preprocess_size = [40, 40]
        cfg.dataset.rgb_random_crop_size = [32, 32]
    return cfg


def checkpoint_fixture(directory, name="dp3", *, action_mode="joint"):
    from dexmani_real.deployment.smoke_test import export_fixture

    zarr_path = export_fixture(directory, frames=16)
    cfg = tiny_config(name, zarr_path, directory)
    if action_mode == "eef":
        cfg.action_key = "action_ee"
        cfg.action_dim = 21
        cfg.dataset.action_key = "action_ee"
        cfg.agent.action_dim = 21
        cfg.env_runner.env_kwargs.control_mode = "ee"
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset)
    model, _, _ = build_model_and_ema(cfg, torch.device("cpu"), normalizer)
    contract = build_resume_contract(cfg, model, loader)
    checkpoint = TrainCheckpoint(
        epoch=0,
        global_step=0,
        next_micro_step=0,
        model_state=model.state_dict(),
        ema_model_state=None,
        optimizer_state={},
        scheduler_state={},
        monitor={},
        resume_contract=contract,
        ema_updater_step=None,
        ema_decay=None,
        rng_states=[{}],
    )
    experiment = Path(directory) / "experiment"
    store = CheckpointStore(experiment / "checkpoints")
    store.save("train.pt", checkpoint)
    OmegaConf.save(cfg, experiment / "config.yaml")
    # Export must neither reopen the now-unavailable dataset nor construct a model.
    zarr_path.rename(zarr_path.with_name("relocated.zarr"))
    with (
        patch("zarr.open_group", side_effect=AssertionError("export opened Zarr")),
        patch(
            "hydra.utils.instantiate",
            side_effect=AssertionError("export constructed model"),
        ),
    ):
        receipt = export_deployment_artifact(experiment, "train.pt")
        info = inspect_experiment(experiment)
    return cfg, receipt, info, contract


class PolicyDeploymentSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.cfg, cls.receipt, cls.info, cls.contract = checkpoint_fixture(cls.temp.name)
        cls.policy = load_experiment(cls.info.experiment_dir, device="cpu")
        cls.payload = torch.load(cls.receipt.checkpoint_path, weights_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.policy.close()
        cls.temp.cleanup()

    def test_order_export_and_end_to_end(self):
        from types import SimpleNamespace

        import zarr
        from dexmani_real.deployment.action import decode_policy_action
        from dexmani_real.deployment.config import validate_policy_runtime_compatibility
        from dexmani_real.deployment.observation import build_policy_observation
        from dexmani_real.deployment.smoke_test import RawFixture
        from dexmani_real.robot.model import ROBOT_JOINT_NAMES

        spec = self.policy.spec
        self.assertEqual(
            tuple(self.contract["deployment_data_semantics"]["joint_names"]),
            ROBOT_JOINT_NAMES,
        )
        self.assertEqual(spec.joint_names, ROBOT_JOINT_NAMES)
        self.assertEqual(spec, self.info.spec)
        self.assertFalse(hasattr(spec, "action_dim"))
        raw = RawFixture()
        validate_policy_runtime_compatibility(spec, raw.runtime)
        with self.assertRaisesRegex(ValueError, "joint_names"):
            validate_policy_runtime_compatibility(
                replace(spec, joint_names=tuple(reversed(spec.joint_names))),
                raw.runtime,
            )
        data = zarr.open_group(str(Path(self.temp.name) / "relocated.zarr"), mode="r")[
            "data"
        ]
        rows = [
            SimpleNamespace(
                arm={"qpos": raw.h5f["arm_qpos"][i : i + 1]},
                hand={"qpos": raw.h5f["hand_qpos"][i : i + 1]},
                point_cloud=data["point_cloud"][i],
            )
            for i in range(spec.n_obs_steps)
        ]
        obs = build_policy_observation(rows, spec)
        self.assertEqual(set(obs), {"joint_state", "point_cloud"})
        chunk = self.policy.predict(obs)
        self.assertEqual(chunk.shape, (spec.n_action_steps, 19))
        self.assertEqual(chunk.dtype, np.float64)
        self.assertTrue(np.isfinite(chunk).all())
        arm, hand, _, _ = decode_policy_action(
            chunk[0],
            spec.action_mode,
            raw.runtime.arm.home_qpos,
            previous_arm_command_qpos=None,
            planner=None,
            workspace=raw.runtime.policy.workspace.as_array(),
            hand_qpos_min_rad=raw.runtime.hand.qpos_min_rad,
            hand_qpos_max_rad=raw.runtime.hand.qpos_max_rad,
        )
        np.testing.assert_array_equal(arm, chunk[0, :7])
        np.testing.assert_allclose(
            hand,
            np.clip(
                chunk[0, 7:],
                raw.runtime.hand.qpos_min_rad,
                raw.runtime.hand.qpos_max_rad,
            ),
        )
        self.assertEqual(len(self.policy.warmup(samples=1)), 1)

    def test_normalizer_and_strict_restore(self):
        restored = restore_deployment_agent(self.payload)
        normalizer = restored.agent.normalizer
        for field, dims in (("action", 19), ("joint_state", 19), ("point_cloud", 6)):
            x = torch.linspace(-0.5, 0.5, dims).unsqueeze(0)
            torch.testing.assert_close(
                normalizer[field].unnormalize(normalizer[field].normalize(x)),
                x,
                atol=1e-5,
                rtol=1e-5,
            )
        payload = copy.deepcopy(self.payload)
        del payload["weights"][
            next(k for k in payload["weights"] if "normalizer" not in k)
        ]
        with self.assertRaises(DeploymentRestoreError):
            restore_deployment_agent(payload)
        for bad in (torch.ones(18), torch.full((19,), float("nan")), torch.zeros(19)):
            payload = copy.deepcopy(self.payload)
            key = next(
                k
                for k in payload["weights"]
                if k.endswith("normalizer.params_dict.action.scale")
            )
            payload["weights"][key] = bad
            with self.assertRaises(DeploymentRestoreError):
                restore_deployment_agent(payload)

    def test_auxiliary_output_is_private(self):
        restored = restore_deployment_agent(self.payload)
        spec = replace(restored.spec, action_dim=28)
        pred = torch.arange(spec.horizon * 28, dtype=torch.float32).reshape(
            1, spec.horizon, 28
        )
        start = spec.n_obs_steps - 1
        physical = pred[:, start : start + spec.n_action_steps, :19]
        result = validate_prediction(
            {"pred_action": pred, "control_action": physical}, spec, batch_size=1
        )
        torch.testing.assert_close(result, physical)
        with self.assertRaises(DeploymentRestoreError):
            validate_prediction(
                {
                    "pred_action": pred,
                    "control_action": pred[:, start : start + spec.n_action_steps],
                },
                spec,
                batch_size=1,
            )

    def test_prediction_dtype(self):
        spec = restore_deployment_agent(self.payload).spec
        for dtype in (torch.int64, torch.complex64):
            pred = torch.zeros((1, spec.horizon, spec.action_dim), dtype=dtype)
            start = spec.n_obs_steps - 1
            with self.assertRaisesRegex(DeploymentRestoreError, "floating-point"):
                validate_prediction(
                    {
                        "pred_action": pred,
                        "control_action": pred[:, start : start + spec.n_action_steps],
                    },
                    spec,
                    batch_size=1,
                )

    def test_training_resume_remains_strict(self):
        from dexmani_policy.common.checkpoint_io import validate_resume_contract

        changed = copy.deepcopy(self.contract)
        changed["deployment_data_semantics"]["pointcloud_config"]["voxel_size_m"] *= 2
        with self.assertRaises(ValueError):
            validate_resume_contract(self.contract, changed)

    def test_eef_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, info, _ = checkpoint_fixture(directory, action_mode="eef")
            policy = load_experiment(info.experiment_dir, device="cpu")
            try:
                self.assertEqual(info.spec.action_mode, "eef")
                obs = {
                    "joint_state": np.zeros((info.spec.n_obs_steps, 19), np.float32),
                    "point_cloud": np.zeros(
                        (info.spec.n_obs_steps, 1024, 6), np.float32
                    ),
                }
                chunk = policy.predict(obs)
                self.assertEqual(chunk.shape, (info.spec.n_action_steps, 21))
                self.assertTrue(np.isfinite(chunk).all())
            finally:
                policy.close()

    def test_artifact_cloud_config_and_provenance(self):
        import json
        from types import SimpleNamespace

        from dexmani_real.config.experiment import resolve_experiment_config
        from dexmani_real.config.pointcloud import PointCloudConfig
        from dexmani_real.deployment.config import RolloutRecordingConfig
        from dexmani_real.deployment.session import _rollout_recorder_config
        from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig

        runtime = resolve_experiment_config()
        runtime = replace(
            runtime, pointcloud=replace(runtime.pointcloud, voxel_size_m=0.02)
        )
        cfg = PointCloudLoopConfig.from_runtime(
            runtime,
            pointcloud=PointCloudConfig.from_dict(self.info.spec.pointcloud_config),
        )
        self.assertEqual(cfg.pointcloud.voxel_size_m, 0.005)
        self.assertNotEqual(
            cfg.pointcloud.voxel_size_m, runtime.pointcloud.voxel_size_m
        )
        worker = SimpleNamespace(
            spec=self.info.spec,
            experiment="smoke",
            artifact="artifact.pt",
            inference_steps=1,
            seed=0,
        )
        recording = _rollout_recorder_config(
            runtime,
            RolloutRecordingConfig(self.temp.name, "smoke", "test"),
            worker,
            1.0,
            1,
            camera_calibration=cfg.camera_calibration,
            pointcloud_config=cfg,
        )
        self.assertIs(recording.camera_calibration, cfg.camera_calibration)
        self.assertEqual(
            json.loads(recording.provenance["pointcloud_table_plane_abcd_json"]),
            list(cfg.table_plane_abcd),
        )
        payload = copy.deepcopy(self.payload)
        payload["extra_provenance"] = {"note": "harmless"}
        restored = restore_deployment_agent(payload)
        self.assertEqual(restored.spec.policy_spec, self.info.spec)

    def test_rgb_variable_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, info, _ = checkpoint_fixture(directory, "dp")
            policy = load_experiment(info.experiment_dir, device="cpu")
            try:
                for h, w in ((24, 32), (48, 64)):
                    obs = {
                        "joint_state": np.zeros(
                            (info.spec.n_obs_steps, 19), np.float32
                        ),
                        "rgb": np.zeros((info.spec.n_obs_steps, h, w, 3), np.uint8),
                    }
                    tensors = policy._observation_tensors(obs)
                    self.assertEqual(
                        tuple(tensors["rgb"].shape),
                        (1, info.spec.n_obs_steps, 3, 32, 32),
                    )
                    self.assertEqual(
                        policy.predict(obs).shape, (info.spec.n_action_steps, 19)
                    )
                for image in (
                    np.zeros((2, 24, 32, 3), np.float32),
                    np.zeros((2, 3, 24, 32), np.uint8),
                ):
                    with self.assertRaises((ValueError, TypeError)):
                        policy.predict(
                            {"joint_state": np.zeros((2, 19), np.float32), "rgb": image}
                        )
            finally:
                policy.close()


class RealTrainingContractSmoke(unittest.TestCase):
    def setUp(self):
        import zarr
        from dexmani_real.deployment.smoke_test import export_fixture

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = export_fixture(self.temp.name)
        self.root = zarr.open_group(str(self.path), mode="a")

    def semantics(self, agent_config=None):
        from dexmani_policy.datasets.real_policy_contract import (
            build_real_policy_data_semantics,
        )

        return build_real_policy_data_semantics(
            self.path,
            task_name="smoke",
            observation_fields=[
                "joint_state",
                "point_cloud",
                "rgb",
                "contact_force",
                "fingertip_points",
                "eef_pose",
                "tactile_force",
            ],
            agent_config={} if agent_config is None else agent_config,
            action_key="action",
        )

    def cloud(self, count, *, features=6, dtype="float32"):
        import json

        self.root["data"].create_dataset(
            "point_cloud", shape=(4, count, features), dtype=dtype, overwrite=True
        )
        config = json.loads(self.root.attrs["pointcloud_config_json"])
        config["num_points"] = count
        self.root.attrs["pointcloud_config_json"] = json.dumps(config)

    def assert_compatible(self, before, after):
        from dexmani_policy.common.checkpoint_io import validate_resume_contract

        validate_resume_contract(
            {"deployment_data_semantics": before},
            {"deployment_data_semantics": after},
        )

    def test_arbitrary_positive_point_counts(self):
        for count in (1, 1536, 3072):
            with self.subTest(count=count):
                self.cloud(count)
                semantics = self.semantics(
                    {
                        "num_points": count,
                        "pc_dim": 6,
                        "pc_encoder_config": {"num_points": count, "pc_in_channels": 6},
                    }
                )
                self.assertEqual(
                    semantics["observation_fields"]["point_cloud"]["shape"], [count, 6]
                )
                self.assertEqual(semantics["pointcloud_config"]["num_points"], count)
                self.semantics()  # Agents need not declare a point count.

    def test_cloud_dimensions_and_order_remain_strict(self):
        import json

        from dexmani_policy.datasets.real_policy_contract import RealPolicyContractError

        for count, features, dtype in (
            (0, 6, "float32"),
            (1536, 5, "float32"),
            (1536, 6, "float64"),
        ):
            with self.subTest(count=count, features=features, dtype=dtype):
                self.cloud(count, features=features, dtype=dtype)
                with self.assertRaises(RealPolicyContractError):
                    self.semantics()
        self.cloud(1536)
        for config in (
            {"num_points": 1024},
            {"pc_dim": 3},
            {"pc_encoder_config": {"num_points": 1024}},
            {"pc_encoder_config": {"pc_in_channels": 3}},
        ):
            with self.subTest(agent_config=config):
                with self.assertRaises(RealPolicyContractError):
                    self.semantics(config)
        self.root.attrs["point_cloud_features"] = ["z", "y", "x", "r", "g", "b"]
        with self.assertRaises(RealPolicyContractError):
            self.semantics()
        self.root.attrs["point_cloud_features"] = ["x", "y", "z", "r", "g", "b"]
        for count, configured in ((1536, 1024), (1536, 1536.0), (1, True)):
            self.cloud(count)
            config = json.loads(self.root.attrs["pointcloud_config_json"])
            config["num_points"] = configured
            self.root.attrs["pointcloud_config_json"] = json.dumps(config)
            with self.subTest(configured=configured):
                with self.assertRaises(RealPolicyContractError):
                    self.semantics()

    def test_descriptive_metadata_does_not_gate_resume(self):
        before = self.semantics()
        changes = {
            "point_cloud_policy_id": "another implementation label",
            "point_cloud_transform": "reworded pipeline description",
            "point_cloud_sampling": "reworded sampling description",
            "point_cloud_color_source": "reworded color aggregation description",
            "fingertip_points_derivation": "reworded FK description",
            "fingertip_points_policy_id": "another FK version label",
            "eef_pose_derivation": "reworded EEF description",
            "eef_pose_algorithm_id": "another EEF version label",
            "camera_extrinsic_semantics": "reworded camera provenance",
            "contact_force_source": "reworded acquisition description",
            "schema_version": 999,
            "contact_force_si_verified": True,
            "tactile_force_si_verified": True,
            "tactile_force_spatial_geometry_verified": True,
        }
        self.root.attrs.update(changes)
        self.assert_compatible(before, self.semantics())
        for key in changes:
            del self.root.attrs[key]
        self.assert_compatible(before, self.semantics())

    def test_calibration_provenance_does_not_gate_resume(self):
        import json

        before = self.semantics()
        fingertip = json.loads(self.root.attrs["fingertip_config_json"])
        fingertip["handbase_position_eef_m"] = [0.05, 0.02, 0.10]
        fingertip["handbase_quat_eef_wxyz"] = [0.0, 0.0, 0.0, 1.0]
        transform = np.eye(4)
        transform[0, 3] = 0.25
        self.root.attrs.update(
            {
                "camera_extrinsic": transform.tolist(),
                "camera_intrinsic": [120.0, 0.0, 20.0, 0.0, 110.0, 16.0, 0.0, 0.0, 1.0],
                "camera_serial": "recalibrated-camera",
                "depth_scale_m_per_unit": 0.002,
                "point_cloud_table_plane_abcd_json": "[0.0,0.0,1.0,-0.2]",
                "fingertip_config_json": json.dumps(fingertip),
            }
        )
        geometry = self.root.attrs["camera_geometry"]
        geometry["color"]["fx"] += 10.0
        self.root.attrs["camera_geometry"] = geometry
        self.assert_compatible(before, self.semantics())
        del self.root.attrs["point_cloud_table_plane_abcd_json"]
        self.assert_compatible(before, self.semantics())

    def test_numeric_training_changes_still_fail(self):
        import json

        from dexmani_policy.datasets.real_policy_contract import RealPolicyContractError

        before = self.semantics()
        attrs = dict(self.root.attrs)
        changes = {
            "dt": attrs["dt"] * 2,
            "joint_names": list(reversed(attrs["joint_names"])),
            "finger_names": list(reversed(attrs["finger_names"])),
            "rgb_channels": ["b", "g", "r"],
            "tactile_sensor_ids": list(reversed(attrs["tactile_sensor_ids"])),
            "tactile_axis_names": ["fz", "fy", "fx"],
            "tactile_point_indices": list(reversed(attrs["tactile_point_indices"])),
        }
        config = json.loads(attrs["pointcloud_config_json"])
        config["voxel_size_m"] *= 2
        changes["pointcloud_config_json"] = json.dumps(config)
        fingertip = json.loads(attrs["fingertip_config_json"])
        fingertip["fingertip_link_names"] = list(
            reversed(fingertip["fingertip_link_names"])
        )
        changes["fingertip_config_json"] = json.dumps(fingertip)
        for key, value in changes.items():
            with self.subTest(key=key):
                self.root.attrs.put(attrs)
                self.root.attrs[key] = value
                with self.assertRaises(ValueError):
                    self.assert_compatible(before, self.semantics())
        for key, value in {
            "fingertip_points_unit": "mm",
            "fingertip_points_frame": "hand",
            "eef_pose_frame": "world",
            "eef_pose_components": "position+quaternion",
            "contact_force_unit": "N",
            "tactile_force_unit": "N",
            "action_ee_frame": "world",
            "action_ee_components": "position+quaternion",
            "obs_alignment": "obs_after_action",
            "state_alignment": "camera_frame",
        }.items():
            with self.subTest(key=key):
                self.root.attrs.put(attrs)
                self.root.attrs[key] = value
                with self.assertRaises(RealPolicyContractError):
                    self.semantics()


if __name__ == "__main__":
    unittest.main()
