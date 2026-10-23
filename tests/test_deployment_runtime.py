import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from dexmani_policy.deployment import PolicyInfo, PolicyRuntime
from dexmani_policy.deployment.runtime import make_warmup_observation
from dexmani_policy.training.data import capture_real_runtime


def policy_info(**updates):
    config = {
        "policy_name": "test",
        "task_name": "test",
        "action_key": "action",
        "agent": {"horizon": 4, "n_obs_steps": 2, "n_action_steps": 2, "action_dim": 28},
        "dataset": {
            "sensor_modalities": [
                "rgb", "depth", "hand_pointcloud", "point_cloud", "tactile_valid",
            ],
            "pc_dim": 3,
        },
        "real_runtime": {
            "dt": 0.02,
            "depth": {
                "shape": [2, 3], "dtype": "uint16", "unit": "raw_counts", "depth_scale_m": 0.001,
            },
            "hand_pointcloud": {"num_points": 4},
            "pointcloud": {"num_points": 5},
        },
        "hand_data": None,
    }
    config.update(updates)
    return PolicyInfo(Path("experiment"), Path("checkpoint.pt"), OmegaConf.create(config), "raw", 2)


def recording_agent(info):
    prediction = torch.arange(info.horizon * info.action_dim, dtype=torch.float32)
    return SimpleNamespace(
        horizon=info.horizon,
        n_obs_steps=info.n_obs_steps,
        n_action_steps=info.n_action_steps,
        action_dim=info.action_dim,
        control_action_dim=info.control_action_dim,
        use_aux_ee=True,
        ee_dim=9,
        obs_encoder=SimpleNamespace(use_coord_only=True),
        consumed_observation_fields=info.observation_fields,
        predict_action=Mock(return_value={"pred_action": prediction.reshape(1, info.horizon, -1)}),
        to=Mock(),
    )


class DeploymentRuntimeTest(unittest.TestCase):
    def test_real_data_recipe_preserves_depth_units_and_production_point_counts(self):
        arrays = {
            "point_cloud": np.zeros((2, 5, 6), dtype=np.float32),
            "hand_pointcloud": np.zeros((2, 4, 6), dtype=np.float32),
            "depth": np.zeros((2, 2, 3), dtype=np.uint16),
        }
        attrs = {
            "format": "dexmani.real.canonical",
            "task_name": "test",
            "dt": 0.02,
            "pointcloud_config": {"num_points": 5, "sampling_method": "quickfps"},
            "hand_pointcloud_config": {
                "num_points": 4, "seed": 7,
                "sampling_method": "fixed_per_link_area_triangle",
                "link_selection": "visual_area_gte_0.0001_m2",
                "color_rule": "urdf_material_rgb_or_fixed_gray_0.5",
            },
            "depth_scale_m_per_unit": 0.001,
        }
        buffer = MagicMock(attrs=attrs)
        buffer.__getitem__.side_effect = arrays.__getitem__
        dataset = SimpleNamespace(replay_buffer=buffer, sensor_modalities=list(arrays))
        recipe = capture_real_runtime(dataset, SimpleNamespace(task_name="test"))
        self.assertEqual(recipe["pointcloud"], attrs["pointcloud_config"])
        self.assertEqual(recipe["hand_pointcloud"], attrs["hand_pointcloud_config"])
        self.assertEqual(recipe["depth"], {
            "shape": [2, 3], "dtype": "uint16", "unit": "raw_counts", "depth_scale_m": 0.001,
        })
        observation = make_warmup_observation(policy_info(real_runtime=recipe), rgb_hw=(2, 3))
        for field, expected in arrays.items():
            self.assertEqual(observation[field].shape, expected.shape)
            self.assertEqual(observation[field].dtype, expected.dtype)

        del attrs["depth_scale_m_per_unit"]
        with self.assertRaises(KeyError):
            capture_real_runtime(dataset, SimpleNamespace(task_name="test"))

    def test_control_period_requires_real_runtime_metadata(self):
        cases = (
            ({"hand_data": None}, 0.02),
            ({"real_runtime": None, "hand_data": None}, None),
            ({"real_runtime": None, "hand_data": {"dt": 0.04}}, None),
            ({"real_runtime": {"dt": None}, "hand_data": {"dt": 0.04}}, None),
        )
        for updates, expected in cases:
            with self.subTest(updates=updates):
                info = policy_info(dataset={"sensor_modalities": ["joint_state"]}, **updates)
                self.assertEqual(info.control_dt_s, expected)

    def test_warmup_requires_the_production_point_count(self):
        info = policy_info()
        recipe = OmegaConf.to_container(info.saved_config.real_runtime)
        for cloud in (None, {}, {"num_points": 0}):
            with self.subTest(cloud=cloud):
                recipe["pointcloud"] = cloud
                with self.assertRaises(ValueError):
                    make_warmup_observation(policy_info(real_runtime=recipe), rgb_hw=(2, 3))

    def test_warmup_repeated_patterns_preserve_exact_values(self):
        recipe = OmegaConf.to_container(policy_info().saved_config.real_runtime)
        recipe["depth"]["shape"] = [17, 19]
        recipe["hand_pointcloud"]["num_points"] = 20
        recipe["pointcloud"]["num_points"] = 23
        fields = [
            "rgb", "depth", "hand_pointcloud", "point_cloud", "joint_state", "eef_pose",
            "finger_tactile_force", "fingertip_xyz", "taxel_tactile_force",
        ]
        info = policy_info(dataset={"sensor_modalities": fields}, real_runtime=recipe)
        observation = make_warmup_observation(info, rgb_hw=(17, 19))
        for name, value in observation.items():
            with self.subTest(name=name):
                indices = np.arange(value.size).reshape(value.shape)
                expected = (
                    (indices % 251 + 1).astype(value.dtype)
                    if value.dtype.kind == "u"
                    else ((indices % 101) / 100.0 - 0.5).astype(value.dtype)
                )
                if name == "hand_pointcloud":
                    expected[..., 3:] = 0.5
                np.testing.assert_array_equal(value, expected)
                self.assertTrue(value.flags.c_contiguous)
                self.assertTrue(value.flags.writeable)

    def test_observation_conversion_preserves_dtype_and_device(self):
        info = policy_info()
        observation = make_warmup_observation(info, rgb_hw=(2, 3))
        observation["point_cloud"] = observation["point_cloud"][:, ::-1].astype(np.float64)
        devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                runtime = PolicyRuntime(recording_agent(info), info, device=device, seed=0)
                tensors = runtime.prepare_observations(observation)
                for name, value in observation.items():
                    if name == "point_cloud":
                        value = value[..., :info.pc_dim]
                    expected = torch.from_numpy(np.ascontiguousarray(value))
                    if name not in {"rgb", "tactile_valid"}:
                        expected = expected.to(dtype=torch.float32)
                    expected = expected.unsqueeze(0).to(device)
                    torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)

    def test_prediction_preserves_input_recipe_and_physical_future(self):
        info = policy_info()
        agent = recording_agent(info)
        runtime = PolicyRuntime(agent, info, device="cpu", seed=0)
        observation = make_warmup_observation(info, rgb_hw=(2, 3))
        observation = {
            name: np.concatenate((value[:1], value), axis=0)
            for name, value in observation.items()
        }
        observation["hand_pointcloud"][0] = np.nan
        future = runtime.predict(observation)
        tensors = agent.predict_action.call_args.args[0]

        for name, value in observation.items():
            expected = value[-2:]
            if name == "point_cloud":
                expected = expected[..., :3]
            np.testing.assert_array_equal(tensors[name][0].numpy(), expected)
            dtype = {"rgb": torch.uint8, "tactile_valid": torch.bool}.get(name, torch.float32)
            self.assertEqual(tensors[name].dtype, dtype)

        prediction = agent.predict_action.return_value["pred_action"][0].numpy()
        np.testing.assert_array_equal(future, prediction[1:, :19])
        self.assertEqual(future.dtype, np.float64)
        self.assertTrue(future.flags.c_contiguous)
        np.testing.assert_array_equal(runtime.predict(observation), future)

        runtime.close()
        runtime.close()
        agent.to.assert_called_once_with("cpu")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.predict(observation)

    def test_invalid_static_recipe_fails_during_metadata_construction(self):
        original = policy_info()
        for field, key, value in (
            ("depth", "depth_scale_m", 0),
            ("depth", "dtype", "float32"),
            ("hand_pointcloud", "num_points", 0),
        ):
            with self.subTest(field=field, key=key):
                recipe = OmegaConf.to_container(original.saved_config.real_runtime)
                recipe[field][key] = value
                with self.assertRaises(ValueError):
                    policy_info(real_runtime=recipe)

    def test_static_recipe_cannot_change_after_validation(self):
        recipe = OmegaConf.to_container(policy_info().saved_config.real_runtime)
        info = policy_info(real_runtime=recipe)
        recipe["depth"]["shape"][0] = 99
        info.depth_config["shape"][0] = 88
        self.assertEqual(info.depth_config["shape"], [2, 3])
        with self.assertRaises(ReadonlyConfigError):
            info.saved_config.real_runtime.depth.shape[0] = 77

    def test_invalid_observation_is_rejected_before_prediction(self):
        info = policy_info()
        agent = recording_agent(info)
        runtime = PolicyRuntime(agent, info, device="cpu", seed=0)
        observation = make_warmup_observation(info, rgb_hw=(2, 3))
        cases = (
            ("depth", np.zeros((2, 2, 3), dtype=np.float32)),
            ("depth", np.zeros((2, 3, 2), dtype=np.uint16)),
            ("depth", np.zeros((1, 2, 3), dtype=np.uint16)),
            ("hand_pointcloud", np.zeros((2, 4, 6), dtype=np.float64)),
            ("hand_pointcloud", np.zeros((2, 3, 6), dtype=np.float32)),
            ("hand_pointcloud", np.full((2, 4, 6), np.nan, dtype=np.float32)),
            ("point_cloud", np.zeros((2, 5, 2), dtype=np.float32)),
        )
        for field, value in cases:
            with self.subTest(field=field, shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    runtime.predict({**observation, field: value})
        with self.assertRaisesRegex(ValueError, "Missing"):
            runtime.predict({name: value for name, value in observation.items() if name != "depth"})
        agent.predict_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
