import copy
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.robot_dataset import RobotDataset
from dexmani_policy.deployment import PolicyRuntime, inspect_policy, load_policy


class SamplingPolicy(BasePolicy):
    def __init__(self, horizon, n_obs_steps, n_action_steps, action_dim, observation_fields):
        encoder = nn.Linear(19, action_dim)
        encoder.consumed_observation_fields = tuple(observation_fields)
        super().__init__(encoder, nn.Identity(), horizon, n_obs_steps, n_action_steps, action_dim)

    @property
    def action_context_shapes(self):
        return {"reference_state": [19]}

    def predict_action(
        self, obs, *, inference_steps, generator, action_context=None,
        rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0,
    ):
        values = self.obs_encoder(obs["joint_state"][:, -1])[:, None]
        noise = torch.randn((1, self.horizon, self.action_dim), generator=generator)
        prediction = self.normalizer["action"].unnormalize(values + noise)
        if action_context is not None:
            prediction = prediction + action_context["reference_state"][:, None]
        if rtc_prefix is not None:
            start = self.action_start_index
            prediction[:, start : start + delay_steps] = rtc_prefix[:, :delay_steps]
        return self.format_action_output(prediction)


@pytest.fixture
def saved_policy(tmp_path):
    fields = ["joint_state", "point_cloud", "rgb", "depth", "hand_pointcloud", "tactile_valid"]
    agent_config = dict(horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19, observation_fields=fields)
    model = SamplingPolicy(**agent_config)
    spec = {key: "identity" for key in fields} | {"joint_state": "limits", "action": "limits"}
    dims = {"joint_state": 19, "action": 19}
    model.set_normalization_spec(spec, dims)
    model.normalizer["action"].scale.fill_(2)
    model.normalizer["action"].offset.fill_(0.25)
    ema = copy.deepcopy(model)
    ema.obs_encoder.weight.data.add_(1)
    config = {
        "policy_name": "sampling", "task_name": "test", "action_key": "action",
        "agent": {"_target_": f"{__name__}.SamplingPolicy", **agent_config},
        "dataset": {
            "sensor_modalities": fields, "pc_dim": 3,
            "rgb_preprocess_size": [8, 8], "rgb_random_crop_size": [4, 4], "rgb_keep_uint8": True,
        },
        "policy_metadata": model.inference_metadata,
        "normalization": spec, "normalization_dims": dims,
        "real_runtime": {
            "dt": 0.04, "pointcloud": {"num_points": 8}, "hand_pointcloud": {"num_points": 5},
            "depth": {"shape": [8, 8], "dtype": "uint16", "unit": "raw_counts", "depth_scale_m": 0.001},
        },
    }
    path = save_checkpoint(tmp_path / "checkpoints/policy.pt", Checkpoint(
        epoch=1, global_step=10, model_state=model.state_dict(), ema_model_state=ema.state_dict(),
        optimizer_state={}, scheduler_state={}, ema_updater_step=10,
        rng_state=get_rng_state(), effective_config=config,
    ))
    OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    info = inspect_policy(tmp_path, checkpoint=path, weights="raw", inference_steps=2)
    rng = np.random.default_rng(7)
    observation = {
        "joint_state": rng.normal(size=(3, 19)),
        "point_cloud": rng.random((3, 8, 6), dtype=np.float32),
        "rgb": rng.integers(0, 256, (3, 8, 8, 3), dtype=np.uint8),
        "depth": np.full((3, 8, 8), 1000, dtype=np.uint16),
        "hand_pointcloud": rng.random((3, 5, 6), dtype=np.float32),
        "tactile_valid": np.ones((3, 5), dtype=np.bool_),
    }
    return info, {"raw": model, "ema": ema}, observation


def test_inspection_without_torch(saved_policy):
    info, _, _ = saved_policy
    code = """
import json, sys
from dexmani_policy.deployment import inspect_policy
info = inspect_policy(sys.argv[1], checkpoint=sys.argv[2], weights='raw', inference_steps=2)
assert 'torch' not in sys.modules
print(json.dumps(info.to_dict()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(info.experiment_dir), str(info.checkpoint_path)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(result.stdout)
    assert "saved_config" not in summary
    assert summary["future_steps"] == 3 and summary["control_dt_s"] == 0.04
    assert summary["action_context_shapes"] == {"reference_state": [19]}


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_deployment_checkpoint_predictions(saved_policy, weights):
    info, models, observation = saved_policy
    info = replace(info, weights=weights)
    rng = get_rng_state()
    runtime = load_policy(info, device="cpu", seed=9)
    after = get_rng_state()
    assert rng["python"] == after["python"]
    np.testing.assert_array_equal(rng["numpy"][1], after["numpy"][1])
    torch.testing.assert_close(rng["torch"], after["torch"], rtol=0, atol=0)
    tensors = runtime.prepare_observations(observation)
    with torch.no_grad():
        expected = models[weights].predict_action(
            tensors, inference_steps=2, generator=torch.Generator().manual_seed(9),
        )["future_actions"][0]
    prediction = runtime.predict(observation)
    np.testing.assert_array_equal(prediction, expected.numpy())
    assert prediction.shape == (3, 19) and prediction.dtype == np.float64
    assert prediction.flags.c_contiguous
    torch.testing.assert_close(tensors["depth"], torch.full((1, 2, 8, 8), 1000.))
    assert tensors["tactile_valid"].dtype == torch.bool
    assert tensors["point_cloud"].shape == (1, 2, 8, 3)
    runtime.close()
    runtime.close()
    assert runtime.agent is None


@pytest.mark.parametrize("keep_uint8", [False, True])
def test_deployment_preprocessing_matches_dataset(saved_policy, demonstrations, keep_uint8):
    info, _, _ = saved_policy
    dataset = RobotDataset(
        demonstrations, sensor_modalities=["joint_state", "point_cloud", "rgb"],
        horizon=2, val_ratio=0.25, pc_dim=3, rgb_preprocess_size=(8, 8),
        rgb_random_crop_size=(4, 4), rgb_keep_uint8=keep_uint8,
    ).get_validation_dataset()
    recipe = dict(info.saved_config["dataset"], rgb_keep_uint8=keep_uint8)
    info = replace(
        info, observation_fields=tuple(dataset.sensor_modalities),
        rgb_preprocessing=rgb_preprocessing_kwargs(recipe),
    )
    runtime = PolicyRuntime(
        SimpleNamespace(consumed_observation_fields=info.observation_fields), info, device="cpu", seed=0,
    )
    rows = dataset.sampler.sample_source_rows(0)
    raw = {key: dataset.replay_buffer[key][:][rows] for key in info.observation_fields}
    actual = runtime.prepare_observations(raw)
    expected = dataset[0]["obs"]
    for key in expected:
        torch.testing.assert_close(actual[key][0], expected[key])


def test_runtime_random_stream_reset_and_warmup(saved_policy):
    info, _, observation = saved_policy
    runtime = load_policy(info, device="cpu", seed=13)
    reference = load_policy(info, device="cpu", seed=13)
    host_rng = torch.get_rng_state()
    first = runtime.predict(observation)
    np.testing.assert_array_equal(first, reference.predict(observation))
    timing = runtime.warmup(samples=2, rgb_hw=(8, 8), rtc_delay=1, rtc_guidance_cap=2.)
    assert all(len(samples) == 2 and min(samples) > 0 for samples in timing.values())
    np.testing.assert_array_equal(runtime.predict(observation), reference.predict(observation))
    runtime.reset_episode()
    np.testing.assert_array_equal(runtime.predict(observation), first)
    torch.testing.assert_close(torch.get_rng_state(), host_rng, rtol=0, atol=0)


def test_action_context_and_rtc_boundary(saved_policy):
    info, _, observation = saved_policy
    runtime = load_policy(info, device="cpu", seed=11)
    first = runtime.predict(observation)
    runtime.reset_episode()
    shifted = runtime.predict(observation, action_context={"reference_state": np.ones(19)})
    np.testing.assert_allclose(shifted, first + 1, rtol=1e-6, atol=1e-6)
    prefix = np.full((2, 19), 0.5)
    prediction = runtime.predict(observation, rtc_prefix=prefix, delay_steps=1)
    np.testing.assert_array_equal(prediction[0], prefix[0])
    with pytest.raises(ValueError, match="action_context"):
        runtime.predict(observation, action_context={"reference_state": np.ones(12)})
    with pytest.raises(ValueError, match="RTC prefix"):
        runtime.predict(observation, rtc_prefix=prefix, delay_steps=3)


@pytest.mark.parametrize("fault", ["history", "depth", "hand_cloud", "nonfinite"])
def test_raw_observation_boundary(saved_policy, fault):
    info, _, observation = saved_policy
    runtime = load_policy(info, device="cpu", seed=0)
    if fault == "history":
        observation["joint_state"] = observation["joint_state"][:1]
    elif fault == "depth":
        observation["depth"] = observation["depth"].astype(np.float32) * 0.001
    elif fault == "hand_cloud":
        observation["hand_pointcloud"] = observation["hand_pointcloud"][:, :3]
    else:
        observation["joint_state"][-1, 0] = np.nan
    with pytest.raises(ValueError):
        runtime.prepare_observations(observation)


@pytest.mark.parametrize("name, shape", [
    ("joint_state", (3, 1)),
    ("finger_tactile_force", (3, 1, 3)),
    ("taxel_tactile_force", (3, 5, 1, 3)),
    ("camera_intrinsic", (3, 3, 3)),
    ("camera_extrinsic", (3, 3, 4)),
    ("tactile_valid", (3, 1)),
    ("point_cloud", (3, 8, 1, 6)),
    ("rgb", (3, 2, 8, 8, 3)),
])
def test_deployment_rejects_incorrect_layout(saved_policy, name, shape):
    info, _, _ = saved_policy
    info = replace(info, observation_fields=(name,))
    runtime = PolicyRuntime(
        SimpleNamespace(consumed_observation_fields=(name,)), info, device="cpu", seed=0,
    )
    observation = {name: np.zeros(shape, dtype=np.uint8 if name == "rgb" else np.float32)}
    with pytest.raises(ValueError, match="shape"):
        runtime.prepare_observations(observation)


def test_loading_rejects_mismatched_recipe_and_missing_ema(saved_policy):
    info, _, _ = saved_policy
    config = copy.deepcopy(info.saved_config)
    config["normalization"]["action"] = "gaussian"
    with pytest.raises(ValueError, match="inference recipe mismatch"):
        load_policy(replace(info, saved_config=config), device="cpu")
    checkpoint = load_checkpoint(info.checkpoint_path)
    save_checkpoint(info.checkpoint_path, replace(checkpoint, ema_model_state=None))
    with pytest.raises(ValueError, match="EMA weights are absent"):
        load_policy(replace(info, weights="ema"), device="cpu")
