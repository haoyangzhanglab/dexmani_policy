import random
from contextlib import closing

import hydra
import numpy as np
import pytest
import torch

from dexmani_policy.common.checkpoint import load_checkpoint, save_checkpoint
from dexmani_policy.common.pytorch_util import get_rng_state, set_seed
from dexmani_policy.deployment import (
    load_policy, predict_actions, prepare_observations, restore_policy, select_inference_weights,
)
from dexmani_policy.evaluation import rollout
from dexmani_policy.training.trainer import build_trainer


def test_calibrated_inference_rejects_empty_xyz_only_in_consumed_history():
    points = np.ones((3, 8, 6), dtype=np.float32)
    points[0, :, :3] = 0
    observations = {
        "point_cloud": points,
        "rgb": np.full((3, 2, 2, 3), 128, dtype=np.uint8),
        "camera_intrinsic": np.tile(np.eye(3, dtype=np.float32).ravel(), (3, 1)),
        "camera_extrinsic": np.tile(np.eye(4, dtype=np.float32)[:3].ravel(), (3, 1)),
    }
    dims = {"point_cloud": 6, "rgb": 3, "camera_intrinsic": 9, "camera_extrinsic": 12}
    options = dict(n_obs_steps=2, device="cpu", rgb_preprocessing={"resize_hw": None, "center_crop_hw": None})

    batch = prepare_observations(observations, normalization_dims=dims, **options)
    assert batch["point_cloud"].shape == (1, 2, 8, 6) and batch["point_cloud"].all()
    points[-1, :, :3] = 0
    with pytest.raises(ValueError, match=r"observation history.*empty frame at index 1"):
        prepare_observations(observations, normalization_dims=dims, **options)

    batch = prepare_observations(observations, normalization_dims={"point_cloud": 6}, **options)
    assert not batch["point_cloud"][0, -1, :, :3].any()


def test_ddim_config_and_checkpoint_overrides(small_policy_checkpoint):
    checkpoint = load_checkpoint(small_policy_checkpoint)
    with pytest.raises(ValueError, match="num_training_steps"):
        select_inference_weights(checkpoint, inference_steps=checkpoint.effective_config["agent"]["num_training_steps"] + 1)
    with pytest.raises(ValueError, match="leading DDIM"):
        load_policy(small_policy_checkpoint, inference_steps=1, device="cpu")
    with pytest.raises(ValueError, match="leading DDIM"):
        rollout.evaluate_policy(small_policy_checkpoint, inference_steps=1, device="cpu")


@pytest.mark.parametrize("width", [23, 46])
def test_proprio_frame_shape(width):
    def prepare(frame_shape):
        return prepare_observations(
            {"proprio": np.zeros((2, *frame_shape))}, normalization_dims={"proprio": width},
            n_obs_steps=2, rgb_preprocessing={}, device="cpu",
        )["proprio"]

    assert prepare((width,)).shape == (1, 2, width)
    with pytest.raises(ValueError, match=rf"proprio.*{width}"):
        prepare((1,))
    with pytest.raises(ValueError, match="proprio"):
        prepare((1, width))


def test_custom_observation_uses_checkpoint_channel_width():
    with pytest.raises(ValueError, match="custom_state.*width 7"):
        prepare_observations(
            {"custom_state": np.ones((2, 1))}, normalization_dims={"custom_state": 7},
            n_obs_steps=2, rgb_preprocessing={}, device="cpu",
        )


def test_mean_std_deployment_matches_training_model(tmp_path, small_policy_config):
    cfg = small_policy_config(tmp_path / "run", "normalization.joint_state=mean_std")
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
        policy = trainer.ema_model.eval()
        dataset = trainer.train_loader.dataset
        obs = {k: v.numpy() for k, v in dataset[0]["obs"].items()}
        batch = prepare_observations(obs, normalization_dims=policy.normalization_dims,
                                     n_obs_steps=1, rgb_preprocessing={}, device="cpu")
        expected = predict_actions(policy, batch, inference_steps=2, generator=torch.Generator().manual_seed(23))
        runtime = load_policy(tmp_path / "run", device="cpu", weights="ema", inference_steps=2, seed=23)
        with closing(runtime):
            torch.testing.assert_close(runtime.agent.normalizer.state_dict(), policy.normalizer.state_dict(),
                                       rtol=0, atol=0)
            np.testing.assert_array_equal(runtime.predict(obs), expected)


def backend_settings():
    return (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.get_float32_matmul_precision(),
    )


def test_set_seed_only_seeds_rng(monkeypatch):
    calls = []
    for module, name, label in (
        (random, "seed", "python"), (np.random, "seed", "numpy"), (torch, "manual_seed", "torch"),
    ):
        original = getattr(module, name)

        def record(seed, original=original, label=label):
            calls.append(label)
            return original(seed)

        monkeypatch.setattr(module, name, record)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    samples = []
    for _ in range(2):
        calls.clear()
        set_seed(73)
        assert backend_settings() == (True, False, "high")
        assert calls == ["python", "numpy", "torch"]
        samples.append((random.random(), np.random.random(4), torch.rand(4)))
    assert samples[0][0] == samples[1][0]
    np.testing.assert_array_equal(samples[0][1], samples[1][1])
    torch.testing.assert_close(samples[0][2], samples[1][2], rtol=0, atol=0)


@pytest.mark.parametrize("api,precision", [("restore", "high"), ("deployment", "medium")])
@pytest.mark.parametrize("case", ["raw", "ema", "missing_ema", "strict_error"])
def test_restoration_inherits_host_settings(small_policy_checkpoint, tmp_path, api, case, precision):
    checkpoint = load_checkpoint(small_policy_checkpoint)
    # Make weight selection and FP32 conversion observable.
    for state in (checkpoint.model_state, checkpoint.ema_model_state):
        for key, value in state.items():
            if value.is_floating_point():
                state[key] = value.double()
    parameter = next(key for key in checkpoint.ema_model_state if not key.startswith("normalizer."))
    checkpoint.ema_model_state[parameter] = checkpoint.ema_model_state[parameter] + 0.125
    if case == "missing_ema":
        checkpoint.ema_model_state = None
    elif case == "strict_error":
        checkpoint.model_state.pop(parameter)
    source = save_checkpoint(tmp_path / "restore.pt", checkpoint)
    weights = "ema" if case in {"ema", "missing_ema"} else "raw"
    state = checkpoint.ema_model_state if weights == "ema" else checkpoint.model_state
    cfg = checkpoint.effective_config
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision(precision)
    rng = get_rng_state()

    def restore():
        if api == "restore":
            _, _, selected = select_inference_weights(checkpoint, weights=weights, inference_steps=2)
            return restore_policy(cfg, selected, "cpu")
        return load_policy(source, weights=weights, device="cpu", seed=73, inference_steps=2)

    if case in {"missing_ema", "strict_error"}:
        error, message = (
            (ValueError, "EMA weights are absent") if case == "missing_ema"
            else (RuntimeError, "Missing key")
        )
        with pytest.raises(error, match=message):
            restore()
    else:
        restored = restore()
    assert backend_settings() == (True, False, precision)
    after = get_rng_state()
    assert rng["python"] == after["python"]
    assert rng["numpy"][0] == after["numpy"][0] and rng["numpy"][2:] == after["numpy"][2:]
    np.testing.assert_array_equal(rng["numpy"][1], after["numpy"][1])
    torch.testing.assert_close(rng["torch"], after["torch"])
    if case in {"missing_ema", "strict_error"}:
        return

    agent = restored if api == "restore" else restored.agent
    assert all(not module.training for module in agent.modules())
    expected_state = {key: value.float() if value.is_floating_point() else value for key, value in state.items()}
    torch.testing.assert_close(agent.state_dict(), expected_state, rtol=0, atol=0)
    assert all(value.dtype == torch.float32 for value in agent.state_dict().values() if value.is_floating_point())
    if api == "restore":
        return

    dataset = hydra.utils.instantiate(cfg["dataset"], _convert_="all")
    observation = {key: value.numpy() for key, value in dataset[0]["obs"].items()}
    tensors = prepare_observations(
        observation, normalization_dims=agent.normalization_dims,
        n_obs_steps=agent.n_obs_steps, rgb_preprocessing={}, device="cpu",
    )
    expected = predict_actions(agent, tensors, inference_steps=2, generator=torch.Generator().manual_seed(73))
    assert restored.info.weights == weights
    assert restored.info.control_action_dim == 19
    assert restored.info.action_mode == "joint"
    assert restored.info.data_processing == cfg["data_processing"]
    generator_state = restored.generator.get_state()
    restored.warmup(observation)
    torch.testing.assert_close(restored.generator.get_state(), generator_state)
    actual = restored.predict(observation)
    restored.reset_episode()
    np.testing.assert_array_equal(restored.predict(observation), actual)
    restored.close()
    np.testing.assert_array_equal(actual, expected)
    assert backend_settings() == (True, False, precision)


def test_deployment_rtc_uses_physical_prefix_and_preserves_warmup_rng(small_policy_checkpoint):
    observation = {
        "joint_state": np.zeros((1, 19), dtype=np.float32),
        "point_cloud": np.zeros((1, 8, 6), dtype=np.float32),
    }
    prefix = np.full((2, 19), 0.1, dtype=np.float64)
    with closing(load_policy(small_policy_checkpoint, device="cpu", seed=73, inference_steps=2)) as runtime:
        tensors = prepare_observations(
            observation, normalization_dims=runtime.agent.normalization_dims,
            n_obs_steps=runtime.info.n_obs_steps, rgb_preprocessing={}, device="cpu",
        )
        expected = predict_actions(
            runtime.agent, tensors, inference_steps=2, generator=torch.Generator().manual_seed(73),
            rtc_prefix=torch.from_numpy(prefix).float().unsqueeze(0), delay_steps=1, rtc_guidance_cap=0.2,
        )
        generator_state = runtime.generator.get_state()
        runtime.warmup(observation, rtc_delay=1, rtc_guidance_cap=0.2)
        torch.testing.assert_close(runtime.generator.get_state(), generator_state)
        actual = runtime.predict(observation, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=0.2)
        np.testing.assert_array_equal(actual, expected)

        for invalid, delay, message in (
            (None, 1, "No prefix"),
            (prefix[:, :-1], 0, "Invalid physical RTC prefix"),
            (prefix, 3, "Invalid physical RTC prefix"),
            (np.full_like(prefix, np.nan), 0, "Nonfinite physical RTC prefix"),
        ):
            with pytest.raises(ValueError, match=message):
                runtime.predict(observation, rtc_prefix=invalid, delay_steps=delay)
