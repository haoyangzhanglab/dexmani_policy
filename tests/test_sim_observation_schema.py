from dataclasses import replace
import io
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.normalization import (
    LinearNormalizer, normalization_dims_from_config, validate_normalizer_state,
)
from dexmani_policy.deployment.runtime import LoadedPolicy, PolicyInfo
from dexmani_policy.datasets.preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import build_dataset_and_normalizer, validate_config


def write_sim_data(path, *, flat_force=False):
    rng = np.random.default_rng(12)
    n = 8
    data = {
        "joint_state": rng.normal(size=(n, 19)).astype("f4"),
        "action": rng.normal(size=(n, 19)).astype("f4"),
        "point_cloud": rng.normal(size=(n, 16, 6)).astype("f4"),
        "fingertip_xyz": rng.normal(size=(n, 5, 3)).astype("f4"),
        "finger_tactile_force": rng.normal(size=(n, 5, 3)).astype("f4"),
        "eef_pose": np.tile(np.array([.2, .1, .3, 1, 0, 0, 0, 1, 0], dtype="f4"), (n, 1)),
    }
    if flat_force:
        data["finger_tactile_force"] = data["finger_tactile_force"].reshape(n, 15)
    root = zarr.open_group(str(path), mode="w")
    for key, values in data.items():
        root.create_dataset(f"data/{key}", data=values, chunks=(4, *values.shape[1:]))
    root.create_dataset("meta/episode_ends", data=np.array([n], dtype=np.int64))
    return data


def small_config(path):
    cfg = load_config("interaction_flow")
    cfg.dataset.zarr_path = str(path)
    cfg.dataset.augmentation_cfg = None
    cfg.dataset.horizon = cfg.agent.horizon = 4
    cfg.dataset.pad_before = cfg.dataset.pad_after = 0
    cfg.agent.n_action_steps = 2
    for key, value in dict(token_dim=24, num_patches=4, group_size=4, patch_depth=1,
                           num_scene_tokens=2, hidden_dim=32, n_head=4,
                           timestep_embed_dim=32, num_inference_steps=1).items():
        cfg.agent[key] = value
    return cfg


def test_interaction_config_and_dataset_preserve_finger_axes(tmp_path):
    cfg = load_config("interaction_flow")
    validate_config(cfg)
    assert set(cfg.dataset.sensor_modalities) == {
        "joint_state", "point_cloud", "fingertip_xyz", "eef_pose", "finger_tactile_force"
    }
    assert "wrist_pose_key" not in cfg.agent

    path = tmp_path / "sim.zarr"
    data = write_sim_data(path)
    cfg = small_config(path)
    dataset, normalizer = build_dataset_and_normalizer(cfg, verbose=False)
    sample = dataset[0]
    for key in ("fingertip_xyz", "finger_tactile_force"):
        assert sample["obs"][key].shape == (2, 5, 3)
        np.testing.assert_array_equal(sample["obs"][key], data[key][:2])
    np.testing.assert_array_equal(sample["obs"]["eef_pose"], data["eef_pose"][:2])
    dims = normalization_dims_from_config(cfg)
    assert (dims["fingertip_xyz"], dims["finger_tactile_force"], dims["eef_pose"]) == (15, 15, 9)
    rows = dataset.sampler.observation_source_rows
    params = normalizer.params_dict["finger_tactile_force"]
    assert params["scale"].shape == params["offset"].shape == (5, 3)
    np.testing.assert_allclose(-params["offset"] / params["scale"],
                               data["finger_tactile_force"][rows].mean(axis=0), atol=1e-6)
    values = sample["obs"]["finger_tactile_force"].unsqueeze(0)
    normalized = normalizer["finger_tactile_force"].normalize(values)
    assert normalized.shape == (1, 2, 5, 3)
    torch.testing.assert_close(normalizer["finger_tactile_force"].unnormalize(normalized), values)

    runner = BaseRunner(2, 1, sensor_modalities=list(cfg.dataset.sensor_modalities))
    for row in range(3):
        runner.update_obs({key: data[key][row] for key in cfg.dataset.sensor_modalities})
    stacked = runner.get_stacked_obs()
    for key in cfg.dataset.sensor_modalities:
        np.testing.assert_array_equal(stacked[key], data[key][1:3])


def test_training_rejects_flat_force_statistics(tmp_path):
    path = tmp_path / "flat.zarr"
    write_sim_data(path, flat_force=True)
    with pytest.raises(ValueError, match="finger_tactile_force requires stored shape"):
        build_dataset_and_normalizer(small_config(path), verbose=False)


def test_cpu_dataset_agent_and_runtime_consume_new_schema(tmp_path):
    path = tmp_path / "sim.zarr"
    data = write_sim_data(path)
    cfg = small_config(path)
    dataset, normalizer = build_dataset_and_normalizer(cfg, verbose=False)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.action_key = cfg.action_key
    agent.load_normalizer_from_dataset(normalizer)
    agent.set_normalization_spec(dict(cfg.normalization), normalization_dims_from_config(cfg))
    agent.validate_normalizer()
    sample = dataset[0]
    batch = {"obs": {key: value.unsqueeze(0) for key, value in sample["obs"].items()},
             "action": sample["action"].unsqueeze(0)}
    agent.eval()
    with torch.no_grad():
        loss, _ = agent.compute_loss(batch)
        result = agent.predict_action(batch["obs"], inference_steps=1)
    assert torch.isfinite(loss)
    assert result["control_action"].shape == (1, 2, 19)
    assert torch.isfinite(result["control_action"]).all()
    for key in ("fingertip_xyz", "finger_tactile_force"):
        bad = dict(batch["obs"])
        bad[key] = bad[key].reshape(1, 2, 15)
        with pytest.raises(ValueError, match=key if key == "fingertip_xyz" else "feature shape"):
            agent.predict_action(bad, inference_steps=1)

    plain = OmegaConf.to_container(cfg, resolve=True)
    info = PolicyInfo(Path("."), Path("unused.pt"), "interaction_flow", cfg.task_name,
                      tuple(cfg.dataset.sensor_modalities), 2, 2, 4, "joint",
                      None, None, None, "raw", 1, 19, 19,
                      rgb_preprocessing_kwargs(plain["dataset"]), dict(cfg.normalization), 6)
    policy = LoadedPolicy(agent, plain, info, device="cpu", seed=0)
    observation = {key: data[key][:3] for key in cfg.dataset.sensor_modalities}
    assert policy.predict(observation).shape == (3, 19)
    # A synthetic producer recipe supplies the point count used only by warmup.
    policy.info = replace(policy.info, pointcloud_config={"num_points": 16})
    timing = policy.warmup(samples=1)
    assert len(timing["bootstrap"]) == 1


def test_dense_tactile_retains_array_layout_and_aggregate_evidence():
    from dexmani_policy.agents.obs_encoder.interaction.encoder import InteractionObsEncoder
    from dexmani_policy.agents.obs_encoder.tactile.xhand_frame import XHandTactileFrameEncoder

    encoder = XHandTactileFrameEncoder("tactile_force").eval()
    valid = torch.ones(1, 2, 5, dtype=torch.bool)
    valid[:, :, 1] = False
    tactile = torch.randn(1, 2, 5, 120, 3)
    tactile[:, :, 1] = float("nan")
    with torch.no_grad():
        output = encoder(tactile, valid)
    assert output.shape == (1, 2, 5, 64)
    assert torch.isfinite(output).all()
    assert not output[:, :, 1].any()
    interaction = InteractionObsEncoder(n_obs_steps=2, token_dim=24, num_patches=4,
                                        group_size=4, patch_depth=1, num_scene_tokens=2,
                                        tactile_input_key="tactile_force")
    assert set(interaction.consumed_observation_fields) == {
        "joint_state", "point_cloud", "fingertip_xyz", "eef_pose", "finger_tactile_force", "tactile_force"
    }


@pytest.mark.parametrize("key", ["fingertip_xyz", "finger_tactile_force"])
@pytest.mark.parametrize("mode", ["gaussian", "limits"])
@pytest.mark.parametrize("chunks", [False, True])
@pytest.mark.parametrize("fit_offset", [False, True])
def test_finger_statistics_preserve_flat15_semantics_and_checkpoint(key, mode, chunks, fit_offset):
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(11, 5, 3, generator=generator)
    values = values * torch.arange(1., 16.).reshape(5, 3) + torch.arange(15.).reshape(5, 3)
    flat = values.reshape(11, 15)
    actual, reference = LinearNormalizer(), LinearNormalizer()
    kwargs = dict(mode=mode, fit_offset=fit_offset)
    if chunks:
        actual.fit_field_chunks(key, values.split(4), **kwargs)
        reference.fit_field_chunks("former_flat", flat.split(4), **kwargs)
    else:
        actual.fit({key: values}, **kwargs)
        reference.fit({"former_flat": flat}, **kwargs)
    for name in ("scale", "offset"):
        assert actual.params_dict[key][name].shape == (5, 3)
        torch.testing.assert_close(actual.params_dict[key][name].flatten(),
                                   reference.params_dict["former_flat"][name], atol=0, rtol=0)
    for name, stat in actual.input_stats[key].items():
        assert stat.shape == (5, 3)
        torch.testing.assert_close(stat.flatten(), reference.input_stats["former_flat"][name], atol=0, rtol=0)

    history = values[:8].reshape(2, 4, 5, 3)
    expected = reference["former_flat"].normalize(history.reshape(2, 4, 15)).reshape(2, 4, 5, 3)
    torch.testing.assert_close(actual[key].normalize(history), expected, atol=0, rtol=0)
    saved = io.BytesIO()
    torch.save(actual.state_dict(), saved)
    saved.seek(0)
    restored = LinearNormalizer()
    restored.load_state_dict(torch.load(saved, weights_only=True))
    validate_normalizer_state(restored, {key: mode}, {key: 15})
    torch.testing.assert_close(restored[key].normalize(history), expected, atol=0, rtol=0)
    torch.testing.assert_close(restored[key].unnormalize(expected), history)

    # Matching element counts must never permit moving time into a feature axis.
    for bad in (history.reshape(2, 4, 15), history.reshape(2, 20, 3), history.reshape(8, 3, 5)):
        with pytest.raises(ValueError, match="feature shape"):
            restored[key].normalize(bad)
    with pytest.raises(ValueError, match="trailing feature shape"):
        LinearNormalizer().fit({key: flat}, **kwargs)
    with pytest.raises(ValueError, match="trailing feature shape"):
        LinearNormalizer().fit_field_chunks(key, flat.split(4), **kwargs)


@pytest.mark.parametrize("shape", [(3,), (15,), (1, 5, 3)])
def test_finger_normalizer_rejects_shared_or_flat_checkpoint_statistics(shape):
    normalizer = LinearNormalizer()
    normalizer.params_dict["finger_tactile_force"] = torch.nn.ParameterDict(
        {"scale": torch.ones(shape), "offset": torch.zeros(shape)}
    )
    with pytest.raises(ValueError, match="expected feature shape"):
        validate_normalizer_state(normalizer, {"finger_tactile_force": "gaussian"},
                                  {"finger_tactile_force": 15})


def test_dense_tactile_normalization_still_shares_three_axis_statistics():
    values = torch.arange(2 * 5 * 120 * 3, dtype=torch.float32).reshape(2, 5, 120, 3)
    actual, reference = LinearNormalizer(), LinearNormalizer()
    actual.fit_field_chunks("tactile_force", values.split(1), mode="gaussian")
    reference.fit_field_chunks("dense_axes", values.reshape(2, 600, 3).split(1), mode="gaussian")
    assert actual.params_dict["tactile_force"]["scale"].shape == (3,)
    validate_normalizer_state(actual, {"tactile_force": "gaussian"}, {"tactile_force": 3})
    torch.testing.assert_close(actual["tactile_force"].normalize(values),
                               reference["dense_axes"].normalize(values), atol=0, rtol=0)
