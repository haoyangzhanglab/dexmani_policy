"""Synthetic dataset, normalization and policy checks on CPU or --device cuda:0."""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.config import validate_observation_fields
from dexmani_policy.common.normalizer import LinearNormalizer, normalizer_shapes_from_state
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.datasets.variants.r3d import R3DDataset
from dexmani_policy.datasets.variants.tactile import AdapTacDataset, InterActDexDataset
from dexmani_policy.deployment import inspect_policy, load_policy
from dexmani_policy.train import run_training
from dexmani_policy.training.common.config import assemble_effective_config, load_policy_config, prepare_training, validate_config
from dexmani_policy.training.common.data import build_dataset, fit_normalizer
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.common.resume import prepare_resume


def check_splits():
    for n, maximum, val, expected in (
        (150, 100, 20, (100, 20, 30)), (150, 100, 0, (100, 0, 50)),
        (150, None, 20, (130, 20, 0)), (150, None, 0, (150, 0, 0)),
        (80, 100, 0, (80, 0, 0)), (1, None, 0, (1, 0, 0)),
    ):
        first = split_episodes(n, seed=42, max_train_episodes=maximum, val_episodes=val)
        second = split_episodes(n, seed=42, max_train_episodes=maximum, val_episodes=val)
        train, validation = first
        assert train.dtype == validation.dtype == bool
        assert not (train & validation).any()
        assert (train.sum(), validation.sum(), (~(train | validation)).sum()) == expected
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
    invalid_cases: tuple[tuple[Any, Any, Any], ...] = (
        (110, 100, 20), (100, 100, 20), (1, None, 1), (0, None, 0),
        (10, 0, 0), (10, -1, 0), (10, None, -1), (True, None, 0),
        (10, 1.5, 0), (10, None, True), (10, None, 1.5),
    )
    for n, maximum, val in invalid_cases:
        try:
            split_episodes(n, max_train_episodes=maximum, val_episodes=val)
        except ValueError:
            pass
        else:
            raise AssertionError((n, maximum, val))
    np.random.seed(17)
    expected_random = np.random.random(4)
    np.random.seed(17)
    split_episodes(150, max_train_episodes=100, val_episodes=20)
    np.testing.assert_array_equal(np.random.random(4), expected_random)


def make_source(path, ends):
    source = zarr.open_group(str(path), mode="w")
    source.create_dataset("meta/episode_ends", data=np.asarray(ends))
    rows = int(ends[-1])
    rng = np.random.default_rng(11)
    for key, shape in (
        ("action", (rows, 19)), ("action_ee", (rows, 21)), ("joint_state", (rows, 19)),
        ("point_cloud", (rows, 8, 6)), ("hand_pointcloud", (rows, 8, 6)),
        ("object_pointcloud", (rows, 16, 6)), ("finger_tactile_force", (rows, 5, 3)),
    ):
        source.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype(np.float32))
    return source


def check_source_statistics(root):
    path = root / "statistics.zarr"
    ends = np.arange(1, 151) * 3
    source = make_source(path, ends)
    train, val = split_episodes(150, max_train_episodes=100, val_episodes=20)
    selected_rows = np.repeat(train, 3)
    data = source["data/action"][:]
    data[np.repeat(val, 3)] = 10000
    data[np.repeat(~(train | val), 3)] = -10000
    source["data/action"][:] = data
    # Identity labels are allowed in raw data but play no role in splitting.
    source.attrs.update(source_id="", episode_ids=["duplicate"], trial_ids={"bad": "group"})
    spec = {"action": "limits", "point_cloud": "limits"}
    dims = {"action": 19, "point_cloud": 3}
    states = []
    for horizon, before, after in ((1, 0, 0), (5, 2, 3)):
        dataset = DemonstrationDataset(
            str(path), sensor_modalities=["point_cloud"], obs_horizon=1, horizon=horizon,
            pad_before=before, pad_after=after, max_train_episodes=100, val_episodes=20, pc_dim=3,
            augmentation_cfg={"pc": {"coord_noise": {"noise_std": 100., "prob": 1.}}},
        )
        before_sample = source["data/point_cloud"][:]
        dataset[0]
        np.testing.assert_array_equal(source["data/point_cloud"][:], before_sample)
        normalizer = fit_normalizer(dataset, spec, feature_dims=dims)
        states.append(normalizer.state_dict())
        actual = normalizer["action"].normalize(data[selected_rows])
        torch.testing.assert_close(actual.amin(0), -torch.ones(19))
        torch.testing.assert_close(actual.amax(0), torch.ones(19))
        assert normalizer["point_cloud"].scale.shape == (3,)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    masks = []
    for seed in (7, 99):
        cfg = load_policy_config("dp3", [f"training.seed={seed}", f"zarr_path={path}",
                                         "horizon=4", "n_action_steps=2",
                                         "dataset.val_episodes=20", "dataset.augmentation_cfg=null"])
        masks.append(hydra.utils.instantiate(cfg.dataset, _convert_="all").train_mask)
    np.testing.assert_array_equal(*masks)


def check_affine_math():
    values = np.random.default_rng(7).normal(size=(7, 5, 3)).astype(np.float32)
    for mode in ("limits", "gaussian", "rms", "max_abs"):
        normalizer = LinearNormalizer()
        normalizer.fit_field("x", [values[:2], values[2:]], mode=mode, last_n_dims=2)
        field = normalizer["x"]
        reference = torch.from_numpy(values).double()
        if mode == "limits":
            scale = 2 / (reference.amax(0) - reference.amin(0))
            offset = -1 - scale * reference.amin(0)
        elif mode == "gaussian":
            scale = reference.std(0).reciprocal()
            offset = -reference.mean(0) * scale
        else:
            magnitude = reference.square().mean().sqrt() if mode == "rms" else reference.abs().max()
            scale = magnitude.reciprocal().expand(5, 3)
            offset = torch.zeros_like(scale)
        torch.testing.assert_close(field.scale, scale.float())
        torch.testing.assert_close(field.offset, offset.float())
        torch.testing.assert_close(field.unnormalize(field.normalize(values)), torch.from_numpy(values))
        for constant in (0., 3.):
            normalizer.fit_field("x", [np.full((4, 5, 3), constant, dtype=np.float32)], mode=mode, last_n_dims=2)
            field = normalizer["x"]
            assert field.scale.shape == field.offset.shape == (5, 3)
            assert torch.isfinite(field.scale).all() and (field.scale != 0).all()
            x = torch.full((4, 5, 3), constant)
            torch.testing.assert_close(field.unnormalize(field.normalize(x)), x)

    class IdentitySource:
        def iter_normalization_data(self, key):
            raise AssertionError("Identity scanned data")

    identity = fit_normalizer(IdentitySource(), {"action": "identity", "joint_state": "identity"},
                              feature_dims={"action": 19, "joint_state": 19})
    assert set(identity.fields) == {"action"}
    imported = {"fields.action.scale": torch.full((19,), 2.), "fields.action.offset": torch.ones(19)}
    inherited = fit_normalizer(IdentitySource(), {"action": "identity"}, feature_dims={"action": 19},
                               initial_state=imported)
    torch.testing.assert_close(inherited.state_dict(), imported, rtol=0, atol=0)

    class TaxelSource:
        def iter_normalization_data(self, key):
            yield np.arange(2 * 5 * 120 * 3, dtype=np.float32).reshape(2, 5, 120, 3)

    taxel = fit_normalizer(TaxelSource(), {"taxel_tactile_force": "rms"},
                           feature_dims={"taxel_tactile_force": 1800})["taxel_tactile_force"]
    assert taxel.scale.shape == (5, 120, 3) and taxel.scale.unique().numel() == 1
    assert (taxel.offset == 0).all()
    state = {"fields.joint_state.scale": torch.ones(19)}
    try:
        fit_normalizer(IdentitySource(), {"joint_state": "limits"}, feature_dims={"joint_state": 19},
                       initial_state=state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Partial imported affine accepted")


def check_method_targets(root):
    path = root / "targets.zarr"
    source = make_source(path, [2, 7, 13])
    source["data/finger_tactile_force"][:] = np.broadcast_to(np.arange(13, dtype=np.float32)[:, None, None], (13, 5, 3))
    kwargs = dict(sensor_modalities=["joint_state", "finger_tactile_force"],
                  horizon=5, obs_horizon=2, pad_before=1, pad_after=3)
    for cls, extra in ((AdapTacDataset, {}), (InterActDexDataset, {"interaction_horizon": 3})):
        dataset = cls(str(path), **kwargs, **extra)
        for index in range(len(dataset)):
            rows = dataset.sampler.sample_source_rows(index)
            query = rows[dataset.action_start_index]
            end = dataset.episode_store.episode_ends[np.searchsorted([2, 7, 13], query, side="right")]
            offsets = np.arange(1, 5) if cls is AdapTacDataset else np.array([3])
            target = query + offsets
            sample = dataset[index]
            np.testing.assert_array_equal(sample["future_force_valid"], target < end)
            np.testing.assert_array_equal(sample["future_force"][:, 0, 0], np.minimum(target, end - 1))
            if cls is AdapTacDataset:
                assert query + 1 < end
    r3d_dataset = R3DDataset(str(path), sensor_modalities=["joint_state"], obs_horizon=2,
                             horizon=5, pad_before=1, pad_after=3)
    for index in range(len(r3d_dataset)):
        rows = r3d_dataset.sampler.sample_source_rows(index)
        expected = np.concatenate((source["data/action"][:][rows], source["data/action_ee"][:][rows, :9]), -1)
        np.testing.assert_array_equal(r3d_dataset[index]["action"], expected)
    actual = np.concatenate(list(r3d_dataset.iter_normalization_data("action")))
    np.testing.assert_array_equal(actual, np.concatenate((source["data/action"][:], source["data/action_ee"][:, :9]), -1))


def check_resume_logging(root):
    for index, history in enumerate((b'{"step": 99}\n', b'broken\n{"step": 99}\n', b'broken', b'')):
        directory = root / f"logging-{index}"
        directory.mkdir()
        path = directory / "metrics.jsonl"
        path.write_bytes(history)
        logger = ExperimentLogger(directory, resume=True, wandb_enabled=False)
        try:
            logger.log({"loss": 1.}, step=2)
            logger.log({"loss": 0.5}, step=2)
        finally:
            logger.close()
        boundary = b"\n" if history and not history.endswith(b"\n") else b""
        prefix = history + boundary
        assert path.read_bytes().startswith(prefix)
        records = [json.loads(line) for line in path.read_bytes()[len(prefix):].splitlines()]
        assert [record["step"] for record in records] == [2, 2]
        assert list(directory.iterdir()) == [path]
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        directory = root / f"fresh-{name}"
        directory.mkdir()
        (directory / name).touch()
        try:
            ExperimentLogger(directory, wandb_enabled=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Fresh training overwrote an existing experiment")


def check_normalizer_restore(root, device):
    cfg = load_policy_config("dp3", ["agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16"])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    dims = dict(joint_state=19, point_cloud=3, action=19)
    model.configure_normalizer(dims, dict(joint_state=(19,), action=(19,), hand_latent=(4,)))
    raw = model.state_dict()
    raw["normalizer.fields.action.scale"].fill_(2)
    averaged = {key: value.clone() for key, value in raw.items()}
    averaged["normalizer.fields.action.scale"].fill_(3)
    for part in ("scale", "offset"):
        averaged[f"normalizer.fields.hand_latent.{part}"] = torch.ones(5)
    config = dict(agent=OmegaConf.to_container(cfg.agent, resolve=True), normalization_dims=dims)
    checkpoint = Checkpoint(7, raw, averaged, {}, {}, {}, config)
    path = root / "normalizer.pt"
    save_checkpoint(path, checkpoint)
    for use_ema, state in ((False, raw), (True, averaged)):
        restored = restore_policy(path, use_ema=use_ema, device=device)
        assert "point_cloud" not in restored.normalizer.fields
        assert restored.normalization_dims["point_cloud"] == 3
        assert tuple(restored.normalizer["hand_latent"].scale.shape) == ((5,) if use_ema else (4,))
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value.to(device), rtol=0, atol=0)
    for bad_key in ("scale", "offset", "shape"):
        state = dict(raw)
        if bad_key == "shape":
            state["normalizer.fields.action.offset"] = torch.zeros(18)
        else:
            del state[f"normalizer.fields.action.{bad_key}"]
        model.configure_normalizer(dims, normalizer_shapes_from_state(state))
        try:
            model.load_state_dict(state, strict=True, assign=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Strict loading accepted invalid affine {bad_key}")
    checkpoint.ema_model_state = None
    save_checkpoint(path, checkpoint)
    try:
        restore_policy(path, use_ema=True, device=device)
    except ValueError as error:
        assert "EMA" in str(error)
    else:
        raise AssertionError("Missing EMA silently selected raw weights")


def check_deployment_facts(root):
    path = root / "deployment.zarr"
    source = make_source(path, [3])
    source.create_dataset("data/depth", data=np.ones((3, 4, 5), dtype=np.uint16))
    source.create_dataset("data/fingertip_xyz", data=np.zeros((3, 5, 3), dtype=np.float32))
    cfg = load_policy_config("dp3", [f"zarr_path={path}", "horizon=2", "n_obs_steps=1", "n_action_steps=1",
        "dataset.sensor_modalities=[joint_state,point_cloud,hand_pointcloud,fingertip_xyz,depth]"])
    cfg.workspace.output_dir = str(root / "deployment-run")
    _, description = build_dataset(cfg)
    assert description["deployment_metadata"] == {"depth_config": {"shape": [4, 5], "depth_scale_m": None}}
    source.attrs.update(dt=0.02, pointcloud_config={"num_points": 8},
                        hand_pointcloud_config={"num_points": 8},
                        fingertip_link_names=["a", "b", "c", "d", "e"], depth_scale_m_per_unit=0.001)
    _, description = build_dataset(cfg)
    assert "format" not in source.attrs
    assert description["deployment_metadata"] == {
        "control_dt_s": 0.02, "pointcloud_config": {"num_points": 8},
        "hand_pointcloud_config": {"num_points": 8}, "fingertip_link_names": ["a", "b", "c", "d", "e"],
        "depth_config": {"shape": [4, 5], "depth_scale_m": 0.001},
    }
    model = SimpleNamespace(normalization_dims={"joint_state": 19, "action": 19},
                            consumed_observation_fields=("joint_state",),
                            inference_metadata={"action_start_index": 0, "control_action_dim": 19},
                            effective_config={})
    effective = assemble_effective_config(cfg, description, model)
    assert effective.deployment_metadata == {"control_dt_s": 0.02}
    assert not OmegaConf.is_readonly(effective)
    cfg.agent.action_dim = 3
    assert effective.agent.action_dim == 19


def check_inherited_policy(root, method, device=torch.device("cpu")):
    from dexmani_policy.agents.policies.afro import AFROEncoder
    from dexmani_policy.agents.policies.cordvip import CordViPEncoder

    path = root / f"{method}.zarr"
    source = make_source(path, [6, 12, 18])
    encoder = AFROEncoder(2).pc_encoder if method == "afro" else CordViPEncoder(2).interaction_encoder
    fields = ("point_cloud",) if method == "afro" else ("hand_pointcloud", "object_pointcloud", "joint_state")
    state = {} if method == "afro" else {
        "fields.hand_pointcloud.scale": torch.full((3,), 1.5),
        "fields.hand_pointcloud.offset": torch.full((3,), 2.),
        "fields.joint_state.scale": torch.full((19,), 2.),
        "fields.joint_state.offset": torch.full((19,), 3.),
    }
    modes = {key: "identity" for key in fields}
    if method == "cordvip":
        modes.update(hand_pointcloud="gaussian", joint_state="limits")
    artifact = root / f"{method}-encoder.pt"
    torch.save(dict(encoder_state=encoder.state_dict(), normalizer_state=state, normalization=modes), artifact)
    overrides = [
        f"zarr_path={path}", f"initialization.encoder={artifact}", f"workspace.output_dir={root / method}",
        "workspace.wandb_cfg=null", f"training.device={device}", "training.use_compile=false",
        "training.lr_warmup_steps=0", "training.loop.total_train_steps=8", "training.loop.log_interval_steps=1",
        "horizon=4", "n_obs_steps=2", "n_action_steps=2", "agent.down_dims=[16,32]",
        "agent.diffusion_step_embed_dim=16", "agent.num_inference_steps=2",
        "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
    ]
    if method == "cordvip":
        overrides += ["normalization.hand_pointcloud=identity"]
    else:
        overrides += ["~normalization.point_cloud"]
    extra = "object_pointcloud" if method == "afro" else "point_cloud"
    sensors = ("point_cloud", "joint_state", extra) if method == "afro" else (*fields, extra)
    overrides += [f"dataset.sensor_modalities=[{','.join(sensors)}]"]
    cfg = prepare_training(load_policy_config(method, overrides))
    if method == "afro":
        incomplete = load_policy_config(method, [*overrides, "~normalization.action"])
        with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Fit before required config")):
            try:
                run_training(incomplete, max_updates=1)
            except ValueError as error:
                assert "Missing normalization modes" in str(error) and "action" in str(error)
            else:
                raise AssertionError("Missing action normalization accepted")
    original = DemonstrationDataset.iter_normalization_data

    def only_new_fields(dataset, key):
        assert key not in (*fields, extra), f"Fitted inherited or unused field {key}"
        yield from original(dataset, key)

    with patch.object(DemonstrationDataset, "iter_normalization_data", only_new_fields):
        assert run_training(cfg, max_updates=2) == 2
    checkpoint_path = root / method / "checkpoints/latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert "normalizer_shapes" not in checkpoint.effective_config
    assert "resume_from" not in checkpoint.effective_config
    assert {key: checkpoint.effective_config["normalization"][key] for key in fields} == modes
    assert extra not in checkpoint.effective_config["normalization_dims"]
    for key, value in state.items():
        torch.testing.assert_close(checkpoint.model_state[f"normalizer.{key}"], value, rtol=0, atol=0)
    artifact.unlink()
    info = inspect_policy(root / method, checkpoint="latest", weights="ema" if cfg.training.use_ema else "raw")
    assert extra not in info.observation_fields
    runtime = load_policy(info, device=device)
    restored = runtime.agent
    observation = {key: source[f"data/{key}"][:2] for key in info.observation_fields}
    assert runtime.predict(observation).shape == (3, 19)
    validate_observation_fields(restored, [*restored.consumed_observation_fields, extra])
    assert restored._preprocess_obs({extra: torch.empty(1, 2, 8, 6, device=device)}) == {}
    resumed_cfg, checkpoint = prepare_resume(root / method, device=str(device))
    with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
        assert run_training(resumed_cfg, checkpoint=checkpoint, max_updates=1) == 3


def check_codebook_training(root, device=torch.device("cpu")):
    from dexmani_policy.agents.hand_prior.codebook import load_codebook

    path = root / "dqrise.zarr"
    source = make_source(path, [6, 12, 18])
    rng = np.random.default_rng(9)
    points = rng.uniform(-0.1, 0.1, (18, 32, 3)).astype(np.float32)
    points[..., 2] += 1
    source.create_dataset("data/point_cloud", data=points, overwrite=True)
    source.create_dataset("data/rgb", data=rng.integers(256, size=(18, 28, 28, 3), dtype=np.uint8))
    intrinsic = np.array([[20, 0, 13.5], [0, 20, 13.5], [0, 0, 1]], dtype=np.float32)
    source.create_dataset("data/camera_intrinsic", data=np.tile(intrinsic.reshape(1, 9), (18, 1)))
    source.create_dataset("data/camera_extrinsic", data=np.tile(np.eye(4, dtype=np.float32)[:3].reshape(1, 12), (18, 1)))
    artifact = root / "codebook.npz"
    scale = np.linspace(1, 2, 12, dtype=np.float32)
    offset = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    np.savez(artifact, sorted_hand_poses=rng.uniform(-1, 1, (3, 12)).astype(np.float32),
             hand_normalizer_scale=scale, hand_normalizer_offset=offset)
    cfg = load_policy_config("dqrise", [
        f"zarr_path={path}", f"initialization.codebook={artifact}", "initialization.rgb=null",
        f"workspace.output_dir={root / 'dqrise'}", "workspace.wandb_cfg=null",
        "horizon=4", "n_action_steps=2", "dataloader.batch_size=2", "dataloader.num_workers=0",
        "dataloader.pin_memory=false", f"training.device={device}", "training.use_compile=false",
        "training.lr_warmup_steps=0", "training.loop.total_train_steps=8", "training.loop.log_interval_steps=1",
    ])
    with open_dict(cfg.agent):
        cfg.agent.update(dict(
            num_points=32, n_layers=1, hidden_dim=32, n_head=4, num_inference_steps=2,
            pc_encoder_config=dict(channels=48, num_patches=8, group_size=4, depth=2, num_heads=4),
            fps_random_config=dict(use_random=False, use_random_start=False),
            rgb_backbone_config=dict(tune_mode="freeze", image_size=[28, 28], architecture=dict(
                model_type="dinov2", hidden_size=24, intermediate_size=48,
                num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14,
            )),
        ))
    prepare_training(cfg)
    with patch("dexmani_policy.agents.hand_prior.codebook.load_codebook", wraps=load_codebook) as load:
        assert run_training(cfg, max_updates=2) == 2
        assert load.call_count == 1
    checkpoint = load_checkpoint(root / "dqrise/checkpoints/latest.pt")
    assert checkpoint.effective_config["agent"]["num_codes"] == 3
    torch.testing.assert_close(checkpoint.model_state["normalizer.fields.action.scale"][-12:], torch.from_numpy(scale))
    torch.testing.assert_close(checkpoint.model_state["normalizer.fields.action.offset"][-12:], torch.from_numpy(offset))
    artifact.unlink()
    resumed_cfg, checkpoint = prepare_resume(root / "dqrise", device=str(device))
    with patch("dexmani_policy.agents.hand_prior.codebook.load_codebook", side_effect=AssertionError("Reload on resume")):
        assert run_training(resumed_cfg, checkpoint=checkpoint, max_updates=1) == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    check_splits()
    check_affine_math()
    for method in ("dp", "dp3", "r3d", "afro", "cordvip", "maniflow", "sat", "dqrise", "dexlatent", "adaptac", "interact_dex"):
        validate_config(load_policy_config(method))
    validate_config(load_policy_config("dp3", ["normalization.action=gaussian", "normalization.point_cloud=identity"]))
    print("PASS split counts, RNG isolation, affine math and YAML overrides", flush=True)
    with tempfile.TemporaryDirectory(prefix="dexmani-dataset-checks-") as directory:
        root = Path(directory)
        check_resume_logging(root)
        check_normalizer_restore(root, args.device)
        check_deployment_facts(root)
        print("PASS append-only logs, state-derived normalizer and deployment facts", flush=True)
        check_source_statistics(root)
        check_method_targets(root)
        print("PASS train-only source statistics, independent seed and method target rows", flush=True)
        for method in ("afro", "cordvip"):
            check_inherited_policy(root, method, args.device)
            print(f"PASS {method} inherited affine/identity and policy fresh/save/resume without source ({args.device})", flush=True)
        check_codebook_training(root, args.device)
        print(f"PASS DQ-RISE loads K once and resumes with saved table and hand affine ({args.device})", flush=True)


if __name__ == "__main__":
    main()
