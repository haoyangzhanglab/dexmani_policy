"""Regression checks for tactile affine semantics and checkpoint boundaries."""

import argparse
import copy
import os
import shutil
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr

from dexmani_policy.common.checkpoint import _restore_policy, load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.config import TACTILE_FEATURE_SHAPES
from dexmani_policy.common.normalizer import normalizer_shapes_from_state
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state
from dexmani_policy.deployment import inspect_policy, load_policy
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.variants.tactile import read_future_force
from dexmani_policy.test_tools.dataset_checks import make_source
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.data import fit_normalizer
from dexmani_policy.training.common.resume import prepare_resume
from dexmani_policy.train import build_trainer, run_trainer


def check_math():
    for key, shape in TACTILE_FEATURE_SHAPES.items():
        x = torch.arange(7 * int(np.prod(shape)), dtype=torch.float32).reshape(7, *shape) / 100
        x[..., 1] = 4
        x[..., 2] = 0
        states = []
        for size in (1, 3, 7):
            class Source:
                def iter_normalization_data(self, field):
                    assert field == key
                    yield from x.split(size)
            field = fit_normalizer(Source(), {key: "limits"}, feature_dims={key: int(np.prod(shape))})[key]
            states.append(field.state_dict())
            z = field.normalize(x)
            assert z[..., 0].min() == -1 and z[..., 0].max() == 1
            assert (field.scale[..., 1:] == 1).all()
            assert (field.offset[..., 1] == -4).all() and (z[..., 1:] == 0).all()
            torch.testing.assert_close(field.unnormalize(z), x)
            assert field.normalize(x * 10)[..., 0].max() > 1
            axes = field.scale.reshape(-1, 3)
            torch.testing.assert_close(axes, axes[:1].expand_as(axes), rtol=0, atol=0)
        for fill, span in ((0., 0.), (3., 0.), (3., 1e-5)):
            x.fill_(fill)
            x[-1, ..., 0] += span
            field = fit_normalizer(Source(), {key: "limits"}, feature_dims={key: int(np.prod(shape))})[key]
            assert (field.scale == 1).all()
            torch.testing.assert_close(field.offset, torch.full(shape, -fill))
            torch.testing.assert_close(field.normalize(x), x - fill)
        for state in states[1:]:
            torch.testing.assert_close(state, states[0], rtol=0, atol=0)


def check_affine_validation():
    for key, shape in TACTILE_FEATURE_SHAPES.items():
        prefix = f"normalizer.fields.{key}."
        for mode in ("limits", "rms", "max_abs"):
            scale = torch.tensor([.5, 1., 2.]) if mode == "limits" else torch.full((3,), .5)
            offset = torch.tensor([-1., .25, -.5]) if mode == "limits" else torch.zeros(3)
            state = {prefix + "scale": scale.expand(shape).clone(), prefix + "offset": offset.expand(shape).clone()}
            state[prefix + "scale"].reshape(-1, 3)[0, 0] *= 2
            if mode == "limits":
                state[prefix + "offset"].reshape(-1, 3)[0, 0] += 1
            spec, dims = {key: mode}, {key: int(np.prod(shape))}
            assert normalizer_shapes_from_state(state, normalization=spec, feature_dims=dims) == {key: shape}
            damages = ["missing", "scale", "offset", "shape", "nan", "inf", "zero", "negative",
                       "offset_nan", "offset_inf", "mode", "invalid_mode"]
            if mode != "limits":
                damages.append("nonzero_offset")
            for damage in damages:
                broken = {name: value.clone() for name, value in state.items()}
                modes = dict(spec)
                if damage == "missing":
                    broken.clear()
                elif damage in ("scale", "offset"):
                    del broken[prefix + damage]
                elif damage == "shape":
                    broken[prefix + "offset"] = torch.zeros(3)
                elif damage == "mode":
                    modes.clear()
                elif damage == "invalid_mode":
                    modes[key] = "identity"
                elif damage == "nonzero_offset":
                    broken[prefix + "offset"].reshape(-1, 3)[0, 0] += 1
                elif damage in ("offset_nan", "offset_inf"):
                    broken[prefix + "offset"].reshape(-1, 3)[0, 0] = float(damage.removeprefix("offset_"))
                else:
                    value = {"nan": float("nan"), "inf": float("inf"), "zero": 0., "negative": -1.}[damage]
                    broken[prefix + "scale"].reshape(-1, 3)[0, 0] = value
                try:
                    normalizer_shapes_from_state(broken, normalization=modes, feature_dims=dims)
                except ValueError as error:
                    assert key in str(error)
                else:
                    raise AssertionError(f"Accepted {key} {mode} {damage}")


def check_unshared_affine_restore(saved, directory, device):
    checkpoint = copy.deepcopy(saved)
    for index, state in enumerate((checkpoint.model_state, checkpoint.ema_model_state)):
        for key in TACTILE_FEATURE_SHAPES.keys() & checkpoint.effective_config["normalization"].keys():
            shape = TACTILE_FEATURE_SHAPES[key]
            prefix = f"normalizer.fields.{key}."
            values = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape) / 1000
            state[prefix + "scale"] = values + .5 + index
            if checkpoint.effective_config["normalization"][key] == "limits":
                state[prefix + "offset"] = values - .25 - index
    path = save_checkpoint(directory / "checkpoints/latest.pt", checkpoint)
    for ema, expected in ((False, checkpoint.model_state), (True, checkpoint.ema_model_state)):
        restored = restore_policy(path, use_ema=ema, device=device)
        for name, value in restored.normalizer.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[f"normalizer.{name}"], rtol=0, atol=0)
    with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")), \
         closing(prepare_resume(directory, device=str(device))) as resumed:
        for model, expected in ((resumed.model, checkpoint.model_state), (resumed.ema_model, checkpoint.ema_model_state)):
            for name, value in model.normalizer.state_dict().items():
                torch.testing.assert_close(value.cpu(), expected[f"normalizer.{name}"], rtol=0, atol=0)


def check_force_targets(model, obs):
    field = model.normalizer["finger_tactile_force"]
    original = model._preprocess_obs(obs)["finger_tactile_force"]
    current = obs["finger_tactile_force"][:, -1:]
    # A nonzero offset exposes accidental affine application to a delta.
    expected = torch.tensor([0., 2., -2.], device=current.device).expand_as(current)
    for pose in (obs["eef_pose"], torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 1.],
                                             device=current.device).expand_as(obs["eef_pose"])):
        changed = {**obs, "eef_pose": pose}
        torch.testing.assert_close(model._preprocess_obs(changed)["finger_tactile_force"], original)
        for target in (torch.zeros_like(expected), expected):
            future = current + target / field.scale
            batch = dict(obs=changed, future_force=future,
                         future_force_valid=torch.ones(current.shape[:2], dtype=torch.bool, device=current.device))
            loss, _ = model._compute_future_force_loss(target, batch)
            torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-11, rtol=0)
            # One invalid row must not dilute the valid loss or propagate NaN labels.
            batch = dict(obs={key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in changed.items()},
                         future_force=torch.cat((future, torch.full_like(future, float("nan")))),
                         future_force_valid=torch.tensor([[True], [False]], device=current.device))
            prediction = torch.cat((target + .5, torch.full_like(target, 100.))).requires_grad_()
            loss, _ = model._compute_future_force_loss(prediction, batch)
            torch.testing.assert_close(loss, torch.full_like(loss, .125))
            batch["future_force_valid"].zero_()
            batch["future_force"].fill_(float("nan"))
            loss, mask = model._compute_future_force_loss(prediction, batch)
            loss.backward()
            assert loss == 0 and not mask.any() and torch.count_nonzero(prediction.grad) == 0


def config(method, source, output, mode="limits", kind="force", device="cpu", *, future_force_loss_weight=0.1):
    overrides = [f"zarr_path={source}", f"workspace.output_dir={output}", "workspace.wandb_cfg=null",
                 f"training.device={device}", "training.use_compile=false", "training.lr_warmup_steps=0",
                 "training.loop.total_train_steps=4", "training.loop.log_interval_steps=10",
                 "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
                 "dataset.augmentation_cfg=null",
                 "agent.num_points=16", "agent.pc_encoder_config.channels=24",
                 "agent.pc_encoder_config.num_patches=12", "agent.pc_encoder_config.group_size=4",
                 "agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_heads=4"]
    if method == "adaptac":
        overrides += [f"tactile={kind}", "agent.feature_dim=16", "agent.state_out_dim=8",
                      "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=32",
                      "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=16"]
        key = "finger_tactile_force" if kind == "force" else "taxel_tactile_force"
        overrides += [f"normalization.{key}={mode}"]
    else:
        overrides += ["horizon=4", "n_action_steps=2", "agent.num_inference_steps=2",
                      f"normalization.finger_tactile_force={mode}", "n_obs_steps=2",
                      "future_force_horizons=[1]", f"agent.future_force_loss_weight={future_force_loss_weight}",
                      "agent.n_layers=1",
                      "agent.hidden_dim=32", "agent.n_head=4", "agent.timestep_embed_dim=32"]
    return load_policy_config(method, overrides)


def run_reproducible_training(trainer, *, max_updates):
    # set_seed enables cuDNN benchmarking; fix algorithms for the resume comparison.
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        return run_trainer(trainer, max_updates=max_updates)


def check_training(root, device="cpu"):
    source = root / "source.zarr"
    group = make_source(source, [8, 17, 26])
    rng = np.random.default_rng(5)
    del group["data/point_cloud"]
    group.create_dataset("data/point_cloud", data=rng.normal(size=(26, 16, 6)).astype("f4"))
    force = rng.normal(size=(26, 5, 3)).astype("f4") * [1, 2, 4] + [2, -3, 1]
    group["data/finger_tactile_force"][:] = force
    dense = rng.normal(size=(26, 5, 120, 3)).astype("f4") + 2
    group.create_dataset("data/taxel_tactile_force", data=dense, chunks=(3, 5, 120, 3))
    # Vary pose independently of the raw force to catch accidental coordinate transforms.
    rotation = torch.tensor([
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
        [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
    ]).repeat(9, 1, 1)[:26]
    pose = np.zeros((26, 9), dtype="f4")
    pose[:, 3:6], pose[:, 6:9] = rotation[:, :, 0].numpy(), rotation[:, :, 1].numpy()
    group.create_dataset("data/eef_pose", data=pose, chunks=(5, 9))
    group.create_dataset("data/fingertip_xyz", data=rng.normal(size=(26, 5, 3)).astype("f4"))
    # Holdout/unused rows and window padding must not affect either affine.
    states = []
    for horizon, padding in ((1, 0), (4, 2)):
        dataset = DemonstrationDataset(source, sensor_modalities=["finger_tactile_force", "taxel_tactile_force", "eef_pose"],
                                       obs_horizon=1, horizon=horizon, pad_before=padding, pad_after=padding,
                                       max_train_episodes=1, val_episodes=1)
        spec = dict(finger_tactile_force="limits", taxel_tactile_force="limits")
        affine = fit_normalizer(dataset, spec, feature_dims=dict(finger_tactile_force=15, taxel_tactile_force=1800))
        states.append(affine.state_dict())
        rows = np.repeat(dataset.train_mask, np.diff(np.r_[0, dataset.episode_store.episode_ends]))
        for key in spec:
            values = torch.tensor(group[f"data/{key}"][:][rows])
            z = affine[key].normalize(values).reshape(-1, 3)
            torch.testing.assert_close(z.amin(0), -torch.ones(3))
            torch.testing.assert_close(z.amax(0), torch.ones(3))
            low, high = values.reshape(-1, 3).amin(0), values.reshape(-1, 3).amax(0)
            expected_scale = 2 / (high - low)
            torch.testing.assert_close(affine[key].scale, expected_scale.expand_as(affine[key].scale))
            torch.testing.assert_close(affine[key].offset, (-1 - low * expected_scale).expand_as(affine[key].offset))
        originals = {key: group[f"data/{key}"][:] for key in spec}
        for key, values in originals.items():
            outliers = values.copy()
            outliers[~rows] = 1e6
            group[f"data/{key}"][:] = outliers
        with patch.object(dataset.episode_store, "read", wraps=dataset.episode_store.read) as read:
            refit = fit_normalizer(dataset, spec, feature_dims=dict(finger_tactile_force=15, taxel_tactile_force=1800))
        assert all(call.args[0] != "eef_pose" for call in read.call_args_list)
        torch.testing.assert_close(refit.state_dict(), affine.state_dict(), rtol=0, atol=0)
        for key, values in originals.items():
            group[f"data/{key}"][:] = values
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    cases = [(method, kind, mode, 0.1)
             for method, kind in (("interact_dex", "force"), ("adaptac", "force"), ("adaptac", "taxel"), ("adaptac", "patch"))
             for mode in ("limits", "rms", "max_abs")]
    cases.append(("interact_dex", "force", "limits", 0.0))
    for method, kind, mode, weight in cases:
        case = f"{method}-{kind}-{mode}-{weight}"
        key = "finger_tactile_force" if kind == "force" else "taxel_tactile_force"
        unused = "taxel_tactile_force" if kind == "force" else "finger_tactile_force"
        selected_source = root / f"source-{case}.zarr"
        shutil.copytree(source, selected_source)
        selected_group = zarr.open_group(str(selected_source), mode="a")
        if mode != "rms":
            del selected_group[f"data/{unused}"]
        read_array = zarr.Array.__getitem__

        def selected_read(array, selection, *args, **kwargs):
            if array.path == f"data/{unused}":
                raise AssertionError(f"Read unselected tactile field {unused}")
            return read_array(array, selection, *args, **kwargs)

        with patch.object(zarr.Array, "__getitem__", selected_read), \
             patch("dexmani_policy.datasets.variants.tactile.read_future_force", wraps=read_future_force) as future_read:
            cfg = config(method, selected_source, root / case, mode, kind, device,
                         future_force_loss_weight=weight)
            trainer = build_trainer(cfg)
            initial_rng = get_rng_state(device)
            model = trainer.model
            field = model.normalizer[key]
            obs = {k: torch.tensor(selected_group[f"data/{k}"][:cfg.n_obs_steps], device=device)[None]
                   for k in cfg.dataset.sensor_modalities}
            original = {k: v.clone() for k, v in obs.items()}
            processed = model._preprocess_obs(obs)
            raw = obs[key]
            expected_input = raw * field.scale if kind == "patch" else field.normalize(raw)
            torch.testing.assert_close(processed[key], expected_input.flatten(0, 1))
            torch.testing.assert_close(obs, original, rtol=0, atol=0)
            if method == "interact_dex":
                if mode == "limits":
                    assert field.offset.abs().sum() > 0
                    z = field.normalize(torch.tensor(force, dtype=torch.float32, device=device))
                    torch.testing.assert_close(z.amin((0, 1)), -torch.ones(3, device=device))
                    torch.testing.assert_close(z.amax((0, 1)), torch.ones(3, device=device))
                if weight > 0:
                    check_force_targets(model, obs)
                else:
                    assert model.future_force_head is None
                    assert type(trainer.train_loader.dataset) is DemonstrationDataset
                    batch = dict_apply(next(iter(trainer.train_loader)), lambda x: x.to(device))
                    assert "future_force" not in batch and "future_force_valid" not in batch
                    _, metrics = model.compute_loss(batch)
                    assert not any("future_force" in name for name in metrics)
            else:
                encoder = model.obs_encoder.tactile_encoder
                for value in ((0., 2.) if kind == "patch" else ()):
                    obs["taxel_tactile_force"] = torch.full_like(obs["taxel_tactile_force"], value, requires_grad=True)
                    u = model._preprocess_obs(obs)["taxel_tactile_force"]
                    expected = obs["taxel_tactile_force"] * model.normalizer["taxel_tactile_force"].scale
                    torch.testing.assert_close(u, expected.flatten(0, 1), rtol=0, atol=0)
                    desc = encoder._patch_descriptors(u)
                    assert bool(desc.abs().sum() > 0) == bool(value)
                    desc.sum().backward()
                    assert torch.isfinite(obs["taxel_tactile_force"].grad).all()
                batch = dict_apply(next(iter(trainer.train_loader)), lambda x: x.to(device))
                with patch.object(model.force_predictor, "compute_loss_and_prediction", wraps=model.force_predictor.compute_loss_and_prediction) as call, \
                     patch.object(model, "_action_condition", wraps=model._action_condition) as condition:
                    model.compute_loss(batch)
                torch.testing.assert_close(call.call_args.args[1], field.normalize(batch["future_force"]).flatten(start_dim=2))
                torch.testing.assert_close(condition.call_args.args[1], field.normalize(batch["obs"][key]).flatten(start_dim=2))
            set_rng_state(initial_rng, device)
            assert run_reproducible_training(trainer, max_updates=2) == 2
            checkpoint = Path(cfg.workspace.output_dir) / "checkpoints/latest.pt"
            saved = load_checkpoint(checkpoint)
            assert set(saved.effective_config["dataset"]["sensor_modalities"]) & TACTILE_FEATURE_SHAPES.keys() == {key}
            assert set(saved.effective_config["normalization"]) & TACTILE_FEATURE_SHAPES.keys() == {key}
            for ema in (False, True):
                restored = restore_policy(checkpoint, use_ema=ema, device=device)
                torch.testing.assert_close(restored.normalizer.state_dict(), model.normalizer.state_dict(), rtol=0, atol=0)
                assert set(restored.normalization_dims) & TACTILE_FEATURE_SHAPES.keys() == {key}
                assert set(restored.normalizer.fields) & TACTILE_FEATURE_SHAPES.keys() == {key}
                info = inspect_policy(cfg.workspace.output_dir, weights="ema" if ema else "raw")
                assert set(info.observation_fields) & TACTILE_FEATURE_SHAPES.keys() == {key}
                runtime = load_policy(info, device=device)
                history = {name: value[0].detach().cpu().numpy() for name, value in original.items()}
                result = runtime.predict(history)
                assert result.shape == (cfg.horizon - cfg.n_obs_steps + 1, 19)
                assert np.isfinite(result).all()
                try:
                    runtime.predict({name: value for name, value in history.items() if name != key})
                except KeyError as error:
                    assert error.args == (key,)
                else:
                    raise AssertionError("Accepted missing selected tactile history")
                runtime.close()
                if method == "interact_dex" and mode == "limits":
                    output = restored.predict_action(obs)["future_actions"]
                    assert output.shape == (1, cfg.horizon - cfg.n_obs_steps + 1, 19)
                    assert torch.isfinite(output).all()
                    torch.testing.assert_close(restored.normalizer.state_dict(), model.normalizer.state_dict(), rtol=0, atol=0)
            with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
                resumed = prepare_resume(cfg.workspace.output_dir, device=str(device))
            assert run_reproducible_training(resumed, max_updates=1) == 3
            continuous_cfg = config(method, selected_source, root / f"continuous-{case}", mode, kind, device,
                                    future_force_loss_weight=weight)
            continuous = build_trainer(continuous_cfg)
            assert run_reproducible_training(continuous, max_updates=3) == 3
            for expected, actual in ((continuous.model, resumed.model), (continuous.ema_model, resumed.ema_model)):
                expected_state, actual_state = expected.state_dict(), actual.state_dict()
                torch.testing.assert_close(expected_state, actual_state, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(expected.normalizer.state_dict(), actual.normalizer.state_dict(), rtol=0, atol=0)
            for key in TACTILE_FEATURE_SHAPES.keys() & model.normalization_dims.keys():
                prefix = f"normalizer.fields.{key}."
                broken = {name: value for name, value in saved.model_state.items() if not name.startswith(prefix)}
                try:
                    _restore_policy(saved.effective_config, broken, device)
                except ValueError as error:
                    assert key in str(error) and "Missing required tactile affine" in str(error)
                else:
                    raise AssertionError(f"Restored policy without required {key} affine")
            if method == "adaptac":
                check_unshared_affine_restore(saved, root / f"unshared-{kind}-{mode}", device)
            if method == "interact_dex" and weight == 0:
                future_read.assert_not_called()
                assert not any("future_force_head" in name for name in saved.model_state)
            print(f"PASS tactile train/save/raw/EMA/resume: {case}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if torch.device(args.device).type == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    check_math()
    check_affine_validation()
    with tempfile.TemporaryDirectory(prefix="dexmani-tactile-") as temporary:
        check_training(Path(temporary), args.device)
    print("PASS tactile normalization")
