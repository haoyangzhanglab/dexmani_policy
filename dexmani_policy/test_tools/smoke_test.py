"""Check the synthetic DP3/CordViP lifecycle, or train and resume a supplied recipe."""

import argparse
import shutil
import tempfile
from contextlib import closing, nullcontext
from pathlib import Path

import numpy as np
import torch
import zarr

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int, set_project_root
from dexmani_policy.common.normalizer import LinearNormalizer, normalization_feature_shapes
from dexmani_policy.deployment import load_policy
from dexmani_policy.train import run_training
from dexmani_policy.training.common.config import load_config
from dexmani_policy.training.common.data import build_dataset, fit_normalizer
from dexmani_policy.training.common.resume import prepare_resume


def run_smoke(cfg, directory: Path, *, max_updates: int, resume_updates: int, resume_zarr_path=None) -> tuple[int, int]:
    cfg.workspace.output_dir = str(directory)
    cfg.workspace.wandb_cfg = None
    if cfg.training.loop.total_train_steps <= max_updates:
        raise ValueError("The saved training budget must leave updates for resume; reduce --max-updates")
    step = run_training(cfg, max_updates=max_updates)
    assert step == max_updates, "Fresh update count mismatch"
    trainer = prepare_resume(directory, zarr_path=resume_zarr_path)
    assert trainer is not None
    with closing(trainer):
        assert trainer.global_step == step, "Resume source mismatch"
        if resume_zarr_path is not None:
            assert trainer.cfg.dataset.zarr_path == str(Path(resume_zarr_path).expanduser().resolve())
        checkpoint = load_checkpoint(directory / "checkpoints" / "latest.pt")
        torch.testing.assert_close(trainer.model.state_dict(), checkpoint.model_state, check_device=False)
        if trainer.ema_model is not None:
            torch.testing.assert_close(trainer.ema_model.state_dict(), checkpoint.ema_model_state, check_device=False)
        optimizer_state = trainer.optimizer.state_dict()
        torch.testing.assert_close(optimizer_state["state"], checkpoint.optimizer_state["state"], check_device=False)
        assert optimizer_state["param_groups"] == checkpoint.optimizer_state["param_groups"]
        assert trainer.scheduler.state_dict() == checkpoint.scheduler_state
        assert trainer.cfg.control_dt_s == checkpoint.effective_config["control_dt_s"]
        data_dt = float(zarr.open_group(cfg.dataset.zarr_path, mode="r").attrs["dt"])
        assert checkpoint.effective_config["control_dt_s"] == data_dt
        del checkpoint, optimizer_state
        trainer.compile_models()
        trainer.train(max_updates=resume_updates)
        final = trainer.global_step
    expected = min(step + resume_updates, cfg.training.loop.total_train_steps)
    assert final == expected, "Resume update count mismatch"
    saved = load_checkpoint(directory / "checkpoints" / "latest.pt")
    assert saved.global_step == final, "Resumed checkpoint mismatch"
    assert saved.effective_config["control_dt_s"] == data_dt
    return step, final


def make_source(path):
    source = zarr.open_group(str(path), mode="w-")
    source.attrs.update(task_name="pick_apple_messy", dt=0.05, action_space="joint")
    source.create_dataset("meta/episode_ends", data=np.array([6, 12], dtype=np.int64))
    rng = np.random.default_rng(11)
    for key, shape in (
        ("action", (12, 19)), ("joint_state", (12, 19)), ("point_cloud", (12, 8, 6)),
    ):
        source.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype(np.float32))
    return source


def check_dataset(cfg, source):
    dataset, description = build_dataset(cfg)
    assert len(dataset) == 10
    raw = {key: torch.from_numpy(source[f"data/{key}"][:]) for key in description["field_dims"]}
    for index, rows in (
        (0, [0, 0, 1, 2]), (4, [3, 4, 5, 5]),
        (5, [6, 6, 7, 8]), (9, [9, 10, 11, 11]),
    ):
        sample = dataset[index]
        torch.testing.assert_close(sample["action"], raw["action"][rows], rtol=0, atol=0)
        for key in ("joint_state", "point_cloud"):
            torch.testing.assert_close(sample["obs"][key], raw[key][rows[:2]], rtol=0, atol=0)

    normalizer = LinearNormalizer(normalization_feature_shapes(description["field_dims"], cfg.normalization))
    fit_normalizer(dataset, normalizer, normalizer.fields)
    assert set(normalizer.fields) == {key for key in description["field_dims"] if cfg.normalization[key] == "limits"}
    for key, field in normalizer.fields.items():
        values = raw[key]
        normalized = field.normalize(values)
        flattened = normalized.reshape(-1, values.shape[-1])
        torch.testing.assert_close(flattened.amin(0), -torch.ones(values.shape[-1]))
        torch.testing.assert_close(flattened.amax(0), torch.ones(values.shape[-1]))
        torch.testing.assert_close(field.unnormalize(normalized), values)
    print("PASS dataset: windows, episode boundaries, padding and limits normalization", flush=True)
    return normalizer.state_dict()


def check_synthetic(root, device, *, max_updates, resume_updates):
    source = make_source(root / "source.zarr")
    replacement_path = root / "replacement.zarr"
    replacement = make_source(replacement_path)
    replacement.create_dataset("meta/episode_ends", data=np.array([3, 6, 9, 12]), overwrite=True)
    replacement["data/action"][:] += 1
    cfg = load_config("dp3", [
        f"zarr_path={root / 'source.zarr'}", f"training.device={device}",
        "horizon=4", "n_obs_steps=2", "n_action_steps=2",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16",
        # Match the stored count so the optional FPS backend is unnecessary.
        "agent.num_points=8", "agent.num_inference_steps=2",
        "training.use_compile=false", "training.use_bfloat16=false",
        "training.lr_warmup_steps=0", f"training.loop.total_train_steps={max_updates + resume_updates}",
        "training.loop.log_interval_steps=1", "dataset.augmentation_cfg=null",
        "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
    ])
    normalizer_state = check_dataset(cfg, source)
    run = root / "run"
    step, final = run_smoke(cfg, run, max_updates=max_updates, resume_updates=resume_updates,
                           resume_zarr_path=replacement_path)
    print(f"PASS training/save/resume: fresh={step}, resumed={final}", flush=True)

    observation = {key: source[f"data/{key}"][:2] for key in cfg.dataset.sensor_modalities}
    # Deployment must use the checkpoint alone, including its fitted affine.
    shutil.rmtree(root / "source.zarr")
    shutil.rmtree(replacement_path)
    with closing(load_policy(run, device=device)) as runtime:
        assert isinstance(runtime.info.checkpoint_path, str)
        assert not runtime.info.requires_prev_action
        assert runtime.info.control_dt_s == 0.05
        assert runtime.info.observation_fields == tuple(cfg.dataset.sensor_modalities)
        torch.testing.assert_close(
            runtime.agent.normalizer.state_dict(), normalizer_state, rtol=0, atol=0, check_device=False,
        )
        prediction = runtime.predict(observation)
        assert prediction.shape == (3, 19)
    print(f"PASS deployment: saved normalization and finite actions {prediction.shape} without source data", flush=True)


def check_cordvip(root, device, *, max_updates, resume_updates):
    root.mkdir()
    source = make_source(root / "source.zarr")
    rng = np.random.default_rng(17)
    for key, count in (("hand_pointcloud", 8), ("object_pointcloud", 12)):
        source.create_dataset(f"data/{key}", data=rng.normal(size=(12, count, 3)))
    replacement = root / "replacement.zarr"
    shutil.copytree(root / "source.zarr", replacement)
    replaced = zarr.open_group(str(replacement), mode="a")
    replaced["data/action"][:] += 1
    replaced["meta/episode_ends"][:] = [4, 12]
    cfg = load_config("cordvip", [
        f"zarr_path={root / 'source.zarr'}", f"training.device={device}",
        "horizon=4", "n_obs_steps=2", "n_action_steps=2",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16",
        "agent.num_inference_steps=2", "training.use_compile=false",
        "training.use_bfloat16=false", "training.lr_warmup_steps=0",
        f"training.loop.total_train_steps={max_updates + resume_updates}",
        "training.loop.log_interval_steps=1", "dataloader.batch_size=2",
        "dataloader.num_workers=0", "dataloader.pin_memory=false",
    ])
    run_smoke(cfg, root / "run", max_updates=max_updates, resume_updates=resume_updates,
              resume_zarr_path=replacement)
    observation = {key: source[f"data/{key}"][:2] for key in cfg.dataset.sensor_modalities}
    with closing(load_policy(root / "run", device=device)) as runtime:
        assert runtime.info.control_dt_s == .05
        prediction = runtime.predict(observation)
        assert prediction.shape == (3, 19) and np.isfinite(prediction).all()
    print("PASS CordViP: float64 variable point counts, resume and finite 19D actions")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Recipe name; omit for synthetic DP3/CordViP checks")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override (requires config)")
    parser.add_argument("--device", help="Defaults to CPU for synthetic checks, or the recipe device")
    parser.add_argument("--output-dir", type=Path, help="Keep test artifacts here instead of a temporary directory")
    parser.add_argument("--max-updates", type=int, help="Fresh updates: 2 for synthetic checks, 4 for a recipe")
    parser.add_argument("--resume-updates", type=int, default=2)
    args = parser.parse_args(argv)
    if args.config is None and args.override:
        parser.error("--override requires a config name")
    max_updates = args.max_updates if args.max_updates is not None else (4 if args.config else 2)
    positive_int(max_updates, "max_updates")
    positive_int(args.resume_updates, "resume_updates")
    set_project_root()
    output = nullcontext(args.output_dir) if args.output_dir is not None else tempfile.TemporaryDirectory(prefix="dexmani-smoke-")
    with output as directory:
        directory = Path(directory).expanduser().resolve()
        if args.config is None:
            check_synthetic(directory, args.device or "cpu",
                            max_updates=max_updates, resume_updates=args.resume_updates)
            check_cordvip(directory / "cordvip", args.device or "cpu",
                          max_updates=max_updates, resume_updates=args.resume_updates)
        else:
            cfg = load_config(args.config, args.override)
            if args.device is not None:
                cfg.training.device = args.device
            step, final = run_smoke(cfg, directory, max_updates=max_updates, resume_updates=args.resume_updates)
            print(f"PASS {args.config}: fresh={step}, resumed={final}; run={directory}", flush=True)


if __name__ == "__main__":
    main()
