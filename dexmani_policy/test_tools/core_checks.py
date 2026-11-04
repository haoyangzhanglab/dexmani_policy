"""Check the DP3 data, training, resume and inference path using temporary synthetic data."""

import argparse
import shutil
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import zarr

from dexmani_policy.common.config import set_project_root
from dexmani_policy.deployment import inspect_policy, load_policy
from dexmani_policy.test_tools.smoke_test import run_smoke
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.data import build_dataset, fit_normalizer


def make_source(path):
    source = zarr.open_group(str(path), mode="w")
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

    normalizer = fit_normalizer(dataset, cfg.normalization, feature_dims=description["field_dims"])
    for key, values in raw.items():
        field = normalizer[key]
        normalized = field.normalize(values)
        flattened = normalized.reshape(-1, values.shape[-1])
        torch.testing.assert_close(flattened.amin(0), -torch.ones(values.shape[-1]))
        torch.testing.assert_close(flattened.amax(0), torch.ones(values.shape[-1]))
        torch.testing.assert_close(field.unnormalize(normalized), values)
    print("PASS dataset: windows, episode boundaries, padding and limits normalization", flush=True)
    return normalizer.state_dict()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    set_project_root()
    with tempfile.TemporaryDirectory(prefix="dexmani-core-") as temporary:
        root = Path(temporary)
        source = make_source(root / "source.zarr")
        cfg = load_policy_config("dp3", [
            f"zarr_path={root / 'source.zarr'}", f"training.device={args.device}",
            "horizon=4", "n_obs_steps=2", "n_action_steps=2",
            "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16",
            # Match the stored count so the optional FPS backend is unnecessary.
            "agent.num_points=8", "agent.num_inference_steps=2",
            "training.use_compile=false", "training.use_bfloat16=false",
            "training.lr_warmup_steps=0", "training.loop.total_train_steps=4",
            "training.loop.log_interval_steps=1", "dataset.augmentation_cfg=null",
            "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
        ])
        normalizer_state = check_dataset(cfg, source)
        run = root / "run"
        step, final = run_smoke(cfg, run, max_updates=2, resume_updates=2)
        print(f"PASS training/save/resume: fresh={step}, resumed={final}", flush=True)

        observation = {key: source[f"data/{key}"][:2] for key in cfg.dataset.sensor_modalities}
        # Deployment must use the checkpoint alone, including its fitted affine.
        shutil.rmtree(root / "source.zarr")
        info = inspect_policy(run)
        with closing(load_policy(info, device=args.device)) as runtime:
            torch.testing.assert_close(
                runtime.agent.normalizer.state_dict(), normalizer_state, rtol=0, atol=0, check_device=False,
            )
            prediction = runtime.predict(observation)
            assert prediction.shape == (3, 19)
        print(f"PASS deployment: saved normalization and finite actions {prediction.shape} without source data", flush=True)


if __name__ == "__main__":
    main()
