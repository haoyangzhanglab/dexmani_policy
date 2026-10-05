"""Small synthetic, offline memory/I/O benchmark; use separate processes per run.

Set PYTHONPATH to a source checkout to compare implementations on the same store.
No training, devices, existing experiments or checkpoints are involved.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.training.build_utils import build_normalizer


def io_counts():
    return {
        key: int(value)
        for key, value in (
            line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--prepare", type=int)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        root = zarr.open_group(args.path, mode="w-")
        count = args.prepare
        root.create_dataset("meta/episode_ends", data=np.arange(128, count + 1, 128))
        pc = root.create_dataset(
            "data/point_cloud", shape=(count, 1024, 6), chunks=(32, 1024, 6), dtype="f4"
        )
        action = root.create_dataset(
            "data/action_ee", shape=(count, 21), chunks=(32, 21), dtype="f4"
        )
        joint = root.create_dataset(
            "data/joint_state", shape=(count, 19), chunks=(32, 19), dtype="f4"
        )
        rng = np.random.default_rng(20261005)
        for start in range(0, count, 32):
            pc[start : start + 32] = rng.normal(size=(32, 1024, 6)).astype("f4")
            action[start : start + 32] = rng.normal(size=(32, 21)).astype("f4")
            joint[start : start + 32] = rng.normal(size=(32, 19)).astype("f4")
        return
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    io_before = io_counts()
    start = time.perf_counter()
    dataset = BaseDataset(
        args.path,
        sensor_modalities=["joint_state", "point_cloud"],
        action_key="action_ee",
        horizon=8,
        obs_horizon=2,
    )
    normalizer = build_normalizer(
        dataset,
        {"action": "auto", "joint_state": "limits", "point_cloud": "limits"},
        "action_ee",
    )
    startup = time.perf_counter() - start
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=args.workers,
        **(
            {"multiprocessing_context": "spawn", "persistent_workers": True}
            if args.workers
            else {}
        ),
    )
    iterator = iter(loader)
    next(iterator)  # exclude worker startup from steady throughput
    start = time.perf_counter()
    samples = 0
    worker_peak_rss_mib = 0.0
    try:
        for _ in range(16):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            samples += len(batch["action"])
    finally:
        elapsed = time.perf_counter() - start
        if args.workers and loader._iterator is not None:
            for worker in loader._iterator._workers:
                status = Path(f"/proc/{worker.pid}/status").read_text()
                hwm = next(
                    line for line in status.splitlines() if line.startswith("VmHWM:")
                )
                worker_peak_rss_mib = max(
                    worker_peak_rss_mib, int(hwm.split()[1]) / 1024
                )
            loader._iterator._shutdown_workers()
    io_after = io_counts()
    print(
        json.dumps(
            {
                "rows": int(dataset.replay_buffer.episode_ends[-1]),
                "workers": args.workers,
                "max_worker_peak_rss_mib": worker_peak_rss_mib,
                "startup_s": startup,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "rss_increase_mib": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
                )
                / 1024,
                "samples_per_s": samples / elapsed,
                "parent_rchar": io_after["rchar"] - io_before["rchar"],
                "parent_read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                "action_scale_sum": float(
                    normalizer["action"].params_dict["scale"].sum()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
