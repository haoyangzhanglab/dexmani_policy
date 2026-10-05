"""Offline Dataset throughput and memory benchmark; run each variant separately.

Read the supplied Zarr store without modifying it; --prepare generates synthetic
input. Set PYTHONPATH to compare source checkouts on the same store and indices.
Opt-in --gpu-config measures short model forward/backward data wait, without
optimizer steps, checkpoint loading/writing or robot/sensor connections.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import resource
import time
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.agents.normalization import resolve_normalization_spec
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.training.build_utils import build_normalizer
from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.utils.tensor import dict_apply


def io_counts(pid="self"):
    return {
        key: int(value)
        for key, value in (
            line.split(":") for line in Path(f"/proc/{pid}/io").read_text().splitlines()
        )
    }


def profile_chunk_processing(metrics, worker_id=0):
    original = zarr.Array._process_chunk

    def process_chunk(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            metrics[worker_id * 2] += 1
            metrics[worker_id * 2 + 1] += time.perf_counter() - start

    zarr.Array._process_chunk = process_chunk


def measure_gpu_data_wait(args):
    """Measure short forward/backward data wait without optimizer, EMA or artifacts.

    finite_loss covers every executed batch, including warmup. Its aggregate
    check runs after timing ends and does not add per-batch host synchronization.
    """
    if args.order != "random":
        raise ValueError("GPU data-wait measurement requires --order random")
    register_resolvers()
    config = Path(args.gpu_config).resolve()
    with hydra.initialize_config_dir(config_dir=str(config.parent), version_base=None):
        cfg = hydra.compose(config_name=config.stem)
    cfg.dataset.zarr_path = str(Path(args.path).resolve())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    begin = time.perf_counter()
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = build_normalizer(
        dataset, resolve_normalization_spec(cfg), cfg.action_key
    )
    startup = time.perf_counter() - begin
    model = hydra.utils.instantiate(cfg.agent)
    model.initialize_training()
    model.load_normalizer_from_dataset(normalizer)
    model.set_normalization_spec(resolve_normalization_spec(cfg))
    model.action_key = cfg.action_key
    model = model.cuda().train()
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=order,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        **(
            {"multiprocessing_context": "spawn", "persistent_workers": True}
            if args.workers
            else {}
        ),
    )
    begin = time.perf_counter()
    iterator = iter(loader)
    waits, events, losses = [], [], []
    warmup = 3
    for step in range(warmup + args.batches):
        if step == warmup:
            torch.cuda.synchronize()
            warmup_s = time.perf_counter() - begin
            begin = time.perf_counter()
        wait = time.perf_counter()
        batch = next(iterator)
        wait = time.perf_counter() - wait
        model.zero_grad(set_to_none=True)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        batch = dict_apply(batch, lambda value: value.cuda(non_blocking=True))
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=cfg.training.use_bfloat16
        ):
            loss, _ = model.compute_loss(batch)
        loss.backward()
        end.record()
        losses.append(loss.detach())
        if step >= warmup:
            waits.append(wait)
            events.append((start, end))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    # Include warmup and every measured batch, outside the timing interval.
    finite_loss = bool(torch.isfinite(torch.stack(losses)).all())
    compute_ms = [start.elapsed_time(end) for start, end in events]
    gaps_ms = [
        events[i - 1][1].elapsed_time(events[i][0]) for i in range(1, len(events))
    ]
    worker_rss = []
    if args.workers:
        for worker in loader._iterator._workers:
            status = Path(f"/proc/{worker.pid}/status").read_text()
            worker_rss.append(
                int(
                    next(
                        line
                        for line in status.splitlines()
                        if line.startswith("VmHWM:")
                    ).split()[1]
                )
                / 1024
            )
        loader._iterator._shutdown_workers()
    print(
        json.dumps(
            {
                "config": str(config),
                "resolved_dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
                "gpu": torch.cuda.get_device_name(),
                "rows": int(dataset.replay_buffer.episode_ends[-1]),
                "windows": len(dataset),
                "batch_size": args.batch_size,
                "workers": args.workers,
                "measured_batches": len(waits),
                "warmup_batches": warmup,
                "dataset_normalizer_startup_s": startup,
                "worker_and_gpu_warmup_s": warmup_s,
                "samples_per_s": len(waits) * args.batch_size / elapsed,
                "batch_wait_ms_p50_p95_p99": (
                    np.quantile(waits, [0.5, 0.95, 0.99]) * 1000
                ).tolist(),
                "mean_gpu_h2d_forward_backward_ms": float(np.mean(compute_ms)),
                "gpu_between_steps_ms_p50_p95_p99": np.quantile(
                    gaps_ms, [0.5, 0.95, 0.99]
                ).tolist(),
                "parent_peak_rss_mib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024,
                "worker_peak_rss_mib": worker_rss,
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "action_scale_sum": float(
                    normalizer["action"].params_dict["scale"].sum()
                ),
                "indices_sha256": hashlib.sha256(
                    np.asarray(order, dtype="i8").tobytes()
                ).hexdigest(),
                "finite_loss": finite_loss,
                "compile": False,
                "optimizer": False,
                "ema": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--prepare", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--gpu-config", help="Optional short GPU workload using a Policy YAML"
    )
    parser.add_argument(
        "--order", choices=["sequential", "random"], default="sequential"
    )
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.gpu_config:
        if args.prepare:
            raise ValueError("--gpu-config cannot prepare or overwrite data")
        measure_gpu_data_wait(args)
        return
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
    order = (
        torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(args.seed)
        ).tolist()
        if args.order == "random"
        else list(range(len(dataset)))
    )
    metrics = mp.get_context("spawn").Array("d", 2 * max(1, args.workers), lock=False)
    if not args.workers:
        profile_chunk_processing(metrics)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=order,
        worker_init_fn=partial(profile_chunk_processing, metrics),
        num_workers=args.workers,
        **(
            {"multiprocessing_context": "spawn", "persistent_workers": True}
            if args.workers
            else {}
        ),
    )
    launch = time.perf_counter()
    iterator = iter(loader)
    next(iterator)  # exclude worker startup from steady throughput
    first_batch_s = time.perf_counter() - launch
    chunks_before = list(metrics)
    io_steady = io_counts()
    workers = loader._iterator._workers if args.workers else []
    worker_io_before = [io_counts(worker.pid) for worker in workers]
    waits = []
    start = time.perf_counter()
    samples = 0
    worker_peak_rss_mib = 0.0
    try:
        for _ in range(args.batches):
            try:
                begin = time.perf_counter()
                batch = next(iterator)
                waits.append(time.perf_counter() - begin)
            except StopIteration:
                break
            samples += len(batch["action"])
    finally:
        elapsed = time.perf_counter() - start
        chunks_after = list(metrics)
        io_after = io_counts()  # before waitpid: exclude reaped worker I/O from parent
        worker_io_after = [io_counts(worker.pid) for worker in workers]
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
    # Potential within-batch chunk reuse, independent of reader implementation.
    visits = unique = 0
    roles = {key: dataset.obs_horizon for key in dataset.sensor_modalities}
    roles[dataset.action_key] = dataset.horizon
    physical = zarr.open_group(args.path, mode="r")["data"]
    for offset in range(args.batch_size, args.batch_size + samples, args.batch_size):
        rows = dataset.sampler.source_rows(
            dataset.sampler.indices[order[offset : offset + args.batch_size]]
        )
        for key, length in roles.items():
            chunks = rows[:, :length] // physical[key].chunks[0]
            visits += sum(len(np.unique(row)) for row in chunks)
            unique += len(np.unique(chunks))
    print(
        json.dumps(
            {
                "rows": int(dataset.replay_buffer.episode_ends[-1]),
                "workers": args.workers,
                "order": args.order,
                "windows": len(dataset),
                "batch_size": args.batch_size,
                "measured_samples": samples,
                "seed": args.seed,
                "indices_sha256": hashlib.sha256(
                    np.asarray(order, dtype="i8").tobytes()
                ).hexdigest(),
                "first_batch_s": first_batch_s,
                "batch_wait_ms_p50_p95_p99": (
                    np.quantile(waits, [0.5, 0.95, 0.99]) * 1000
                ).tolist(),
                "within_batch_repeated_chunk_fraction": 1 - unique / visits,
                "steady_worker_io": [
                    {key: after[key] - before[key] for key in ("rchar", "read_bytes")}
                    for before, after in zip(worker_io_before, worker_io_after)
                ],
                "steady_chunk_process_calls": sum(chunks_after[::2])
                - sum(chunks_before[::2]),
                "steady_chunk_process_s": sum(chunks_after[1::2])
                - sum(chunks_before[1::2]),
                "steady_parent_rchar": io_after["rchar"] - io_steady["rchar"],
                "steady_parent_read_bytes": io_after["read_bytes"]
                - io_steady["read_bytes"],
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
