"""Bounded DP RGB transport benchmark; run each variant in a fresh process.

Uses the current DP recipe with CPU augmentation, one GPU, and fixed loader knobs.
No checkpoint, W&B, simulation, or permanent Trainer synchronization. Compile is
opt-in here; report it separately from the production configuration default.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer, build_model_and_ema,
    build_optimizer_and_scheduler, compile_models, validate_config,
)
from dexmani_policy.training.resume import build_train_loader
from dexmani_policy.utils.random import set_seed
from dexmani_policy.utils.tensor import dict_apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', choices=['float32', 'uint8'], required=True)
    parser.add_argument('--batches', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batches <= 0 or args.warmup <= 0:
        raise ValueError('batches and warmup must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for RGB H2D benchmark')
    torch.set_num_threads(1)
    cfg = load_config('dp')
    cfg.dataset.rgb_keep_uint8 = args.transport == 'uint8'
    cfg.training.use_compile = args.compile
    # Same worker start/seed behavior in both variants; workers do CPU work only.
    with open_dict(cfg.dataloader):
        cfg.dataloader.multiprocessing_context = 'spawn'
    assert (cfg.dataloader.batch_size, cfg.dataloader.num_workers,
            cfg.dataloader.prefetch_factor, cfg.dataloader.pin_memory) == (64, 8, 2, True)
    validate_config(cfg)
    set_seed(cfg.training.seed)
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset)
    model, ema_model, ema_updater = build_model_and_ema(cfg, torch.device('cuda'), normalizer)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, len(loader))
    if args.compile:
        compile_models(model, ema_model)
    model.train()
    indices_hash = hashlib.sha256(np.asarray(list(loader.sampler), dtype='i8').tobytes()).hexdigest()
    iterator = iter(loader)
    waits, h2d_events, losses = [], [], []
    begin = None
    try:
        for step in range(args.warmup + args.batches):
            if step == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
            start_wait = time.perf_counter()
            batch = next(iterator)
            wait = time.perf_counter() - start_wait
            rgb = batch['obs']['rgb']
            dtype, payload = str(rgb.dtype), rgb.numel() * rgb.element_size()
            assert rgb.is_pinned()
            assert rgb.shape == (64, 2, 3, 224, 224)
            assert rgb.dtype == (torch.uint8 if cfg.dataset.rgb_keep_uint8 else torch.float32)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            batch = dict_apply(batch, lambda x: x.to('cuda', non_blocking=True))
            end.record()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.training.use_bfloat16):
                loss, _ = model(batch, **model.get_training_loss_kwargs(ema_model))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if ema_updater is not None:
                ema_updater.step(model)
            losses.append(loss.detach())
            if step >= args.warmup:
                waits.append(wait)
                h2d_events.append((start, end))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        peak_cuda = torch.cuda.max_memory_allocated() / 1024**2
        finite_loss = bool(torch.isfinite(torch.stack(losses)).all())
        if not finite_loss:
            raise FloatingPointError('non-finite warmup/measured loss')
        worker_hwm = []
        for worker in iterator._workers:
            status = Path(f'/proc/{worker.pid}/status').read_text()
            worker_hwm.append(int(next(line for line in status.splitlines()
                                      if line.startswith('VmHWM:')).split()[1]) / 1024)
    finally:
        iterator._shutdown_workers()

    # Isolate range-check synchronization after timing, on actual CUDA tensors.
    processor = model.obs_encoder.image_processor
    value = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
    checks = {}
    for validate in (True, False):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            processor.process_images(value, validate_float_range=validate)
        checks[str(validate)] = {event.key: event.count for event in profile.key_averages()
                                if event.key in ('aten::amin', 'aten::amax', 'aten::item', 'aten::_local_scalar_dense')}
    key = ('cuda', torch.cuda.current_device(), torch.float32)
    cached = processor._normalization_cache[key]
    expected = (value - processor.image_mean.cuda().view(1, 3, 1, 1)) / processor.image_std.cuda().view(1, 3, 1, 1)
    torch.testing.assert_close(processor.normalize(value), expected, rtol=0, atol=0)
    assert processor._normalization_cache[key] is cached
    record = {
        'transport': args.transport, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'data': dataset.zarr_path, 'windows': len(dataset), 'seed': cfg.training.seed,
        'loader': OmegaConf.to_container(cfg.dataloader, resolve=True),
        'dataset': OmegaConf.to_container(cfg.dataset, resolve=True),
        'agent': OmegaConf.to_container(cfg.agent, resolve=True),
        'bf16': cfg.training.use_bfloat16, 'compile': args.compile,
        'optimizer_scheduler_ema': True, 'indices_sha256': indices_hash,
        'warmup_batches': args.warmup, 'measured_batches': args.batches,
        'rgb_dtype': dtype, 'rgb_payload_bytes': payload, 'rgb_payload_mib': payload / 1024**2,
        'batch_h2d_ms_mean': float(np.mean([a.elapsed_time(b) for a, b in h2d_events])),
        'wait_ms_p50_p95': (np.quantile(waits, [.5, .95]) * 1000).tolist(),
        'samples_per_s': args.batches * 64 / elapsed, 'elapsed_s': elapsed,
        'parent_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'worker_peak_rss_mib': worker_hwm, 'peak_cuda_allocated_mib': peak_cuda,
        'finite_loss_all_batches': finite_loss,
        'range_check_cpu_ops_on_cuda_input': checks,
        'cuda_mean_std_cache_reused_and_equal': True,
    }
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
