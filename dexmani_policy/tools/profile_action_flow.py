"""Segment-level latency / memory profiler for the ActionFlow policy.

Measures training (forward / backward / optimizer+EMA step) and inference
(predict_action) latency using ``torch.cuda.Event`` — NOT full simulator episode
wall-clock — so numbers are attributable to the model, not the environment.

Usage:
    python dexmani_policy/tools/profile_action_flow.py [config_name] \
        [--warmup 50] [--measurement 500] [--mode train|infer|both]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys

_script_dir = pathlib.Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

import torch
from torch.utils.data import DataLoader

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import dict_apply, set_seed, worker_init_fn
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
)

register_resolvers()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_cfg(config_name: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass
    config_dir = os.path.join(_project_root, "dexmani_policy", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    cfg.workspace.output_dir = "/tmp/profile_action_flow"
    OmegaConf.resolve(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


class CudaTimer:
    """Context manager timing a block with a pair of CUDA events."""

    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.ms = 0.0

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *exc):
        self._end.record()
        torch.cuda.synchronize()
        self.ms = self._start.elapsed_time(self._end)


def _cycle(loader: DataLoader):
    """Infinite iterator over a DataLoader (re-shuffles each pass)."""
    while True:
        for batch in loader:
            yield batch


def _summarize(times_ms: list[float]) -> dict:
    n = len(times_ms)
    if n == 0:
        return {}
    times_ms = sorted(times_ms)
    total_s = sum(times_ms) / 1000.0
    return {
        "mean_ms": round(statistics.mean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(times_ms[int(0.95 * (n - 1))], 3),
        "samples_per_sec": round(n / total_s, 1) if total_s > 0 else 0.0,
    }


def _report(title: str, summary: dict):
    if not summary:
        print(f"  {title:<20} (no samples)")
        return
    print(
        f"  {title:<20} mean={summary['mean_ms']:>8.3f}ms  "
        f"median={summary['median_ms']:>8.3f}ms  "
        f"p95={summary['p95_ms']:>8.3f}ms  "
        f"{summary['samples_per_sec']:>9.1f} it/s"
    )


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _profile_training(model, ema_model, ema_updater, optimizer, batches, cfg, warmup, measure):
    device = torch.device(cfg.training.device)
    timers = {"forward": [], "backward": [], "opt_step": [], "total_step": []}

    use_ema_teacher = cfg.training.use_ema_teacher_for_consistency
    loss_kwargs = (
        {"ema_backbone": ema_model.action_decoder.model}
        if use_ema_teacher and ema_model is not None
        else {}
    )
    max_grad_norm = cfg.training.max_grad_norm

    print(
        f"\n== Training (bf16={cfg.training.use_bfloat16}, "
        f"accum={cfg.training.loop.gradient_accumulation_steps}) =="
    )
    for i in range(warmup + measure):
        batch = dict_apply(next(batches), lambda x: x.to(device, non_blocking=True))
        model.train()

        if i < warmup:
            raw_loss, _ = model.compute_loss(batch, **loss_kwargs)
            raw_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_updater is not None:
                ema_updater.step(model)
            continue

        t_total = CudaTimer()
        with t_total:
            t_fwd = CudaTimer()
            with t_fwd:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=cfg.training.use_bfloat16
                ):
                    raw_loss, _ = model.compute_loss(batch, **loss_kwargs)
            t_bwd = CudaTimer()
            with t_bwd:
                raw_loss.backward()
            t_opt = CudaTimer()
            with t_opt:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema_updater is not None:
                    ema_updater.step(model)

        timers["forward"].append(t_fwd.ms)
        timers["backward"].append(t_bwd.ms)
        timers["opt_step"].append(t_opt.ms)
        timers["total_step"].append(t_total.ms)

    for name in ("forward", "backward", "opt_step", "total_step"):
        _report(name, _summarize(timers[name]))


def _profile_inference(model, batches, cfg, warmup, measure):
    device = torch.device(cfg.training.device)
    timers = {"obs_encoder": [], "decoder_denoise": [], "predict_total": []}
    model.eval()

    print(f"\n== Inference (NFE={cfg.agent.denoise_steps}, solver={cfg.agent.solver}) ==")
    for i in range(warmup + measure):
        batch = dict_apply(next(batches), lambda x: x.to(device, non_blocking=True))
        obs = {k: v[:1] for k, v in batch["obs"].items()}

        if i < warmup:
            with torch.no_grad():
                model.predict_action(obs)
            continue

        with torch.no_grad():
            t_enc = CudaTimer()
            with t_enc:
                cond, _ = model._build_cond(obs)
            t_dec = CudaTimer()
            with t_dec:
                model.predict_action_from_cond(cond)
            t_tot = CudaTimer()
            with t_tot:
                model.predict_action(obs)

        timers["obs_encoder"].append(t_enc.ms)
        timers["decoder_denoise"].append(t_dec.ms)
        timers["predict_total"].append(t_tot.ms)

    for name in ("obs_encoder", "decoder_denoise", "predict_total"):
        _report(name, _summarize(timers[name]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="action_flow")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measurement", type=int, default=500)
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    train_loader = DataLoader(dataset, worker_init_fn=worker_init_fn, **cfg.dataloader)
    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)
    optimizer, _ = build_optimizer_and_scheduler(cfg, model, len(train_loader))

    batches = _cycle(train_loader)

    if args.mode in ("train", "both"):
        _profile_training(model, ema_model, ema_updater, optimizer, batches, cfg, args.warmup, args.measurement)
    if args.mode in ("infer", "both"):
        _profile_inference(model, batches, cfg, args.warmup, args.measurement)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"\npeak CUDA memory (during warmup+measurement): {peak:.1f} MiB")


if __name__ == "__main__":
    main()
