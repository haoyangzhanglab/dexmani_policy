"""ActionFlow model-side B=1 steady-state GPU inference profiler.

Inference reuses one real dataset observation already on the GPU. CUDA events
measure condition building (preprocessing + obs encoder) and action generation
on the same pass; loading, H2D, D2H and warmup are excluded. Models are freshly
initialized, with dataset-derived normalization; this does not measure quality.
The separate training mode retains forward / backward / optimizer+EMA profiling.

Usage:
    python dexmani_policy/tools/profile_action_flow.py [config_name] \
        [--warmup 50] [--measurement 500] [--mode infer|train] \
        [--precision fp32|bf16] [--compile] [--nfe N]
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

import hydra
import torch
from torch.utils.data import DataLoader

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import (
    compile_models,
    count_params,
    dict_apply,
    set_seed,
    worker_init_fn,
)
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
        f"p50={summary['median_ms']:>8.3f}ms  "
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


def _profile_inference(model, obs, device, warmup, measure, *, precision, nfe):
    timers = {"obs_encoder": [], "action_generate": [], "total": []}
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=(precision == "bf16")
    ):
        for i in range(warmup):
            result = model.predict_action(obs, denoise_timesteps=nfe)
            if i == 0:
                expected_shapes = {
                    "pred_action": (1, model.horizon, model.action_dim),
                    "control_action": (1, model.n_action_steps, model.control_action_dim),
                }
                for name, shape in expected_shapes.items():
                    if tuple(result[name].shape) != shape:
                        raise ValueError(f"{name}: expected {shape}, got {tuple(result[name].shape)}")
                if not torch.isfinite(result["pred_action"]).all().item():
                    raise ValueError("Warmup produced non-finite actions")
                print(f"  output shapes     {expected_shapes}")
            del result

        # No warmup outputs or conditions remain live at the resident baseline.
        torch.cuda.synchronize(device)
        resident_mib = torch.cuda.memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(measure):
            start.record()
            cond, aux = model._build_cond(obs)
            mid.record()
            result = model.predict_action_from_cond(cond, denoise_timesteps=nfe)
            end.record()

            # One uninterrupted pipeline; only synchronize after generation.
            end.synchronize()
            timers["obs_encoder"].append(start.elapsed_time(mid))
            timers["action_generate"].append(mid.elapsed_time(end))
            timers["total"].append(start.elapsed_time(end))
            # Do not overlap the previous request's tensors with the next one.
            del result, cond, aux

        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    print("\nLatency (GPU-side; warmup and data transfer excluded)")
    print("  obs_encoder = condition_build: preprocessing + normalization + encoder")
    print("  action_generate includes noise, KV cache, solver and action unnormalization")
    for name in timers:
        _report(name, _summarize(timers[name]))
    print(f"\n  resident allocated (steady-state): {resident_mib:.1f} MiB")
    print(f"  peak allocated (measurement):     {peak_mib:.1f} MiB")


def _run_inference(cfg, device, args):
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.action_key
    model.to(device)
    model.eval()

    sample = dataset[0]
    obs = {name: value.unsqueeze(0).to(device) for name, value in sample["obs"].items()}
    del sample, dataset, normalizer

    params = {
        "total": count_params(model)[0],
        "obs_encoder": count_params(model.obs_encoder)[0],
        "ActionDiT": count_params(model.action_decoder.model)[0],
    }
    # Keep solver-specific validation in the decoder, including even midpoint NFE.
    nfe = model.action_decoder._resolve_nfe(args.nfe)
    compile_mode = cfg.training.get("compile_mode", "reduce-overhead")
    print("\n== ActionFlow model-side B=1 steady-state GPU inference ==")
    print(f"  device            {device} ({torch.cuda.get_device_name(device)})")
    print(f"  precision         {args.precision}")
    print(f"  compile           {args.compile}" + (f" (mode={compile_mode})" if args.compile else ""))
    print(f"  solver            {model.action_decoder.solver}")
    print(f"  NFE               {nfe}")
    print("  batch             1")
    print(f"  warmup/measurement {args.warmup}/{args.measurement}")
    print("\nParameters (before compile)")
    for name, count in params.items():
        print(f"  {name:<18} {count:,} ({count / 1e6:.3f} M)")

    if args.compile:
        compile_models(model, None, mode=compile_mode)
    _profile_inference(
        model, obs, device, args.warmup, args.measurement,
        precision=args.precision, nfe=nfe,
    )


def _run_training(cfg, device, args):
    print(f"Device: {device}  ({torch.cuda.get_device_name(device)})")
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    train_loader = DataLoader(dataset, worker_init_fn=worker_init_fn, **cfg.dataloader)
    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)
    optimizer, _ = build_optimizer_and_scheduler(cfg, model, len(train_loader))

    batches = _cycle(train_loader)

    _profile_training(model, ema_model, ema_updater, optimizer, batches, cfg, args.warmup, args.measurement)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"\npeak CUDA memory (during warmup+measurement): {peak:.1f} MiB")


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="action_flow")
    parser.add_argument("--warmup", type=_positive_int, default=50)
    parser.add_argument("--measurement", type=_positive_int, default=500)
    parser.add_argument("--mode", choices=["train", "infer"], default="infer")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                        help="inference autocast precision (default: fp32)")
    parser.add_argument("--compile", action="store_true",
                        help="compile inference via the Agent's existing compile_backbone protocol")
    parser.add_argument("--nfe", type=_positive_int,
                        help="inference NFE override (default: configured decoder NFE)")
    args = parser.parse_args(argv)
    if args.mode == "train" and (args.precision != "fp32" or args.compile or args.nfe is not None):
        parser.error("--precision bf16, --compile and --nfe are inference-only options")
    return args


def main():
    args = _parse_args()
    cfg = _load_cfg(args.config)
    device = torch.device(cfg.training.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ActionFlow GPU profiling requires an available CUDA device")
    set_seed(cfg.training.seed)
    # Events must record on the model's device, including non-default GPU indices.
    with torch.cuda.device(device):
        if args.mode == "infer":
            _run_inference(cfg, device, args)
        else:
            _run_training(cfg, device, args)


if __name__ == "__main__":
    main()
