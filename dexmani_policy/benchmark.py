#!/usr/bin/env python3
"""Benchmark script for DexMani_Policy agents.

Trains each agent for 10 epochs, collecting parameter counts, training time,
GPU memory usage, and running post-training validation checks.

Usage::

    python dexmani_policy/benchmark.py              # all 6 agents
    python dexmani_policy/benchmark.py --agent dp3  # single agent
"""

import gc
import json
import os
import pathlib
import sys
import tempfile
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from termcolor import colored, cprint

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import (
    compile_models,
    count_params,
    dict_apply,
    fix_state_dict,
    set_seed,
    worker_init_fn,
)
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
    compute_num_training_steps,
)

register_resolvers()

# ═════════════════════════════════════════════════════════════════════════════════════
# Config loading
# ═════════════════════════════════════════════════════════════════════════════════════

BENCHMARK_OVERRIDES = [
    "training.loop.num_epochs=10",
    "training.loop.eval_interval_epochs=0",  # skip env_runner construction
]


def load_benchmark_config(config_name: str):
    """Load a Hydra config for benchmarking (no Hydra runtime needed)."""
    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass
    config_dir = os.path.join(ROOT_DIR, "dexmani_policy", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=BENCHMARK_OVERRIDES)
        # compose has no runtime — manually replace ${hydra:runtime.output_dir}
        cfg.workspace.output_dir = f"/tmp/benchmark_{config_name}"
        OmegaConf.resolve(cfg)
    return cfg


# ═════════════════════════════════════════════════════════════════════════════════════
# Parameter collection
# ═════════════════════════════════════════════════════════════════════════════════════

def collect_param_info(module: nn.Module) -> Dict[str, Any]:
    """Collect structured parameter counts for *module* and its children."""
    total, trainable = count_params(module)
    info: Dict[str, Any] = {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "children": {},
    }
    for name, child in module.named_children():
        t, tr = count_params(child)
        info["children"][name] = {
            "total": t,
            "trainable": tr,
            "frozen": t - tr,
        }
    return info


# ═════════════════════════════════════════════════════════════════════════════════════
# GPU memory helpers
# ═════════════════════════════════════════════════════════════════════════════════════

def _mem_snapshot() -> Dict[str, int]:
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def cleanup_gpu():
    """Aggressively free GPU memory between agent runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ═════════════════════════════════════════════════════════════════════════════════════
# Gradient NaN check (layer-2 protection, mirrors trainer.py:128-141)
# ═════════════════════════════════════════════════════════════════════════════════════

def _grad_has_nan(model: nn.Module) -> bool:
    """Return True if any parameter has a non-finite gradient."""
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════════════

def train_benchmark_epochs(
    model: nn.Module,
    ema_model,
    ema_updater,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    cfg,
    device: torch.device,
    num_epochs: int = 10,
) -> Tuple[List[float], List[float], List[Dict], Dict[str, int]]:
    """Train *model* for *num_epochs* with timing and memory instrumentation.

    Returns:
        epoch_losses:    mean loss per epoch (length = num_epochs)
        epoch_times:     wall-clock seconds per epoch
        nan_events:      list of {epoch, type, ...} dicts for each NaN encounter
        peak_mem:        dict with peak_allocated, peak_reserved, after_warmup_*
    """
    use_bfloat16 = cfg.training.get("use_bfloat16", False)
    use_compile = cfg.training.get("use_compile", False)
    max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
    use_ema_teacher = cfg.training.get("use_ema_teacher_for_consistency", False)

    use_ema = ema_model is not None and ema_updater is not None

    # -- compile ----------------------------------------------------------------
    if use_compile:
        compile_models(model, ema_model if use_ema else None)

    # -- loss kwargs for consistency training -----------------------------------
    loss_kwargs: Dict[str, Any] = {}
    if use_ema_teacher and use_ema:
        loss_kwargs["ema_backbone"] = ema_model.action_decoder.model

    # -- precision & warmup (trigger CUDA graph capture / JIT for compiled models)
    torch.set_float32_matmul_precision('high')
    WARMUP_STEPS = 5
    model.train()
    warmup_iter = iter(train_loader)
    for _ in range(WARMUP_STEPS):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            batch = next(warmup_iter)
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            raw_loss, _ = model.compute_loss(batch, **loss_kwargs)
        raw_loss.backward()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # -- memory baseline (post-warmup, pre-training) ----------------------------
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    mem_after_warmup = _mem_snapshot()

    # -- training epochs --------------------------------------------------------
    epoch_losses: List[float] = []
    epoch_times: List[float] = []
    nan_events: List[Dict] = []

    for epoch in range(num_epochs):
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        epoch_start = time.perf_counter()
        step_losses: List[float] = []
        epoch_nan = 0

        for batch in train_loader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # --- forward ---
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                raw_loss, _loss_dict = model.compute_loss(batch, **loss_kwargs)

            # --- layer-1 NaN guard: loss ---
            if not torch.isfinite(raw_loss):
                nan_events.append({
                    "epoch": epoch,
                    "type": "loss",
                    "value": float(raw_loss.item()) if not torch.isnan(raw_loss) else "NaN",
                })
                epoch_nan += 1
                optimizer.zero_grad(set_to_none=True)
                if epoch_nan >= 3:
                    raise RuntimeError(
                        f"Aborting: ≥3 NaN losses in epoch {epoch}"
                    )
                continue

            # --- backward ---
            raw_loss.backward()

            # --- gradient clipping ---
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # --- layer-2 NaN guard: gradients ---
            if _grad_has_nan(model):
                nan_events.append({"epoch": epoch, "type": "grad"})
                optimizer.zero_grad(set_to_none=True)
                continue

            # --- optimizer / scheduler / EMA ---
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if use_ema:
                ema_updater.step(model)

            step_losses.append(raw_loss.item())

        # end of epoch
        torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)
        epoch_losses.append(
            sum(step_losses) / len(step_losses) if step_losses else float("nan")
        )

    # -- peak memory ------------------------------------------------------------
    peak_mem = {
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "after_warmup_allocated": mem_after_warmup["allocated"],
        "after_warmup_reserved": mem_after_warmup["reserved"],
    }

    return epoch_losses, epoch_times, nan_events, peak_mem


# ═════════════════════════════════════════════════════════════════════════════════
# Validation suite (post-training)
# ═════════════════════════════════════════════════════════════════════════════════

def run_validations(
    model: nn.Module,
    ema_model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    cfg,
    device: torch.device,
    use_ema_teacher: bool,
    epoch_losses: List[float],
    nan_events: List[Dict],
) -> Dict[str, Any]:
    """Run all post-training checks.  Returns {check_name: True/False/"n/a"}."""
    results: Dict[str, Any] = {}
    use_ema = ema_model is not None

    # Shared loss kwargs
    loss_kwargs: Dict[str, Any] = {}
    if use_ema_teacher and use_ema:
        loss_kwargs["ema_backbone"] = ema_model.action_decoder.model

    # Grab a batch for validation
    batch = next(iter(train_loader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

    # ── V1: forward pass loss finite (base model, eval mode) ──────────────────
    try:
        model.eval()
        with torch.no_grad():
            raw_loss, _ = model.compute_loss(batch, **loss_kwargs)
        results["V1_forward_loss_finite"] = bool(torch.isfinite(raw_loss))
    except Exception as exc:
        results["V1_forward_loss_finite"] = False
        results["V1_error"] = str(exc)

    # ── V2: predict_action shape ──────────────────────────────────────────────
    try:
        model.eval()
        with torch.no_grad():
            obs_sample = {k: v[:1] for k, v in batch["obs"].items()}
            pred = model.predict_action(obs_sample)
        expected = (1, cfg.n_action_steps, cfg.action_dim)
        actual = tuple(pred["control_action"].shape)
        results["V2_predict_shape_ok"] = (actual == expected)
        if not results["V2_predict_shape_ok"]:
            results["V2_detail"] = f"expected {expected}, got {actual}"
    except Exception as exc:
        results["V2_predict_shape_ok"] = False
        results["V2_error"] = str(exc)

    # ── V3: checkpoint save → load roundtrip ──────────────────────────────────
    try:
        num_steps = compute_num_training_steps(cfg, len(train_loader))
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = pathlib.Path(tmpdir)
            store = CheckpointStore(ckpt_dir)

            model_sd = {k: v.clone() for k, v in model.state_dict().items()}
            ema_sd = (
                {k: v.clone() for k, v in ema_model.state_dict().items()}
                if use_ema else None
            )

            ckpt = TrainCheckpoint(
                epoch=10,
                global_step=10 * len(train_loader),
                model_state=model_sd,
                ema_model_state=fix_state_dict(ema_sd, is_current_ddp=False) if ema_sd is not None else None,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                monitor={"test_mean_score": 0.5},
                train_params={
                    "n_obs_steps": model.n_obs_steps,
                    "n_action_steps": model.n_action_steps,
                    "action_dim": model.action_dim,
                    "horizon": model.horizon,
                    "action_key": getattr(model, "action_key", "action"),
                    "num_training_steps": num_steps,
                },
            )
            ckpt_path = store.save("benchmark_ckpt.pt", ckpt)
            loaded = store.load(ckpt_path)

            # Verify model state dict
            loaded_sd = fix_state_dict(loaded.model_state, is_current_ddp=False)
            loaded_sd = {k: v.to(device) for k, v in loaded_sd.items()}
            match = all(
                torch.equal(model_sd[k], loaded_sd[k])
                for k in model_sd
            )
            results["V3_checkpoint_roundtrip_ok"] = match
    except Exception as exc:
        results["V3_checkpoint_roundtrip_ok"] = False
        results["V3_error"] = str(exc)

    # ── V4: loss decreases over training ──────────────────────────────────────
    results["V4_loss_decreased"] = (
        len(epoch_losses) >= 2 and epoch_losses[-1] < epoch_losses[0]
    )

    # ── V5: no NaN during training ────────────────────────────────────────────
    results["V5_no_nan_during_training"] = len(nan_events) == 0
    if nan_events:
        results["V5_nan_events"] = nan_events[:5]

    # ── V6: EMA parameters differ from base model ─────────────────────────────
    if use_ema:
        try:
            differs = False
            for (_, p1), (_, p2) in zip(
                model.named_parameters(), ema_model.named_parameters()
            ):
                if not torch.equal(p1, p2):
                    differs = True
                    break
            results["V6_ema_differs_from_model"] = differs
        except Exception:
            results["V6_ema_differs_from_model"] = False
    else:
        results["V6_ema_differs_from_model"] = "n/a"

    # ── V7: EMA model predict_action ──────────────────────────────────────────
    if use_ema:
        try:
            ema_model.eval()
            with torch.no_grad():
                obs_sample = {k: v[:1] for k, v in batch["obs"].items()}
                ema_pred = ema_model.predict_action(obs_sample)
            expected = (1, cfg.n_action_steps, cfg.action_dim)
            results["V7_ema_predict_shape_ok"] = (
                tuple(ema_pred["control_action"].shape) == expected
            )
        except Exception as exc:
            results["V7_ema_predict_shape_ok"] = False
            results["V7_error"] = str(exc)
    else:
        results["V7_ema_predict_shape_ok"] = "n/a"

    return results


# ═════════════════════════════════════════════════════════════════════════════════
# Single-agent benchmark
# ═════════════════════════════════════════════════════════════════════════════════

def benchmark_one_agent(config_name: str) -> Dict[str, Any]:
    """Run the full benchmark pipeline for a single agent config.

    Returns a results dict (see ``generate_report`` for expected keys).
    """
    cprint(f"\n{'='*70}", "cyan", attrs=["bold"])
    cprint(f"  Benchmark: {config_name}", "cyan", attrs=["bold"])
    cprint(f"{'='*70}", "cyan", attrs=["bold"])

    # ── config ────────────────────────────────────────────────────────────────
    cfg = load_benchmark_config(config_name)
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

    use_ema_teacher = cfg.training.get("use_ema_teacher_for_consistency", False)

    # ── build components ──────────────────────────────────────────────────────
    cprint("  [1/4] Building components ...", "yellow")
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    train_loader = DataLoader(
        dataset, worker_init_fn=worker_init_fn, **cfg.dataloader
    )
    batches_per_epoch = len(train_loader)
    print(f"        Dataset: {len(dataset)} samples, {batches_per_epoch} batches/epoch")

    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)
    use_ema = ema_model is not None

    optimizer, scheduler = build_optimizer_and_scheduler(
        cfg, model, batches_per_epoch,
    )

    # ── parameter info ────────────────────────────────────────────────────────
    param_info = collect_param_info(model)
    print(f"        Params: {param_info['total']/1e6:.2f}M total  "
          f"({param_info['trainable']/1e6:.2f}M trainable, "
          f"{param_info['frozen']/1e6:.2f}M frozen)")

    # ── train ─────────────────────────────────────────────────────────────────
    cprint("  [2/4] Training 10 epochs ...", "yellow")
    epoch_losses, epoch_times, nan_events, peak_mem = train_benchmark_epochs(
        model, ema_model, ema_updater,
        optimizer, scheduler,
        train_loader, cfg, device,
        num_epochs=10,
    )

    total_time = sum(epoch_times)
    mean_epoch = total_time / len(epoch_times) if epoch_times else 0.0
    print(f"        Time: {total_time:.1f}s total, {mean_epoch:.2f}s/epoch mean, "
          f"{epoch_times[0]:.2f}s (epoch 1), {epoch_times[-1]:.2f}s (epoch 10)")
    print(f"        Peak GPU: {peak_mem['peak_allocated']/1024**2:.0f} MB allocated, "
          f"{peak_mem['peak_reserved']/1024**2:.0f} MB reserved")

    # ── validations ───────────────────────────────────────────────────────────
    cprint("  [3/4] Running validations ...", "yellow")
    validations = run_validations(
        model, ema_model, optimizer, scheduler,
        train_loader, cfg, device,
        use_ema_teacher, epoch_losses, nan_events,
    )

    # Determine pass/fail from validation results
    checks = {k: v for k, v in validations.items()
              if not k.startswith("V") or k[1].isdigit()}  # filter non-check keys
    failed = [k for k, v in validations.items()
              if v is not True and v != "n/a"]

    if failed:
        cprint(f"        FAILED checks: {failed}", "red")
    else:
        cprint("        All validations PASSED", "green")

    # Capture dataset size before cleanup
    dataset_size_val: int = len(dataset) if hasattr(dataset, '__len__') else -1

    # ── cleanup ───────────────────────────────────────────────────────────────
    cprint("  [4/4] Cleaning up ...", "yellow")
    # Remove references so GC can free GPU memory
    del train_loader, dataset, normalizer
    del model, ema_model, ema_updater
    del optimizer, scheduler
    cleanup_gpu()

    return {
        "config_name": config_name,
        "policy_name": cfg.policy_name,
        "status": "fail" if failed else "pass",
        "params": param_info,
        "time_total_seconds": round(total_time, 2),
        "time_per_epoch_seconds": [round(t, 3) for t in epoch_times],
        "time_mean_epoch_seconds": round(mean_epoch, 3),
        "gpu_peak_allocated_bytes": peak_mem["peak_allocated"],
        "gpu_peak_reserved_bytes": peak_mem["peak_reserved"],
        "gpu_after_warmup_allocated_bytes": peak_mem["after_warmup_allocated"],
        "gpu_after_warmup_reserved_bytes": peak_mem["after_warmup_reserved"],
        "epoch_losses": [round(x, 6) for x in epoch_losses],
        "nan_events": nan_events,
        "validations": validations,
        "batches_per_epoch": batches_per_epoch,
        "dataset_size": dataset_size_val,
    }


# ═════════════════════════════════════════════════════════════════════════════════
# Report generation
# ═════════════════════════════════════════════════════════════════════════════════

def generate_report(all_results: List[Dict]) -> str:
    """Print a formatted table and write JSON.  Returns the JSON file path."""
    # Ensure output directory
    output_dir = pathlib.Path(ROOT_DIR) / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    json_path = output_dir / f"benchmark_{timestamp}.json"

    # Write JSON
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    # ── Terminal table ────────────────────────────────────────────────────────
    print()
    cprint("=" * 105, "cyan", attrs=["bold"])
    cprint("  Benchmark Results — 10 epochs per agent, single GPU", "cyan", attrs=["bold"])
    cprint("=" * 105, "cyan", attrs=["bold"])
    print()

    # Header
    header = (
        f"{'Agent':<14} {'Params(M)':>10} {'Train(M)':>9} "
        f"{'Time(s)':>9} {'Mean/Epoch(s)':>14} {'PeakMem(MB)':>12} "
        f"{'E1→E10 Loss':>18}   {'Status':>8}"
    )
    sep = (
        f"{'─'*14} {'─'*10} {'─'*9} "
        f"{'─'*9} {'─'*14} {'─'*12} "
        f"{'─'*18}   {'─'*8}"
    )
    print(header)
    print(sep)

    counts = {"pass": 0, "fail": 0, "skip": 0}

    for r in all_results:
        cfg_name = r["config_name"]
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        color = {"pass": "green", "fail": "red", "skip": "yellow"}.get(status, "white")

        p = r.get("params", {})
        total_m = p.get("total", 0) / 1e6
        train_m = p.get("trainable", 0) / 1e6

        t_total = r.get("time_total_seconds", 0)
        t_mean = r.get("time_mean_epoch_seconds", 0)
        mem_mb = r.get("gpu_peak_allocated_bytes", 0) / 1024 ** 2

        losses = r.get("epoch_losses", [])
        l1 = f"{losses[0]:.4f}" if losses else "N/A"
        l10 = f"{losses[-1]:.4f}" if losses else "N/A"

        print(
            f"{cfg_name:<14} {total_m:>10.2f} {train_m:>9.2f} "
            f"{t_total:>9.1f} {t_mean:>14.2f} {mem_mb:>12.0f} "
            f"{l1:>8} → {l10:<8}  {colored(status.upper(), color):>8}"
        )

    print(sep)
    total_time = sum(r.get("time_total_seconds", 0) for r in all_results)
    n = len(all_results)
    summary = f"Summary: {counts['pass']}/{n} PASS"
    if counts.get("fail", 0):
        summary += f", {counts['fail']}/{n} FAIL"
    if counts.get("skip", 0):
        summary += f", {counts['skip']}/{n} SKIP"
    print(f"\n  {summary}  |  Total wall-clock: {total_time:.1f}s")

    # ── Validation details ────────────────────────────────────────────────────
    print(f"\n  Validation Details:")
    for r in all_results:
        cfg_name = r["config_name"]
        v = r.get("validations", {})
        # Only show V1–V7 checks
        check_items = sorted(
            [(k, v[k]) for k in v if k.startswith("V") and k[1].isdigit()],
            key=lambda x: x[0],
        )
        parts = []
        for key, val in check_items:
            if val is True:
                parts.append(f"{colored('✓', 'green')}{key}")
            elif val == "n/a":
                continue
            else:
                parts.append(f"{colored('✗', 'red')}{key}")
        line = "  ".join(parts)
        print(f"    {cfg_name:<14} {line}")

    # ── NaN warnings ──────────────────────────────────────────────────────────
    for r in all_results:
        if r.get("nan_events"):
            cprint(
                f"\n  ⚠ {r['config_name']} NaN events during training: "
                f"{r['nan_events']}", "yellow",
            )

    # ── Fail details ──────────────────────────────────────────────────────────
    for r in all_results:
        if r["status"] in ("fail", "skip") and "error" in r:
            cprint(f"  ⚠ {r['config_name']}: {r['error']}", "yellow")

    print(f"\n  JSON report: {json_path}")
    return str(json_path)


# ═════════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark DexMani_Policy agents (10 epochs each)"
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        help="Run a single agent (e.g. dp3) instead of all 6.",
    )
    args = parser.parse_args()

    agent_names = (
        [args.agent] if args.agent
        else ["dp", "dp3", "moe_dp3", "maniflow", "r3d", "multitask_dit"]
    )

    all_results: List[Dict] = []
    for name in agent_names:
        cleanup_gpu()
        try:
            result = benchmark_one_agent(name)
            all_results.append(result)
        except FileNotFoundError as exc:
            cprint(f"\n  SKIP {name}: dataset not found — {exc}", "yellow")
            all_results.append({
                "config_name": name, "status": "skip", "error": str(exc),
            })
        except torch.cuda.OutOfMemoryError as exc:
            cprint(f"\n  FAIL {name}: CUDA OOM — {exc}", "red")
            cleanup_gpu()
            all_results.append({
                "config_name": name, "status": "fail",
                "error": f"CUDA OOM: {exc}",
            })
        except Exception as exc:
            cprint(f"\n  FAIL {name}: {type(exc).__name__}: {exc}", "red")
            traceback.print_exc()
            cleanup_gpu()
            all_results.append({
                "config_name": name, "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            })

    generate_report(all_results)

    has_failure = any(r["status"] == "fail" for r in all_results)
    if has_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
