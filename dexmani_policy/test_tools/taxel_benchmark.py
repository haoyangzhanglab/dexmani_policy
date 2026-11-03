"""One fresh CUDA process per condition; see README for the alternating A/B command."""

import argparse
import copy
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.pytorch_util import dict_apply, set_seed
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.data import fit_normalizer
from dexmani_policy.training.common.optimization import build_optimizer, build_scheduler


def summary(times):
    quartiles = statistics.quantiles(times, n=4)
    return dict(median_ms=statistics.median(times), iqr_ms=quartiles[2] - quartiles[0],
                min_ms=min(times), max_ms=max(times), times_ms=times)


def memory(device):
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if args.steps < 20:
        parser.error("--steps must be at least 20")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; CPU timings are not comparable")
    torch.set_num_threads(1)
    cfg = load_policy_config("adaptac", [
        "tactile=taxel", "dataset.augmentation_cfg=null", "dataloader.num_workers=0",
        f"training.cudnn_benchmark={args.benchmark}",
    ])
    device = torch.device(cfg.training.device)
    torch.cuda.set_device(device)
    set_seed(cfg.training.seed)
    # Match PolicyTrainer.train; do not change deterministic or cuDNN TF32 for A/B.
    torch.set_float32_matmul_precision("highest")
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    batch_size, n = cfg.dataloader.batch_size, cfg.n_obs_steps
    obs = dict(
        point_cloud=torch.randn(batch_size, n, cfg.agent.num_points, cfg.agent.pc_dim),
        joint_state=torch.randn(batch_size, n, 19),
        taxel_tactile_force=torch.randn(batch_size, n, *model.force_shape),
    )
    batch = dict(
        obs=obs, action=torch.randn(batch_size, cfg.horizon, cfg.action_dim),
        future_force=torch.randn(batch_size, model.future_steps, *model.force_shape),
        future_force_valid=torch.ones(batch_size, model.future_steps, dtype=torch.bool),
    )
    dims = dict(point_cloud=cfg.agent.pc_dim, joint_state=19,
                taxel_tactile_force=model.force_dim, action=cfg.action_dim)
    model.configure_normalizer(dims, {})
    # Fit the real limits normalization on synthetic source values, outside timing.
    source = dict(obs, action=batch["action"])
    dataset = SimpleNamespace(action_key="action", iter_normalization_data=lambda key: iter([source[key]]))
    model.normalizer = fit_normalizer(dataset, cfg.normalization, feature_dims=dims)
    ema_model = copy.deepcopy(model).to(device)
    model.to(device).train()
    ema = hydra.utils.instantiate(cfg.ema, model=model, ema_model=ema_model)
    optimizer = build_optimizer(model, **cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.training.lr_scheduler,
                                cfg.training.lr_warmup_steps, cfg.training.loop.total_train_steps)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    batch = dict_apply(batch, lambda x: x.pin_memory())
    optimizer.zero_grad(set_to_none=True)
    # Last backend assignment, after set_seed and all initialization. No training warmup yet.
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    backend = dict(benchmark=torch.backends.cudnn.benchmark,
                   deterministic=torch.backends.cudnn.deterministic,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32,
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   matmul_precision=torch.get_float32_matmul_precision(),
                   benchmark_limit=torch.backends.cudnn.benchmark_limit)
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(device),
                       total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
                       threads=torch.get_num_threads(),
                       nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True))
    print(json.dumps(dict(backend=backend, environment=environment)), flush=True)

    def update():
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        gpu_batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        loss, _ = model(gpu_batch, log_metrics=False)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        # Same gradient clipping and nonfinite check as PolicyTrainer.optimizer_step.
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.training.max_grad_norm,
                                                  error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.step()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, loss.item(), grad_norm.item()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    first_ms, first_loss, first_grad_norm = update()
    first = dict(ms=first_ms, loss=first_loss, grad_norm=first_grad_norm, **memory(device))
    print(json.dumps(dict(first=first)), flush=True)
    warmup = []
    # Require two adjacent 10-update windows within 3%, after at least 30 updates.
    for _ in range(100):
        warmup.append(update()[0])
        if len(warmup) >= 30 and len(warmup) % 10 == 0:
            left, right = statistics.median(warmup[-20:-10]), statistics.median(warmup[-10:])
            if abs(right / left - 1) <= 0.03:
                break
    else:
        raise RuntimeError(f"Warmup did not stabilize within 100 updates: {warmup}")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = [update() for _ in range(args.steps)]
    steady = dict(**summary([s[0] for s in samples]), **memory(device),
                  losses=[s[1] for s in samples], grad_norms=[s[2] for s in samples])
    # Check both raw and EMA outputs only AFTER memory/timing measurements.
    inference_obs = dict_apply(batch["obs"], lambda x: x[:1].to(device))
    for policy in (model, ema_model):
        policy.eval()
        with torch.inference_mode():
            output = policy.predict_action(inference_obs)
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise RuntimeError("Nonfinite inference output")
    result = dict(backend=backend, environment=environment,
                  config=OmegaConf.to_container(cfg, resolve=False), first=first,
                  warmup=summary(warmup), steady=steady, finite_loss_grad_output=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), steady_median_ms=steady["median_ms"],
                          finite_loss_grad_output=True)), flush=True)


if __name__ == "__main__":
    main()
