from typing import Any

import torch


def accumulate_metrics(totals, metrics):
    """Accumulate detached scalars without synchronizing the device."""
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().reshape(()).to(dtype=torch.float64)
        else:
            value = float(value)
        totals[key] = totals.get(key, 0.0) + value


def to_log_scalars(metrics: dict[str, Any]) -> dict[str, float]:
    scalars = {}
    tensors = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            keys, values = tensors.setdefault(value.device, ([], []))
            keys.append(key)
            values.append(value.detach().reshape(()).to(dtype=torch.float64))
        else:
            scalars[key] = float(value)
    for keys, values in tensors.values():
        scalars.update(zip(keys, torch.stack(values).cpu().tolist()))
    return scalars


def count_params(module) -> tuple[int, int]:
    total = trainable = 0
    for parameter in module.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def print_param_count(agent) -> None:
    total, _ = count_params(agent)
    fields = agent.consumed_observation_fields
    if fields:
        print(f"[Modalities] {', '.join(fields)}")
    print(f"[{type(agent).__name__}] Parameter Count")
    print(f"  Total: {total / 1e6:.2f} M")

    for name, child in agent.named_children():
        t, tr = count_params(child)
        frozen = t - tr
        print(
            f"  {name:<20}: {t / 1e6:.2f} M  "
            f"(trainable={tr / 1e6:.2f} M  frozen={frozen / 1e6:.2f} M)"
        )


def print_training_recipe(
    cfg, *, batches_per_epoch: int, samples_per_epoch: int
) -> None:
    """Distinguish nominal batches from the windows actually processed."""
    batch_size = int(cfg.dataloader.batch_size)
    grad_accum = cfg.training.loop.gradient_accumulation_steps
    total_train_steps = int(cfg.training.loop.total_train_steps)
    nominal_batch = batch_size * grad_accum
    updates_per_epoch = (batches_per_epoch + grad_accum - 1) // grad_accum
    remainder = batches_per_epoch % grad_accum
    last_group_microbatches = grad_accum if remainder == 0 else remainder
    epoch_samples = min(samples_per_epoch, batches_per_epoch * batch_size)
    last_group_batch = epoch_samples - (updates_per_epoch - 1) * grad_accum * batch_size
    full_epochs, partial_updates = divmod(total_train_steps, updates_per_epoch)
    processed_budget = (
        full_epochs * epoch_samples
        + min(partial_updates * grad_accum * batch_size, epoch_samples)
    )
    has_partial_accumulation_group = last_group_microbatches != grad_accum
    lr = cfg.optimizer.lr
    obs_lr = cfg.optimizer.get("obs_lr")
    obs_lr = lr if obs_lr is None else obs_lr

    rows = [
        (
            "temporal sampling",
            "recorded_rows (no interpolation; eligibility follows data_recipe)",
        ),
        ("micro-batch size", batch_size),
        ("gradient accumulation", grad_accum),
        ("nominal effective batch", nominal_batch),
        ("batches / epoch", batches_per_epoch),
        ("optimizer updates / epoch", updates_per_epoch),
        ("partial accum group", "yes" if has_partial_accumulation_group else "no"),
        ("last group micro-batches", last_group_microbatches),
        ("last group batch", last_group_batch),
        ("total optimizer updates", total_train_steps),
        ("nominal sample budget", nominal_batch * total_train_steps),
        ("processed window budget", processed_budget),
        ("window counting", "counts processed windows across epochs; not unique source rows"),
        ("drop_last", cfg.dataloader.drop_last),
        ("learning rate", f"{lr:.3e}"),
        ("obs learning rate", f"{obs_lr:.3e}"),
        ("warmup updates", cfg.training.lr_warmup_steps),
        ("scheduler", cfg.training.lr_scheduler),
        ("precision", "bf16" if cfg.training.get("use_bfloat16", False) else "fp32"),
        ("torch.compile", cfg.training.get("use_compile", False)),
        ("EMA", cfg.training.use_ema),
    ]
    if "contract" in cfg.get("hand_data", {}):
        dt = cfg.hand_data.contract.dt
        rows.extend([
            ("recorded dt (s)", dt),
            ("prediction control duration (s)", cfg.horizon * dt),
            ("execution control duration (s)", cfg.n_action_steps * dt),
            ("codec precision", "fp32; policy master parameters fp32, autocast as above"),
        ])
    print("=" * 60)
    print("Training Recipe")
    print("-" * 60)
    for label, value in rows:
        print(f"{label:<26}: {value}")
    print("=" * 60)
