import math
from typing import Optional, Union

import torch.optim.lr_scheduler as _lrs
from diffusers.optimization import (
    TYPE_TO_SCHEDULER_FUNCTION,
    Optimizer,
    SchedulerType,
)


def _cosine_with_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_min_ratio: float = 0.1,
    last_epoch: int = -1,
):
    """Cosine warmup + cosine decay that floors at ``lr_min_ratio * lr`` (not 0).

    Unlike diffusers ``cosine`` (which anneals to 0), this keeps a minimum learning
    rate so late-training steps keep updating rather than decaying to noise.
    """

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, progress)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lrs.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def compute_num_training_steps(cfg, batches_per_epoch: int) -> int:
    """Return total training steps directly from config.

    The ``batches_per_epoch`` parameter is retained for backward compatibility
    with the call signature but is no longer used for calculation.
    """
    return cfg.training.loop.total_train_steps


def get_scheduler(
    optimizer: Optimizer,
    name: Union[str, SchedulerType],
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    lr_min_ratio = kwargs.pop("lr_min_ratio", 0.1)

    if name in ("cosine_min_lr",):
        if num_warmup_steps is None or num_training_steps is None:
            raise ValueError(
                f"{name} requires num_warmup_steps and num_training_steps"
            )
        return _cosine_with_min_lr(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_min_ratio=lr_min_ratio,
            last_epoch=kwargs.get("last_epoch", -1),
        )

    if name in ("one_cycle",):
        if num_training_steps is None:
            raise ValueError(f"{name} requires num_training_steps")
        return _lrs.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", [pg["lr"] for pg in optimizer.param_groups]),
            total_steps=num_training_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
        )

    if name in ("cosine_annealing",):
        raise ValueError(
            "cosine_annealing is deprecated and has been removed. "
            "Use lr_scheduler='cosine' instead (diffusers CosineWithWarmup, "
            "supports warmup)."
        )

    # --- diffusers standard schedulers ---
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs
    )
