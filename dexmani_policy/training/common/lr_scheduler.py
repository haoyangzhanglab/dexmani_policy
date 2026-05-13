from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)
import torch.optim.lr_scheduler as _lrs


def get_scheduler(
    optimizer: Optimizer,
    name: Union[str, SchedulerType],
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
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
        if num_training_steps is None:
            raise ValueError(f"{name} requires num_training_steps")
        return _lrs.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=kwargs.get("eta_min", 0),
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

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)