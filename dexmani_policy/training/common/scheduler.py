import math

import torch.optim.lr_scheduler as lr_scheduler
from diffusers.optimization import get_scheduler as diffusion_scheduler


def cosine_with_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_min_ratio: float = 0.1,
    last_epoch: int = -1,
):
    """LAMP's zero-based linspace warmup followed by its discrete cosine schedule."""
    if num_training_steps < 1 or num_warmup_steps < 0 or not 0 <= lr_min_ratio <= 1:
        raise ValueError("Invalid cosine schedule length, warmup or minimum LR ratio")
    num_warmup_steps = min(num_warmup_steps, num_training_steps)

    def lr_lambda(step: int) -> float:
        step = min(step, num_training_steps - 1)
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps - 1))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, progress)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_scheduler(
    optimizer,
    name: str,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
):
    """Use library schedules, with the two project-specific choices kept local."""
    lr_min_ratio = kwargs.pop("lr_min_ratio", 0.1)
    if name == "cosine_min_lr":
        return cosine_with_min_lr(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_min_ratio=lr_min_ratio,
            last_epoch=kwargs.get("last_epoch", -1),
        )

    if name == "one_cycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.pop("max_lr", [pg["lr"] for pg in optimizer.param_groups]),
            total_steps=num_training_steps,
            **kwargs,
        )

    return diffusion_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
