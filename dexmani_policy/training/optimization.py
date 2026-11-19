import torch
from torch.optim.swa_utils import get_ema_multi_avg_fn

from dexmani_policy.training.logging import accumulate_metrics


def restore_optimizer(optimizer, state):
    groups, saved = optimizer.param_groups, state["param_groups"]
    if len(groups) != len(saved) or any(
        current["param_names"] != previous["param_names"]
        for current, previous in zip(groups, saved)
    ):
        raise ValueError("Optimizer parameter groups changed")
    optimizer.load_state_dict(state)


def backward_microbatches(iterator, group_size, parameters, forward, *, device, log_metrics):
    finite_loss = torch.ones((), dtype=torch.bool, device=device)
    totals: dict[str, torch.Tensor | float] = {}
    sample_count = 0
    for _ in range(group_size):
        loss, metrics, count = forward(next(iterator))
        sample_count += count
        finite_loss.logical_and_(torch.isfinite(loss.detach()))
        (loss * count if group_size > 1 and count != 1 else loss).backward()
        if log_metrics:
            accumulate_metrics(totals, metrics, weight=count)
    if group_size > 1:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(sample_count)
    return finite_loss, {key: value / sample_count for key, value in totals.items()}


def clip_gradients(parameters, finite_loss, max_grad_norm=0.0, *, step):
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_grad_norm or float("inf"), error_if_nonfinite=False,
    ).detach()
    if not (finite_loss & torch.isfinite(norm)):
        raise FloatingPointError(f"Non-finite loss or gradient norm at step {step}")
    return norm


def build_optimizer(
    model,
    lr: float,
    weight_decay: float,
    obs_lr: float | None = None,
    obs_weight_decay: float | None = None,
    betas: tuple[float, float] = (0.95, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Check trainable parameter coverage and retain parameter names for resume."""
    obs_lr = obs_lr if obs_lr is not None else lr
    obs_weight_decay = obs_weight_decay if obs_weight_decay is not None else weight_decay
    groups = model.optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
    groups = [group for group in groups if group["params"]]
    names = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    if {id(param) for group in groups for param in group["params"]} != names.keys():
        raise ValueError("Optimizer groups must cover all trainable parameters and only those parameters")

    for group in groups:
        group["param_names"] = [names[id(p)] for p in group["params"]]
    beta1, beta2 = betas
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        fused=next(model.parameters()).is_cuda,
    )


class EMA:
    """Exponential moving average with power-law warmup."""

    def __init__(
        self,
        model: torch.nn.Module,
        ema_model: torch.nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        ema_model.eval()
        ema_model.requires_grad_(False)
        self.parameters = []
        self.ema_parameters = []
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            if param.requires_grad:
                self.parameters.append(param)
                self.ema_parameters.append(ema_param)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

    def get_decay(self, global_step: int) -> float:
        step = max(0, global_step - self.update_after_step - 1)
        if step == 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, global_step: int) -> None:
        decay = self.get_decay(global_step)
        update = get_ema_multi_avg_fn(decay)
        update(self.ema_parameters, self.parameters, global_step)


def build_scheduler(optimizer, name, num_warmup_steps, num_training_steps, *, state=None):
    from diffusers.optimization import get_scheduler as diffusion_scheduler

    scheduler = diffusion_scheduler(
        name, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    if state is not None:
        scheduler.load_state_dict(state)
        # LambdaLR state does not capture the budget held by its new closures.
        rates = [base * fn(scheduler.last_epoch) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate
        scheduler._last_lr = rates
    return scheduler
