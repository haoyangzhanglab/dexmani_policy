import torch
from torch.optim.swa_utils import get_ema_multi_avg_fn


def build_optimizer(
    model,
    lr: float,
    weight_decay: float,
    obs_lr: float | None = None,
    obs_weight_decay: float | None = None,
    betas: tuple[float, float] = (0.95, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Require each trainable parameter once and retain its name for resume checks."""
    obs_lr = obs_lr if obs_lr is not None else lr
    obs_weight_decay = obs_weight_decay if obs_weight_decay is not None else weight_decay
    groups = model.optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
    groups = [group for group in groups if group["params"]]
    names = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    param_ids = [id(param) for group in groups for param in group["params"]]
    if len(param_ids) != len(set(param_ids)) or set(param_ids) != names.keys():
        raise ValueError("Optimizer groups must contain every trainable parameter exactly once")

    for group in groups:
        group["param_names"] = [names[id(p)] for p in group["params"]]
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=next(model.parameters()).is_cuda,
    )


class EMA:
    """Exponential moving average with power-law warmup."""

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = parameter.data.float()
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        if step == 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        decay = self.get_decay(self.optimization_step)
        # Frozen parameters are copied at initialization and remain unchanged.
        groups: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for param, ema_param in zip(new_model.parameters(), self.averaged_model.parameters()):
            if not param.requires_grad:
                continue
            targets, sources = groups.setdefault((ema_param.device, ema_param.dtype), ([], []))
            targets.append(ema_param)
            sources.append(param.detach().to(device=ema_param.device, dtype=ema_param.dtype))
        update = get_ema_multi_avg_fn(decay)
        for targets, sources in groups.values():
            update(targets, sources, self.optimization_step)

        self.optimization_step += 1


def build_scheduler(optimizer, name, num_warmup_steps, num_training_steps):
    from diffusers.optimization import get_scheduler as diffusion_scheduler

    return diffusion_scheduler(
        name, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
