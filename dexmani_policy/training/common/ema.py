import torch


class EMA:
    """Exponential moving average with a power-law decay warmup."""

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        foreach=False,
    ):
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.foreach = foreach
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        if step == 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)
        buckets = {}
        # Active BatchNorm is rejected at construction; frozen parameters are copied.
        for param, ema_param in zip(new_model.parameters(), self.averaged_model.parameters()):
            value = param.detach().to(dtype=ema_param.dtype)
            if not param.requires_grad:
                ema_param.copy_(value)
            elif self.foreach:
                targets, values = buckets.setdefault((ema_param.device, ema_param.dtype), ([], []))
                targets.append(ema_param)
                values.append(value)
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(value, alpha=1 - self.decay)

        for targets, values in buckets.values():
            torch._foreach_mul_(targets, self.decay)
            torch._foreach_add_(targets, values, alpha=1 - self.decay)
        self.optimization_step += 1
