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
        # Active BatchNorm is rejected at construction; frozen parameters are copied.
        for param, ema_param in zip(new_model.parameters(), self.averaged_model.parameters()):
            value = param.detach().to(dtype=ema_param.dtype)
            if not param.requires_grad:
                ema_param.copy_(value)
            else:
                ema_param.mul_(decay)
                ema_param.add_(value, alpha=1 - decay)

        self.optimization_step += 1
