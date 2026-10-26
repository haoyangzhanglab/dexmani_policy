import torch


class EMA:
    """Exponential moving average with power-law warmup or a fixed decay."""

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        fixed_decay=None,
    ):
        if fixed_decay is not None and not 0 <= fixed_decay < 1:
            raise ValueError("fixed_decay must be finite and in [0,1)")
        self.fixed_decay = fixed_decay
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
        if self.fixed_decay is not None:
            return self.fixed_decay
        step = max(0, optimization_step - self.update_after_step - 1)
        if step == 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        decay = self.get_decay(self.optimization_step)
        # Fresh EMA copies raw weights; restored frozen parameters were checked before training.
        for param, ema_param in zip(new_model.parameters(), self.averaged_model.parameters()):
            if not param.requires_grad:
                continue
            value = param.detach().to(dtype=ema_param.dtype)
            ema_param.mul_(decay)
            ema_param.add_(value, alpha=1 - decay)

        self.optimization_step += 1
