import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    Exponential Moving Average of models weights.

    @crowsonkb's notes on EMA Warmup:
        If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3
        are good values for models you plan to train for a million or more steps
        (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
        gamma=1, power=3/4 for models you plan to train for less (reaches decay
        factor 0.999 at 10K steps, 0.9999 at 215.4k steps).
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False)
            ):
                if isinstance(module, _BatchNorm):
                    # BatchNorm affine params (weight/bias) are copied directly
                    # rather than EMA-averaged.  The running_mean / running_var
                    # inside the EMA model are *not* updated during training
                    # (the EMA model stays in eval mode), so they will diverge
                    # from the main model's running stats over time.  This is
                    # acceptable for the current codebase because the action
                    # decoder backbones (UNet1D / DiT / DiTX / OneWayTransformer)
                    # do not use BatchNorm.
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        self.optimization_step += 1
