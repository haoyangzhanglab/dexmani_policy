import torch


class EMAModel:
    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

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

        ema_params = dict(self.averaged_model.named_parameters())
        ema_buffers = dict(self.averaged_model.named_buffers())

        for name, param in new_model.named_parameters():
            ema_param = ema_params.get(name)
            if ema_param is None:
                raise KeyError(f"EMA model missing parameter '{name}'. Model structure may have changed since EMA was initialized.")
            if not param.requires_grad:
                ema_param.copy_(param.to(dtype=ema_param.dtype).data)
            else:
                ema_param.mul_(self.decay).add_(
                    param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                )

        for name, buf in new_model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buf.to(dtype=ema_buffers[name].dtype).data)

        self.optimization_step += 1

    def state_dict(self):
        return {
            'decay': self.decay,
            'optimization_step': self.optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.optimization_step = state_dict['optimization_step']