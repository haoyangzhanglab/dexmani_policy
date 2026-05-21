import torch


class EMAModel:
    # @crowsonkb: power=2/3 for 1M+ steps (0.9999 at 1M), power=3/4 for <1M steps (0.9999 at 215K)
    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
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

        ema_params = dict(self.averaged_model.named_parameters())
        ema_buffers = dict(self.averaged_model.named_buffers())

        for name, param in new_model.named_parameters():
            ema_param = ema_params.get(name)
            if ema_param is None:
                raise RuntimeError(f"EMA model missing parameter '{name}'")

            if not param.requires_grad:
                ema_param.copy_(param.to(dtype=ema_param.dtype).data)
            else:
                ema_param.mul_(self.decay).add_(
                    param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                )

        for name, buf in new_model.named_buffers():
            ema_buf = ema_buffers.get(name)
            if ema_buf is not None:
                ema_buf.copy_(buf.to(dtype=ema_buf.dtype).data)

        self.optimization_step += 1

    def state_dict(self):
        return {
            'decay': self.decay,
            'optimization_step': self.optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.optimization_step = state_dict['optimization_step']