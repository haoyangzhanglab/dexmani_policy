from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseShiftSampler:
    """Noise-shift timestep sampling: biases t toward 0 for harder denoising tasks."""

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def sample(self, batch: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch, device=device)
        return u / (1 + (self.alpha - 1) * (1 - u))


class SimpleRectifiedFlowDecoder(nn.Module):
    """Rectified flow matching with noise-shift sampling and KV-cache inference."""

    def __init__(
        self,
        model: nn.Module,
        denoise_steps: int = 2,
        noise_shift_alpha: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.denoise_steps = denoise_steps
        self.sampler = NoiseShiftSampler(noise_shift_alpha)

    def compute_loss(self, cond, actions, **kwargs):
        B = actions.shape[0]
        noise = torch.randn_like(actions)
        t = self.sampler.sample(B, actions.device)

        xt = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        target = actions - noise

        pred = self.model(x=xt, timestep=t, context=cond)
        loss = F.mse_loss(pred, target)
        return loss, {
            "loss": loss,
            "loss_action": loss,
            "loss_flow": loss,
            "loss_consistency": torch.zeros_like(loss),
            "pred_v_magnitude": torch.sqrt(torch.mean(pred ** 2)),
            "t_mean": t.mean(),
        }

    @torch.no_grad()
    def predict_action(self, cond, action_template, denoise_timesteps=None):
        steps = denoise_timesteps if denoise_timesteps is not None else self.denoise_steps
        B = action_template.shape[0]
        device = action_template.device
        dtype = action_template.dtype

        self.model.setup_kv_cache(cond)
        try:
            x = torch.randn_like(action_template)
            dt = 1.0 / steps
            for i in range(steps):
                ti = torch.full((B,), i * dt, device=device, dtype=dtype)
                v = self.model(x=x, timestep=ti, context=cond)
                x = x + v * dt
        finally:
            self.model.clear_kv_cache()
        return x