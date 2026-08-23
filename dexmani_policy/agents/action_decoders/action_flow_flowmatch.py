from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureTimeSampler:
    """i.i.d. mixture of NoiseShift and uniform timestep sampling.

    Each sample independently draws from either the shifted distribution
    (with probability ``shifted_ratio``) or the uniform distribution.
    This avoids batch-level forced partitioning and gives smoother
    gradient signals across the full t-range.
    """

    def __init__(self, alpha: float = 4.0, shifted_ratio: float = 0.75):
        self.alpha = alpha
        self.shifted_ratio = shifted_ratio

    def _shift(self, u: torch.Tensor) -> torch.Tensor:
        return u / (1 + (self.alpha - 1) * (1 - u))

    def sample(self, batch: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch, device=device)
        if self.shifted_ratio == 0.0:
            return u
        shifted = self._shift(u)
        if self.shifted_ratio == 1.0:
            return shifted
        use_shifted = torch.rand(batch, device=device) < self.shifted_ratio
        return torch.where(use_shifted, shifted, u)


class SimpleRectifiedFlowDecoder(nn.Module):
    """Rectified flow matching with mixture time sampling and KV-cache inference."""

    def __init__(
        self,
        model: nn.Module,
        denoise_steps: int = 2,
        noise_shift_alpha: float = 4.0,
        noise_shift_ratio: float = 0.75,
        solver: str = "euler",
    ):
        super().__init__()
        if solver not in {"euler", "midpoint"}:
            raise ValueError(f"Unknown solver: {solver}")

        self.model = model
        self.denoise_steps = denoise_steps
        self.solver = solver
        self.sampler = MixtureTimeSampler(
            alpha=noise_shift_alpha,
            shifted_ratio=noise_shift_ratio,
        )

    def _resolve_nfe(self, denoise_timesteps=None) -> int:
        nfe = self.denoise_steps if denoise_timesteps is None else denoise_timesteps
        nfe = int(nfe)
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}")
        return nfe

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

    def _sample_euler(
        self, x: torch.Tensor, cond: torch.Tensor, nfe: int
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        dt = 1.0 / nfe

        for i in range(nfe):
            timestep = torch.full(
                (batch_size,),
                i * dt,
                device=x.device,
                dtype=x.dtype,
            )
            velocity = self.model(x=x, timestep=timestep, context=cond)
            x = x + dt * velocity

        return x

    def _sample_midpoint(
        self, x: torch.Tensor, cond: torch.Tensor, nfe: int
    ) -> torch.Tensor:
        if nfe % 2 != 0:
            raise ValueError(f"midpoint solver requires an even NFE, got {nfe}")

        batch_size = x.shape[0]
        num_macro_steps = nfe // 2
        dt = 1.0 / num_macro_steps

        for i in range(num_macro_steps):
            t0 = i * dt
            t_start = torch.full(
                (batch_size,),
                t0,
                device=x.device,
                dtype=x.dtype,
            )
            k1 = self.model(x=x, timestep=t_start, context=cond)

            x_mid = x + 0.5 * dt * k1
            t_mid = torch.full(
                (batch_size,),
                t0 + 0.5 * dt,
                device=x.device,
                dtype=x.dtype,
            )
            k2 = self.model(x=x_mid, timestep=t_mid, context=cond)
            x = x + dt * k2

        return x

    @torch.no_grad()
    def predict_action(self, cond, action_template, denoise_timesteps=None):
        nfe = self._resolve_nfe(denoise_timesteps)
        x = torch.randn_like(action_template)

        self.model.setup_kv_cache(cond)
        try:
            if self.solver == "euler":
                x = self._sample_euler(x, cond, nfe)
            else:
                x = self._sample_midpoint(x, cond, nfe)
        finally:
            self.model.clear_kv_cache()
        return x
