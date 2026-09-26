"""Standard conditional rectified-flow action decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.common.inference import positive_int, resolve_inference_steps

from dexmani_policy.agents.action_decoders.time_sampler import (
    TimeSampler,
    shift_time_to_noise,
)


class RectifiedFlow(nn.Module):
    """Standard conditional rectified flow with fixed-step Euler inference.

    ``num_flow_train_timesteps`` controls discrete training sampling only;
    ``num_inference_steps`` is the independent default Euler NFE.
    """

    requires_ema_for_loss = False

    def __init__(
        self,
        model: nn.Module,
        num_inference_steps: int = 4,
        t_sample_mode: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        time_shift_alpha: float = 1.0,
        num_flow_train_timesteps: int = 4,
    ) -> None:
        super().__init__()
        positive_int(num_inference_steps, "num_inference_steps")
        positive_int(num_flow_train_timesteps, "num_flow_train_timesteps")
        if time_shift_alpha < 1.0:
            raise ValueError("time_shift_alpha must be >= 1")

        self.model = model
        self.num_inference_steps = num_inference_steps
        self.num_flow_train_timesteps = num_flow_train_timesteps
        self.t_sample_mode = t_sample_mode
        self.time_shift_alpha = float(time_shift_alpha)
        self.time_sampler = TimeSampler(
            num_steps=num_flow_train_timesteps,
            beta_s=beta_s,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )

    def compute_loss(
        self,
        cond: torch.Tensor,
        actions: torch.Tensor,
        *,
        dim_groups=None,
        model_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del dim_groups
        model_kwargs = {} if model_kwargs is None else model_kwargs

        batch_size = actions.shape[0]
        x0 = torch.randn_like(actions)
        x1 = actions

        t = self.time_sampler.sample(
            batch_size,
            self.t_sample_mode,
            device=actions.device,
        )
        t = shift_time_to_noise(t, self.time_shift_alpha)
        t_view = t.view(batch_size, *([1] * (actions.ndim - 1)))

        xt = (1.0 - t_view) * x0 + t_view * x1
        target_v = x1 - x0

        pred_v = self.model(
            x=xt,
            timestep=t,
            context=cond,
            **model_kwargs,
        )
        loss = F.mse_loss(pred_v, target_v)

        return loss, {
            "loss": loss,
            "loss_action": loss,
            "pred_v_magnitude": torch.sqrt(torch.mean(pred_v**2)),
            "target_v_magnitude": torch.sqrt(torch.mean(target_v**2)),
            "t_mean": t.mean(),
        }

    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_steps = resolve_inference_steps(self.num_inference_steps, inference_steps)

        batch_size = action_template.shape[0]
        x = torch.randn_like(action_template)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full(
                (batch_size,),
                step * dt,
                device=x.device,
                dtype=x.dtype,
            )
            velocity = self.model(x=x, timestep=t, context=cond)
            x = x + dt * velocity

        return x
