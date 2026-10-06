"""Consistency Flow Matching decoder used by ManiFlow."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from dexmani_policy.agents.action_decoders.time_sampler import TimeSampler
from dexmani_policy.utils.validation import positive_int
from dexmani_policy.agents.action_decoders.utils import resolve_inference_steps


class ConsistencyFlowMatch(nn.Module):
    """Rectified flow plus the ManiFlow EMA consistency-training protocol.

    ``denoise_timesteps`` sets the training grid; ``num_inference_steps``
    sets only the default Euler inference NFE.
    """

    requires_ema_for_loss = True

    def __init__(
        self,
        model: nn.Module,
        num_inference_steps: int = 10,
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",
    ) -> None:
        super().__init__()
        if type(denoise_timesteps) is not int or denoise_timesteps <= 0:
            raise ValueError("denoise_timesteps must be a positive integer")
        positive_int(num_inference_steps, "num_inference_steps")
        if not 0 < flow_batch_ratio < 1:
            raise ValueError("flow_batch_ratio must be between 0 and 1")
        if target_t_sample_mode not in {"relative", "absolute"}:
            raise ValueError(
                "target_t_sample_mode must be either 'relative' or 'absolute'"
            )

        self.model = model
        self.num_inference_steps = num_inference_steps
        self.denoise_timesteps = denoise_timesteps
        self.flow_batch_ratio = flow_batch_ratio
        self.t_sample_mode_for_flow = t_sample_mode_for_flow
        self.t_sample_mode_for_consistency = t_sample_mode_for_consistency
        self.dt_sample_mode_for_consistency = dt_sample_mode_for_consistency
        self.target_t_sample_mode = target_t_sample_mode
        # Training grid resolution is independent of the Euler inference NFE.
        self.time_sampler = TimeSampler(num_steps=denoise_timesteps)

    @staticmethod
    def linear_interpolate(
        noise: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return (1.0 - timestep) * noise + timestep * target

    def get_flow_velocity(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        t = self.time_sampler.sample(
            batch_size,
            self.t_sample_mode_for_flow,
            device=actions.device,
        )
        t_view = t.view(batch_size, *([1] * (actions.ndim - 1)))
        dt = torch.zeros((batch_size,), device=actions.device, dtype=t.dtype)
        target_t = (
            dt
            if self.target_t_sample_mode == "relative"
            else torch.clamp(t + dt, max=1.0)
        )

        x0 = torch.randn_like(actions)
        x1 = actions
        return {
            "xt": self.linear_interpolate(x0, x1, t_view),
            "t": t,
            "target_t": target_t,
            "vt_target": x1 - x0,
        }

    def get_consistency_velocity(
        self,
        actions: torch.Tensor,
        cond: torch.Tensor,
        ema_model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        t = self.time_sampler.sample(
            batch_size,
            self.t_sample_mode_for_consistency,
            device=actions.device,
        )
        dt = self.time_sampler.sample(
            batch_size,
            self.dt_sample_mode_for_consistency,
            device=actions.device,
        )
        t_next = torch.clamp(t + dt, max=1.0)

        if self.target_t_sample_mode == "relative":
            target_t_next = dt
            target_t = dt
        else:
            target_t_next = t_next + dt
            target_t = t_next

        t_view = t.view(batch_size, *([1] * (actions.ndim - 1)))
        t_next_view = t_next.view(batch_size, *([1] * (actions.ndim - 1)))

        x0 = torch.randn_like(actions)
        x1 = actions
        xt = self.linear_interpolate(x0, x1, t_view)
        xt_next = self.linear_interpolate(x0, x1, t_next_view)

        with torch.no_grad():
            v_to_target = ema_model(
                x=xt_next,
                timestep=t_next,
                target_t=target_t_next,
                context=cond,
            )

        pred_x1 = xt_next + v_to_target * (1.0 - t_next_view)
        denominator = (1.0 - t_view).clamp_min(1e-6)
        vt_target = (pred_x1 - xt) / denominator

        return {
            "xt": xt,
            "t": t,
            "target_t": target_t,
            "vt_target": vt_target,
        }

    def compute_loss(
        self,
        cond: torch.Tensor,
        actions: torch.Tensor,
        *,
        ema_decoder: nn.Module | None = None,
        dim_groups=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if dim_groups:
            raise ValueError("This flow decoder does not support nonempty dim_groups")
        if ema_decoder is None:
            raise RuntimeError(
                "ConsistencyFlowMatch requires an EMA action decoder during training"
            )
        if not hasattr(ema_decoder, "model"):
            raise TypeError("EMA action decoder must expose its backbone as '.model'")

        ema_model = ema_decoder.model
        batch_size = actions.shape[0]

        if batch_size < 2:
            raise ValueError("ConsistencyFlowMatch requires micro-batch >= 2 for flow/consistency groups; "
                             "gradient accumulation cannot repair this. Use RectifiedFlow for pure flow.")

        flow_batch_size = max(
            1,
            min(batch_size - 1, int(batch_size * self.flow_batch_ratio)),
        )
        consistency_batch_size = batch_size - flow_batch_size

        self.time_sampler.validate_batch(flow_batch_size, self.t_sample_mode_for_flow)
        self.time_sampler.validate_batch(consistency_batch_size, self.t_sample_mode_for_consistency)
        self.time_sampler.validate_batch(consistency_batch_size, self.dt_sample_mode_for_consistency)
        flow_targets = self.get_flow_velocity(actions[:flow_batch_size])
        consistency_targets = self.get_consistency_velocity(
            actions[flow_batch_size:],
            cond[flow_batch_size:],
            ema_model,
        )

        x_merged = torch.cat(
            [flow_targets["xt"], consistency_targets["xt"]],
            dim=0,
        )
        t_merged = torch.cat(
            [flow_targets["t"], consistency_targets["t"]],
            dim=0,
        )
        target_t_merged = torch.cat(
            [flow_targets["target_t"], consistency_targets["target_t"]],
            dim=0,
        )

        pred = self.model(
            x=x_merged,
            timestep=t_merged,
            target_t=target_t_merged,
            context=cond,
        )
        pred_flow = pred[:flow_batch_size]
        pred_consistency = pred[flow_batch_size:]

        loss_flow = F.mse_loss(
            pred_flow,
            flow_targets["vt_target"],
            reduction="none",
        )
        loss_flow = reduce(loss_flow, "b ... -> b (...)", "mean").mean()
        loss_consistency = F.mse_loss(
            pred_consistency,
            consistency_targets["vt_target"],
            reduction="none",
        )
        loss_consistency = reduce(
            loss_consistency,
            "b ... -> b (...)",
            "mean",
        ).mean()

        loss = loss_flow + loss_consistency
        return loss, {
            "loss": loss,
            "loss_action": loss,
            "loss_flow": loss_flow,
            "loss_consistency": loss_consistency,
            "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_flow**2)),
            "pred_vt_consistency_magnitude": torch.sqrt(
                torch.mean(pred_consistency**2)
            ),
            "flow_batch_size": flow_batch_size,
            "consistency_batch_size": consistency_batch_size,
            "t_flow_mean": flow_targets["t"].mean(),
            "t_consistency_mean": consistency_targets["t"].mean(),
            "has_consistency": 1,
        }

    @torch.no_grad()
    def sample_ode(
        self,
        x0: torch.Tensor,
        num_steps: int,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        positive_int(num_steps, "inference_steps")

        batch_size = x0.shape[0]
        x = x0
        dt = 1.0 / num_steps
        context_kv = self.model.prepare_context_kv(cond)

        for step in range(num_steps):
            t = torch.full(
                (batch_size,),
                step * dt,
                device=x.device,
                dtype=x.dtype,
            )
            target_t = (
                torch.full(
                    (batch_size,),
                    dt,
                    device=x.device,
                    dtype=x.dtype,
                )
                if self.target_t_sample_mode == "relative"
                else t + dt
            )
            velocity = self.model(
                x=x,
                timestep=t,
                target_t=target_t,
                context=cond,
                context_kv=context_kv,
            )
            x = x + dt * velocity

        return x

    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample actions with a per-call NFE override; training grid is unchanged."""
        num_steps = resolve_inference_steps(self.num_inference_steps, inference_steps)
        noise = torch.randn_like(action_template)
        return self.sample_ode(noise, num_steps, cond)
