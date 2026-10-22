"""Rectified and consistency flow matching with independent training grids and inference NFE."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_int

from .sampling.time_sampling import TimeSampler, euler_time_grid, shift_time_to_noise


def interpolate_flow(noise, target, timestep):
    return (1.0 - timestep) * noise + timestep * target


class RectifiedFlow(nn.Module):
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
        time_shift_alpha = float(time_shift_alpha)
        if not math.isfinite(time_shift_alpha) or time_shift_alpha < 1.0:
            raise ValueError("time_shift_alpha must be finite and >= 1")

        self.model = model
        self.num_inference_steps = num_inference_steps
        self.num_flow_train_timesteps = num_flow_train_timesteps
        self.t_sample_mode = t_sample_mode
        self.time_shift_alpha = time_shift_alpha
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
        log_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        xt = interpolate_flow(x0, x1, t_view)
        target_v = x1 - x0

        pred_v = self.model(x=xt, timestep=t, context=cond)
        loss = F.mse_loss(pred_v, target_v)

        if not log_metrics:
            return loss, {}

        with torch.no_grad():
            metrics = {
                "loss": loss.detach(),
                "loss_action": loss.detach(),
                "pred_v_magnitude": torch.sqrt(torch.mean(pred_v**2)),
                "target_v_magnitude": torch.sqrt(torch.mean(target_v**2)),
                "t_mean": t.mean(),
            }
        return loss, metrics

    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_steps = (
            self.num_inference_steps
            if inference_steps is None
            else positive_int(inference_steps, "inference_steps")
        )

        batch_size = action_template.shape[0]
        x = torch.randn_like(action_template)
        dt = 1.0 / num_steps

        context_kwargs = {}
        prepare = getattr(self.model, "prepare_context_kv", None)
        if not self.training and prepare is not None:
            context_kv = prepare(cond)
            if context_kv is not None:
                context_kwargs["context_kv"] = context_kv
        for timestep in euler_time_grid(num_steps, x):
            t = timestep.expand(batch_size)
            velocity = self.model(x=x, timestep=t, context=cond, **context_kwargs)
            x = x + dt * velocity

        return x


class ConsistencyFlow(nn.Module):
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
        positive_int(num_inference_steps, "num_inference_steps")
        if not 0 < flow_batch_ratio < 1:
            raise ValueError("flow_batch_ratio must be between 0 and 1")
        if target_t_sample_mode not in {"relative", "absolute"}:
            raise ValueError("target_t_sample_mode must be either 'relative' or 'absolute'")

        self.model = model
        self.num_inference_steps = num_inference_steps
        self.denoise_timesteps = denoise_timesteps
        self.flow_batch_ratio = flow_batch_ratio
        self.t_sample_mode_for_flow = t_sample_mode_for_flow
        self.t_sample_mode_for_consistency = t_sample_mode_for_consistency
        self.dt_sample_mode_for_consistency = dt_sample_mode_for_consistency
        self.target_t_sample_mode = target_t_sample_mode

        self.time_sampler = TimeSampler(num_steps=denoise_timesteps)

    def sample_flow_targets(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        t = self.time_sampler.sample(
            batch_size,
            self.t_sample_mode_for_flow,
            device=actions.device,
        )
        t_view = t.view(batch_size, *([1] * (actions.ndim - 1)))
        dt = torch.zeros((batch_size,), device=actions.device, dtype=t.dtype)
        target_t = dt if self.target_t_sample_mode == "relative" else t

        x0 = torch.randn_like(actions)
        x1 = actions
        return {
            "xt": interpolate_flow(x0, x1, t_view),
            "t": t,
            "target_t": target_t,
            "vt_target": x1 - x0,
        }

    def sample_consistency_targets(
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
        xt = interpolate_flow(x0, x1, t_view)
        xt_next = interpolate_flow(x0, x1, t_next_view)

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
        log_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if ema_decoder is None:
            raise RuntimeError("ConsistencyFlow requires an EMA action decoder during training")

        ema_model = ema_decoder.model
        batch_size = actions.shape[0]

        if batch_size < 2:
            raise ValueError(
                "ConsistencyFlow requires micro-batch >= 2 for flow/consistency groups; "
                "gradient accumulation cannot repair this. Use RectifiedFlow for pure flow."
            )

        flow_batch_size = max(1, min(batch_size - 1, int(batch_size * self.flow_batch_ratio)))
        consistency_batch_size = batch_size - flow_batch_size

        flow_targets = self.sample_flow_targets(actions[:flow_batch_size])
        consistency_targets = self.sample_consistency_targets(
            actions[flow_batch_size:],
            cond[flow_batch_size:],
            ema_model,
        )

        pred = self.model(
            x=torch.cat([flow_targets["xt"], consistency_targets["xt"]]),
            timestep=torch.cat([flow_targets["t"], consistency_targets["t"]]),
            target_t=torch.cat([flow_targets["target_t"], consistency_targets["target_t"]]),
            context=cond,
        )
        pred_flow = pred[:flow_batch_size]
        pred_consistency = pred[flow_batch_size:]

        loss_flow = F.mse_loss(pred_flow, flow_targets["vt_target"])
        loss_consistency = F.mse_loss(pred_consistency, consistency_targets["vt_target"])

        # Sample proportions are not loss weights.
        loss = loss_flow + loss_consistency
        if not log_metrics:
            return loss, {}

        with torch.no_grad():
            metrics = {
                "loss": loss.detach(),
                "loss_action": loss.detach(),
                "loss_flow": loss_flow.detach(),
                "loss_consistency": loss_consistency.detach(),
                "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_flow**2)),
                "pred_vt_consistency_magnitude": torch.sqrt(torch.mean(pred_consistency**2)),
                "flow_batch_size": flow_batch_size,
                "consistency_batch_size": consistency_batch_size,
                "t_flow_mean": flow_targets["t"].mean(),
                "t_consistency_mean": consistency_targets["t"].mean(),
            }
        return loss, metrics

    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_steps = (
            self.num_inference_steps
            if inference_steps is None
            else positive_int(inference_steps, "inference_steps")
        )

        batch_size = action_template.shape[0]
        x = torch.randn_like(action_template)
        dt = 1.0 / num_steps
        context_kv = self.model.prepare_context_kv(cond) if num_steps > 1 else None
        timesteps = euler_time_grid(num_steps, x)
        target_times = (
            torch.full_like(timesteps, dt)
            if self.target_t_sample_mode == "relative"
            else timesteps + dt
        )

        for timestep, target_time in zip(timesteps, target_times):
            t = timestep.expand(batch_size)
            target_t = target_time.expand(batch_size)
            velocity = self.model(
                x=x,
                timestep=t,
                target_t=target_t,
                context=cond,
                context_kv=context_kv,
            )
            x = x + dt * velocity

        return x
