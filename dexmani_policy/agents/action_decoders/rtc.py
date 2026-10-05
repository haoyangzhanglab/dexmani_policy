"""Inference-only DDIM adaptation; endpoint weights act once in the input VJP."""

import math

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


def prefix_weights(length, delay, *, device, dtype):
    if not 0 <= delay <= length:
        raise ValueError("RTC requires 0 <= delay <= prefix length")
    i = torch.arange(length, device=device, dtype=dtype)
    z = (length - i) / (length - delay + 1)
    return torch.where(i < delay, torch.ones_like(i), z * torch.expm1(z) / math.expm1(1))


def validate_scheduler(scheduler, steps, beta):
    if not isinstance(scheduler, DDIMScheduler) or scheduler.config.thresholding:
        raise NotImplementedError("RTC supports DDIM without dynamic thresholding only")
    if isinstance(beta, bool) or not math.isfinite(beta) or beta < 0:
        raise ValueError("RTC guidance cap must be finite and nonnegative")
    if scheduler.config.prediction_type not in {"sample", "epsilon", "v_prediction"}:
        raise NotImplementedError("Unsupported RTC prediction type")
    scheduler.set_timesteps(steps)
    alpha = scheduler.alphas_cumprod[scheduler.timesteps]
    if not bool(((alpha > 0) & (alpha < 1)).all()):
        raise NotImplementedError("RTC denoising timesteps require 0 < alpha_bar < 1")


def clean_estimate(output, sample, a, b, prediction_type):
    if prediction_type == "sample":
        return output
    if prediction_type == "epsilon":
        return (sample - b * output) / a
    if prediction_type == "v_prediction":
        return a * sample - b * output
    raise NotImplementedError(prediction_type)


def guided_step(model, scheduler, sample, timestep, cond, target, weight, beta, *, final):
    # The outer deployment bridge uses inference_mode. Clone *inside* this scope
    # so tensors saved for backward are ordinary tensors, not inference tensors.
    with torch.inference_mode(False), torch.enable_grad():
        x = sample.detach().clone().requires_grad_(True)
        output = model(x=x, timestep=timestep, context=cond)
        alpha = scheduler.alphas_cumprod[int(timestep)].to(x)
        previous = (
            int(timestep) - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )
        alpha_previous = (
            scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
        ).to(x)
        a, b = alpha.sqrt(), (1 - alpha).sqrt()
        ap, bp = alpha_previous.sqrt(), (1 - alpha_previous).sqrt()
        f = clean_estimate(output, x, a, b, scheduler.config.prediction_type)
        # Only constrained coordinates are evaluated; no artificial aux targets.
        residual = torch.zeros_like(f)
        mask = weight != 0
        residual[mask] = (weight[mask] * (target[mask] - f[mask])).detach()
        g = None
        if f.requires_grad:
            g = torch.autograd.grad(
                f, x, grad_outputs=residual, allow_unused=True, create_graph=False
            )[0]
        if g is None:
            g = torch.zeros_like(x)
        # Preserve 0.27.2 clipping and its original epsilon, including the default
        # use_clipped_model_output=False. eta remains the original default zero.
        base = scheduler.step(output.detach(), timestep, x.detach()).prev_sample
        kappa = torch.minimum(x.new_tensor(beta), 1 / (a * b))
        result = base + (ap * b - a * bp) * kappa * g
        if final and scheduler.config.clip_sample:
            result = result.clamp(
                -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
            )
        return result.detach()


def predict_rtc(decoder, cond, template, prefix, *, delay, offset, beta, steps):
    length, control_dim = prefix.shape[-2:]
    if prefix.ndim != 3 or prefix.shape[0] != template.shape[0]:
        raise ValueError("RTC prefix must be (batch, length, control_dim)")
    if not 0 <= delay <= length <= template.shape[1] - offset or control_dim > template.shape[2]:
        raise ValueError("Invalid RTC prefix horizon/control dimensions")
    validate_scheduler(decoder.noise_scheduler, steps, beta)
    with torch.inference_mode(False), torch.no_grad():
        cond = cond.detach().clone()
        sample = torch.randn_like(template, device=template.device)
        target = torch.zeros_like(template)
        weight = torch.zeros_like(template)
        target[:, offset : offset + length, :control_dim] = prefix
        weight[:, offset : offset + length, :control_dim] = prefix_weights(
            length, delay, device=sample.device, dtype=sample.dtype
        )[None, :, None]
        scheduler = decoder.noise_scheduler
        scheduler.set_timesteps(steps, device=sample.device)
        for i, t in enumerate(scheduler.timesteps):
            sample = guided_step(
                decoder.model,
                scheduler,
                sample,
                t,
                cond,
                target,
                weight,
                beta,
                final=i == len(scheduler.timesteps) - 1,
            )
        return sample
