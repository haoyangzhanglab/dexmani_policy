"""DDIM adaptation of RTC soft inpainting: https://arxiv.org/abs/2506.07339."""

import math

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


def soft_mask(length, delay, *, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)
    decay = ((length - position) / (length - delay + 1)).clamp(max=1)
    return decay * torch.expm1(decay) / math.expm1(1)


def prepare_schedule(scheduler, steps, beta):
    if not isinstance(scheduler, DDIMScheduler) or scheduler.config.thresholding:
        raise NotImplementedError("RTC supports DDIM without dynamic thresholding only")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("RTC guidance cap must be finite and nonnegative")

    scheduler.set_timesteps(steps)
    alpha = scheduler.alphas_cumprod[scheduler.timesteps]
    if not ((alpha > 0) & (alpha < 1)).all():
        raise ValueError("RTC denoising timesteps require 0 < alpha_bar < 1")
    return alpha


def denoised_estimate(output, sample, alpha, sigma, prediction_type):
    if prediction_type == "sample":
        return output
    if prediction_type == "epsilon":
        return (sample - sigma * output) / alpha
    if prediction_type == "v_prediction":
        return alpha * sample - sigma * output
    raise NotImplementedError(prediction_type)


@torch.inference_mode(False)
@torch.no_grad()
def guided_inference(
    decoder, cond, action_shape, prefix, *, delay, offset, beta, steps, dtype, generator=None
):
    if prefix.ndim != 3 or prefix.shape[0] != action_shape[0]:
        raise ValueError("RTC prefix must be (batch, length, control_dim)")
    length, control_dim = prefix.shape[1:]
    if (
        offset < 0
        or not 0 <= delay <= length <= action_shape[1] - offset
        or not 0 < control_dim <= action_shape[2]
    ):
        raise ValueError("Invalid RTC prefix horizon/control dimensions")

    scheduler = decoder.noise_scheduler
    alpha_bar = prepare_schedule(scheduler, steps, beta)
    timesteps = scheduler.timesteps.to(cond.device)
    previous = scheduler.timesteps - scheduler.config.num_train_timesteps // steps
    alpha_bar_previous = torch.where(
        previous.to(cond.device) >= 0,
        scheduler.alphas_cumprod[previous.clamp(min=0)],
        scheduler.final_alpha_cumprod,
    )
    # Keep schedule coefficients in FP32; BF16 can round alpha_bar to one.
    alpha, sigma = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
    guidance = alpha_bar_previous.sqrt() * sigma - alpha * (1 - alpha_bar_previous).sqrt()
    guidance *= (alpha * sigma).reciprocal().clamp(max=beta)

    # Deployment conditions may be inference tensors, which autograd cannot save.
    cond = cond.detach().clone()
    prefix = prefix.to(device=cond.device, dtype=dtype).detach()
    weights = soft_mask(length, delay, device=cond.device, dtype=dtype)[None, :, None]
    sample = torch.randn(action_shape, device=cond.device, dtype=dtype, generator=generator)

    for i, timestep in enumerate(scheduler.timesteps.tolist()):
        with torch.enable_grad():
            sample.requires_grad_(True)
            output = decoder.model(x=sample, timestep=timesteps[i], context=cond)
            denoised = denoised_estimate(
                output, sample, alpha[i], sigma[i], scheduler.config.prediction_type
            )[:, offset : offset + length, :control_dim]
            residual = weights * (prefix - denoised.detach())
            # The prefix VJP still propagates to the entire action chunk; weights enter once.
            correction = (
                torch.autograd.grad(denoised, sample, grad_outputs=residual, allow_unused=True)[0]
                if denoised.requires_grad else None
            )

        sample = scheduler.step(output.detach(), timestep, sample, generator=generator).prev_sample
        if correction is not None:
            sample.add_(guidance[i] * correction)

    if scheduler.config.clip_sample:
        sample.clamp_(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    return sample
