"""RTC soft inpainting for flow matching and DDIM: https://arxiv.org/abs/2506.07339."""

import math

import torch

from .time_sampling import euler_time_grid


def rtc_enabled(prefix, action_shape, delay, offset, beta):
    if prefix is None:
        return False
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("RTC guidance cap must be finite and nonnegative")
    if prefix.ndim != 3 or prefix.shape[0] != action_shape[0]:
        raise ValueError("RTC prefix must be (batch, length, control_dim)")
    length, control_dim = prefix.shape[1:]
    if (
        offset < 0
        or not 0 <= delay <= length <= action_shape[1] - offset
        or not 0 < control_dim <= action_shape[2]
    ):
        raise ValueError("Invalid RTC prefix horizon/control dimensions")
    return length > 0 and beta > 0


def soft_mask(length, delay, *, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)
    decay = ((length - position) / (length - delay + 1)).clamp(max=1)
    return decay * torch.expm1(decay) / math.expm1(1)


def denoised_estimate(output, sample, alpha, sigma, prediction_type):
    if prediction_type == "sample":
        return output
    if prediction_type == "epsilon":
        return (sample - sigma * output) / alpha
    if prediction_type == "v_prediction":
        return alpha * sample - sigma * output
    raise NotImplementedError(prediction_type)


def prefix_correction(denoised, sample, prefix, weights, offset):
    denoised = denoised[:, offset : offset + prefix.shape[1], : prefix.shape[2]]
    residual = weights * (prefix - denoised.detach())
    # Weight the residual once; the VJP can affect the entire action chunk.
    return torch.autograd.grad(denoised, sample, grad_outputs=residual)[0]


@torch.inference_mode(False)
@torch.no_grad()
def guided_ddim(
    decoder, cond, action_shape, prefix, *, delay, offset, beta, steps, dtype, generator=None
):
    scheduler = decoder.noise_scheduler
    scheduler.set_timesteps(steps)
    alpha_bar = scheduler.alphas_cumprod[scheduler.timesteps]
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
    weights = soft_mask(prefix.shape[1], delay, device=cond.device, dtype=dtype)[None, :, None]
    sample = torch.randn(action_shape, device=cond.device, dtype=dtype, generator=generator)

    for i, timestep in enumerate(scheduler.timesteps.tolist()):
        with torch.enable_grad():
            sample.requires_grad_(True)
            output = decoder.model(x=sample, timestep=timesteps[i], context=cond)
            denoised = denoised_estimate(
                output, sample, alpha[i], sigma[i], scheduler.config.prediction_type
            )
            correction = prefix_correction(denoised, sample, prefix, weights, offset)

        sample = scheduler.step(output.detach(), timestep, sample, generator=generator).prev_sample
        sample.add_(guidance[i] * correction)

    if scheduler.config.clip_sample:
        sample.clamp_(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    return sample


@torch.inference_mode(False)
@torch.no_grad()
def guided_flow(
    decoder, cond, action_shape, prefix, *, delay, offset, beta, steps, dtype,
    generator=None, target_t_sample_mode=None,
):
    cond = cond.detach().clone()
    prefix = prefix.to(device=cond.device, dtype=dtype).detach()
    weights = soft_mask(prefix.shape[1], delay, device=cond.device, dtype=dtype)[None, :, None]
    sample = torch.randn(action_shape, device=cond.device, dtype=dtype, generator=generator)
    dt = 1.0 / steps
    timesteps = euler_time_grid(steps, sample)
    # Official RTC coefficient; clipping also resolves the t=0 singularity.
    guidance = ((timesteps.square() + (1 - timesteps).square()) / (timesteps * (1 - timesteps)))
    guidance = guidance.clamp(max=beta)

    model_kwargs = {}
    prepare = getattr(decoder.model, "prepare_context_kv", None)
    if not decoder.training and steps > 1 and prepare is not None:
        model_kwargs["context_kv"] = prepare(cond)

    for timestep, strength in zip(timesteps, guidance):
        t = timestep.expand(action_shape[0])
        if target_t_sample_mode is not None:
            model_kwargs["target_t"] = torch.full_like(t, dt) if target_t_sample_mode == "relative" else t + dt
        with torch.enable_grad():
            sample.requires_grad_(True)
            velocity = decoder.model(x=sample, timestep=t, context=cond, **model_kwargs)
            denoised = sample + (1 - timestep) * velocity
            correction = prefix_correction(denoised, sample, prefix, weights, offset)
        sample = sample + dt * (velocity + strength * correction)

    return sample
