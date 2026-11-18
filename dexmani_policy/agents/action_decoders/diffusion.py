"""Action diffusion with squared-cosine training noise and DDIM sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from dexmani_policy.common.config import validate_ddim_steps

from .sampling.rtc import guided_ddim, rtc_enabled


class Diffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        clip_sample: bool = True,
    ) -> None:
        super().__init__()
        validate_ddim_steps(num_inference_steps, "num_inference_steps")
        if type(clip_sample) is not bool:
            raise TypeError("agent.clip_sample must be a bool")
        if prediction_type not in ("epsilon", "sample", "v_prediction"):
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        self.model = model
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            timestep_spacing="leading",
            prediction_type=prediction_type,
        )
        self.num_inference_steps = num_inference_steps

    def _training_prediction(
        self, cond: torch.Tensor, samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(samples)
        timestep = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (samples.shape[0],),
            device=samples.device,
        )

        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(samples.device)
        # BF16 can round alpha_bar to one and erase low-noise training steps.
        alpha_bar = self.noise_scheduler.alphas_cumprod[timestep].float().view(-1, 1, 1)
        alpha, sigma = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
        noisy_samples = (alpha * samples + sigma * noise).to(samples.dtype)
        pred = self.model(
            x=noisy_samples,
            timestep=timestep,
            context=cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = samples
        else:
            target = (alpha * noise - sigma * samples).to(samples.dtype)
        return pred, target

    def compute_loss(
        self,
        cond: torch.Tensor,
        actions: torch.Tensor,
        dim_groups: dict[str, tuple[int, int, float]] | None = None,
        *,
        log_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred, target = self._training_prediction(cond, actions)
        loss_dict = {}
        # Each group retains its own mean before applying its explicit weight.
        if dim_groups is not None:
            group_losses = []
            for name, (start, end, weight) in dim_groups.items():
                group_loss = F.mse_loss(pred[..., start:end].float(), target[..., start:end].float())
                if log_metrics:
                    loss_dict[f"loss_{name}"] = group_loss.detach()
                group_losses.append(group_loss * weight)
            loss = torch.stack(group_losses).sum()
        else:
            loss = F.mse_loss(pred.float(), target.float())

        if log_metrics:
            loss_dict["loss"] = loss.detach()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_shape: tuple[int, ...],
        inference_steps: int | None = None,
        *,
        generator: torch.Generator | None = None,
        rtc_prefix=None,
        delay_steps=0,
        history_offset=0,
        rtc_guidance_cap=0.0,
    ) -> torch.Tensor:
        num_steps = self.num_inference_steps if inference_steps is None else inference_steps
        validate_ddim_steps(num_steps, "inference_steps")

        # Scheduler tensors are not module buffers; restored and trained instances need the same device.
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            cond.device
        )
        self.noise_scheduler.final_alpha_cumprod = self.noise_scheduler.final_alpha_cumprod.to(
            cond.device
        )
        if rtc_enabled(rtc_prefix, action_shape, history_offset, rtc_guidance_cap):
            return guided_ddim(
                self,
                cond,
                action_shape,
                rtc_prefix,
                delay=delay_steps,
                offset=history_offset,
                beta=rtc_guidance_cap,
                steps=num_steps,
                generator=generator,
            )
        self.noise_scheduler.set_timesteps(num_steps)
        sample = torch.randn(action_shape, device=cond.device, dtype=torch.float32, generator=generator)
        timesteps = self.noise_scheduler.timesteps.to(sample.device)
        for index, timestep in enumerate(self.noise_scheduler.timesteps.tolist()):
            output = self.model(x=sample, timestep=timesteps[index], context=cond)
            sample = self.noise_scheduler.step(output, timestep, sample, generator=generator).prev_sample

        return sample


class ForceDiffusion(Diffusion):
    """Standard Gaussian diffusion in dataset force coordinates and units."""

    def __init__(
        self,
        model: nn.Module,
        num_training_steps: int = 100,
        num_inference_steps: int = 20,
    ) -> None:
        super().__init__(
            model,
            num_training_steps,
            num_inference_steps,
            prediction_type="sample",
            clip_sample=False,
        )

    def compute_loss_and_prediction(
        self, condition: torch.Tensor, force: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = valid[..., None]
        # A timestep's force prediction can depend on targets at other timesteps.
        force = torch.where(valid, force.float(), 0.0)
        prediction, target = self._training_prediction(condition, force)
        prediction = prediction.float()
        residual = torch.where(valid, prediction - target, 0.0)
        loss = residual.square().sum() / (force.shape[-1] * valid.sum(dtype=torch.float32))
        return loss, prediction
