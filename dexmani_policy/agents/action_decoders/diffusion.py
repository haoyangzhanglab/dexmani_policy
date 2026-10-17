"""Action diffusion with squared-cosine training noise and DDIM sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from dexmani_policy.utils.validation import positive_int


class Diffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        aux_loss_weight: float = 1.0,
        clip_sample: bool = True,
    ) -> None:
        super().__init__()
        positive_int(num_inference_steps, "num_inference_steps")

        self.model = model
        self.aux_loss_weight = aux_loss_weight
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=prediction_type,
        )
        self.num_inference_steps = num_inference_steps

    def compute_loss(
        self,
        cond: torch.Tensor,
        actions: torch.Tensor,
        dim_groups: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        noise = torch.randn_like(actions)
        timestep = torch.randint(
            self.noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=actions.device
        )

        pred = self.model(
            x=self.noise_scheduler.add_noise(actions, noise, timestep),
            timestep=timestep,
            context=cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = actions
        elif pred_type == "v_prediction":
            # add_noise places the schedule on-device; v-target coefficients stay in FP32.
            alpha_bar = self.noise_scheduler.alphas_cumprod[timestep].float()
            alpha_t = alpha_bar.sqrt()
            sigma_t = (1 - alpha_bar).sqrt()
            target = alpha_t.view(-1, 1, 1) * noise - sigma_t.view(-1, 1, 1) * actions
            target = target.to(dtype=actions.dtype)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # R3D averages each action group separately and weights only auxiliary losses.
        if dim_groups is not None:
            group_losses = []
            loss_dict = {}
            for name, (start, end) in dim_groups.items():
                group_loss = F.mse_loss(pred[..., start:end], target[..., start:end])
                loss_dict[f"loss_{name}"] = group_loss.detach()
                if name != "joint":
                    group_loss = group_loss * self.aux_loss_weight
                group_losses.append(group_loss)
            loss = sum(group_losses)
            loss_dict["loss_action"] = loss.detach()
        else:
            loss = F.mse_loss(pred, target)
            loss_dict = {"loss_action": loss.detach()}

        loss_dict["loss"] = loss.detach()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        inference_steps: int | None = None,
        *,
        rtc_prefix=None,
        delay_steps=0,
        history_offset=0,
        rtc_guidance_cap=0.0,
    ) -> torch.Tensor:
        num_steps = (
            self.num_inference_steps
            if inference_steps is None
            else positive_int(inference_steps, "inference_steps")
        )

        # Scheduler tensors are not module buffers; restored and trained instances need the same device.
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            action_template.device
        )
        self.noise_scheduler.final_alpha_cumprod = self.noise_scheduler.final_alpha_cumprod.to(
            action_template.device
        )
        if rtc_prefix is not None and rtc_prefix.shape[-2] and rtc_guidance_cap != 0:
            from dexmani_policy.agents.action_decoders.sampling.rtc import guided_inference

            return guided_inference(
                self,
                cond,
                action_template,
                rtc_prefix,
                delay=delay_steps,
                offset=history_offset,
                beta=rtc_guidance_cap,
                steps=num_steps,
            )
        sample = torch.randn_like(action_template)
        self.noise_scheduler.set_timesteps(num_steps)
        timesteps = self.noise_scheduler.timesteps.to(sample.device)
        for index, timestep in enumerate(self.noise_scheduler.timesteps.tolist()):
            output = self.model(x=sample, timestep=timesteps[index], context=cond)
            sample = self.noise_scheduler.step(output, timestep, sample).prev_sample

        return sample
