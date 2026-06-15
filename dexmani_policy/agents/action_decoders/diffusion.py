import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class Diffusion(nn.Module):
    """Denoising diffusion probabilistic model for action prediction.

    Wraps a backbone (UNet or DiT) with a DDIM noise scheduler for training
    and inference. Supports three prediction types controlled by
    ``prediction_type``:

    - ``"sample"``: model predicts the clean action directly (x0-prediction)
    - ``"epsilon"``: model predicts the noise added to the action
    - ``"v_prediction"``: model predicts v = sqrt(alpha)*noise - sqrt(1-alpha)*action
      (velocity prediction, see https://arxiv.org/abs/2202.00512)

    Training: adds noise to actions, model predicts target, MSE loss.
    Inference: DDIM iterative denoising from random noise (default 10 steps).

    The noise scheduler uses a fixed configuration (squaredcos_cap_v2 beta
    schedule, beta_start=0.0001, beta_end=0.02) that is intentionally
    non-configurable — see CLAUDE.md "Known Hard-coded Values" for rationale.
    """
    def __init__(
        self,
        model: nn.Module,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
    ) -> None:
        super().__init__()

        self.model = model
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=prediction_type,
        )
        self.num_inference_steps = num_inference_steps
        self._prediction_type = prediction_type
        self._cached_alphas_device = None


    def compute_loss(
        self, cond: torch.Tensor, actions: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = actions.shape[0]

        noise = torch.randn_like(actions, device=actions.device)
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device).long()
        noisy_action = self.noise_scheduler.add_noise(actions, noise, timestep)

        pred = self.model(
            x=noisy_action,
            timestep=timestep,
            context=cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = actions
        elif pred_type == 'v_prediction':
            # v_t = sqrt(alpha_cumprod) * noise - sqrt(1-alpha_cumprod) * x_0
            # See: https://arxiv.org/abs/2202.00512 (Progressive Distillation)
            if self._cached_alphas_device != actions.device:
                self._cached_alphas = self.noise_scheduler.alphas_cumprod.to(device=actions.device)
                self._cached_alphas_device = actions.device
            alpha_t = self._cached_alphas[timestep].sqrt()
            sigma_t = (1 - self._cached_alphas[timestep]).sqrt()
            target = alpha_t.view(-1, 1, 1) * noise - sigma_t.view(-1, 1, 1) * actions
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        loss_dict = {"loss_action": loss, "loss": loss}

        return loss, loss_dict


    @torch.no_grad()
    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        denoise_timesteps: int | None = None,
    ) -> torch.Tensor:
        sample = torch.randn_like(action_template, device=action_template.device)
        if denoise_timesteps is None:
            denoise_timesteps = self.num_inference_steps

        self.noise_scheduler.set_timesteps(denoise_timesteps, device=sample.device)
        for t in self.noise_scheduler.timesteps:
            output = self.model(x=sample, timestep=t, context=cond)
            sample = self.noise_scheduler.step(output, t, sample).prev_sample

        return sample


