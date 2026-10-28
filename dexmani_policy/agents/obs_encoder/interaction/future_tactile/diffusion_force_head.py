import torch
import torch.nn.functional as F

from dexmani_policy.agents.action_decoders.diffusion import Diffusion


class DiffusionForceHead(Diffusion):
    def __init__(
        self,
        model: torch.nn.Module,
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
        self, condition: torch.Tensor, force: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(force)
        timestep = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (force.shape[0],),
            device=force.device,
        )
        prediction = self.model(
            x=self.noise_scheduler.add_noise(force, noise, timestep),
            timestep=timestep,
            context=condition,
        )
        loss = F.mse_loss(prediction.float(), force.float())
        return loss, prediction
