import torch
import torch.nn as nn
from einops import reduce
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class Diffusion(nn.Module):

    def __init__(
            self, 
            model,
            device,
            num_training_steps = 100,
            num_inference_steps = 10,
            prediction_type = "sample",
    ):
        super().__init__()

        self.model = model
        self.device = device

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


    def compute_loss(self, cond, actions, **kwargs):
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
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")      

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_dict = {"loss": loss}

        return loss, loss_dict


    @torch.no_grad()
    def conditional_sample(self, x0, N, cond):

        traj = []
        traj.append(x0.clone())
        self.noise_scheduler.set_timesteps(N, device=x0.device)

        for t in self.noise_scheduler.timesteps:
            output = self.model(
                x = traj[-1],
                timestep = t,
                context = cond,
            )
            traj.append(self.noise_scheduler.step(output, t, traj[-1]).prev_sample)
        
        return traj
    

    def predict_action(self, cond, action_template, denoise_timesteps=None):
        '''
        sample: zero_data for action seq, in order to get the shape and device
        '''
        noise = torch.randn_like(action_template, device=action_template.device)
        if denoise_timesteps is None:
            denoise_timesteps = self.num_inference_steps
        traj = self.conditional_sample(noise, denoise_timesteps, cond)
        return traj[-1]


