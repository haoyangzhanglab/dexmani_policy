"""Continuous-joint adaptation of NTUMARS/ODE-POLICY.

Source: bcea95a299dc3fa8b2a571603a5e8b50e1c449e9,
ode_policy/policies/ode/ode_image_policy.py.
Integration accumulates normalized joint actions in FP32.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_int


class FiLMResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.conditioning = nn.Linear(cond_dim, 2 * dim)

    def forward(self, hidden, scale_bias):
        scale, bias = scale_bias
        return hidden + F.silu(self.linear(hidden) * (1 + scale) + bias)


class VectorField(nn.Module):
    def __init__(self, cond_dim, state_dim, hidden_dim, num_layers, action_dim):
        super().__init__()
        self.input_projection = nn.Linear(action_dim + state_dim, hidden_dim)
        self.layers = nn.ModuleList(
            FiLMResidualBlock(hidden_dim, cond_dim) for _ in range(num_layers)
        )
        self.output_projection = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def prepare_conditioning(self, condition):
        return [layer.conditioning(condition).chunk(2, dim=-1) for layer in self.layers]

    def forward(self, action, state, conditioning, *, return_features=False):
        hidden = self.input_projection(torch.cat((action, state), dim=-1))
        for layer, scale_bias in zip(self.layers, conditioning):
            hidden = layer(hidden, scale_bias)
        return hidden if return_features else self.output_projection(hidden)


class ODE(nn.Module):
    def __init__(
        self, obs_dim, action_dim, state_dim, n_action_steps, *,
        cond_dim=512, hidden_dim=512, num_layers=4, solver="euler",
        vel_scale=0.05, perturb_prob=0.5, perturb_decay=0.7,
        side_head=0, eps_center=True,
    ):
        super().__init__()
        if solver not in {"euler", "rk4"}:
            raise ValueError("solver must be euler or rk4")
        if not math.isfinite(vel_scale) or vel_scale <= 0:
            raise ValueError("vel_scale must be finite and positive")
        if not 0 <= perturb_prob <= 1 or not 0 <= perturb_decay <= 1:
            raise ValueError("perturb_prob and perturb_decay must be in [0, 1]")
        if type(side_head) is not int or side_head < 0:
            raise ValueError("side_head must be a nonnegative integer")

        self.model = VectorField(cond_dim, state_dim, hidden_dim, num_layers, action_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(obs_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
        )
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.solver = solver
        self.vel_scale = vel_scale
        self.perturb_prob = perturb_prob
        self.perturb_decay = perturb_decay
        self.num_components = side_head
        self.eps_center = eps_center
        self.min_spread, self.min_mixture_width = 0.02, 0.05

        if side_head:
            trajectory_dim = n_action_steps * action_dim
            self.side_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.SiLU(),
                nn.Linear(256, trajectory_dim + side_head * (1 + 2 * action_dim)),
            )
            nn.init.zeros_(self.side_head[-1].weight)
            nn.init.zeros_(self.side_head[-1].bias)
            mean_start = trajectory_dim + side_head
            mean_end = mean_start + side_head * action_dim
            nn.init.normal_(self.side_head[-1].bias[mean_start:mean_end])

    def integrate(self, initial_action, state, conditioning, *, substeps=1, step_offset=None):
        action = initial_action.float()
        trajectory = []
        step_size = 1.0 / substeps

        for index in range(self.n_action_steps):
            offset = None if step_offset is None else step_offset[:, index]

            def velocity(current_action):
                update = self.vel_scale * self.model(current_action, state, conditioning).float()
                return update if offset is None else update + offset

            for _ in range(substeps):
                if self.solver == "euler":
                    action = action + step_size * velocity(action)
                else:
                    k1 = velocity(action)
                    k2 = velocity(action + 0.5 * step_size * k1)
                    k3 = velocity(action + 0.5 * step_size * k2)
                    k4 = velocity(action + step_size * k3)
                    action = action + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(action)

        return torch.stack(trajectory, dim=1)

    def _predict_diversity(self, initial_action, state, conditioning):
        features = self.model(initial_action, state, conditioning, return_features=True).detach()
        output = self.side_head(features).float()
        trajectory_dim = self.n_action_steps * self.action_dim
        mixture_start = trajectory_dim + self.num_components

        rate = self.vel_scale * F.softplus(output[:, :trajectory_dim]).view(
            -1, self.n_action_steps, self.action_dim,
        )
        log_weights = F.log_softmax(output[:, trajectory_dim:mixture_start], dim=-1)
        means, raw_widths = output[:, mixture_start:].view(
            -1, 2, self.num_components, self.action_dim,
        ).unbind(dim=1)
        widths = self.min_mixture_width + F.softplus(raw_widths + 0.54)
        return rate, log_weights, means, widths

    @staticmethod
    def _sample_noise(log_weights, means, widths, *, center, generator):
        components = torch.multinomial(log_weights.exp(), 1, generator=generator)[:, 0]
        batch_indices = torch.arange(len(components), device=components.device)
        selected_means = means[batch_indices, components]
        if center:
            return selected_means
        noise = torch.randn(
            means[:, 0].shape, device=means.device, dtype=means.dtype, generator=generator,
        )
        return selected_means + widths[batch_indices, components] * noise

    def compute_loss(self, condition, initial_action, state, target, *, perturb_std, log_metrics=True):
        conditioning = self.model.prepare_conditioning(self.condition_encoder(condition))
        initial_action, target = initial_action.float(), target.float()

        if self.training and self.perturb_prob > 0:
            perturb_mask = (
                torch.rand(initial_action.shape[0], 1, device=initial_action.device) < self.perturb_prob
            ).to(initial_action.dtype)
            delta = torch.randn_like(initial_action) * perturb_std * perturb_mask
            decay = self.perturb_decay ** torch.arange(
                1, self.n_action_steps + 1, device=initial_action.device, dtype=initial_action.dtype,
            )
            initial_action = initial_action + delta
            target = target + delta[:, None, :] * decay[None, :, None]

        prediction = self.integrate(initial_action, state, conditioning)
        path_loss = F.mse_loss(prediction, target) / self.vel_scale ** 2
        loss = path_loss
        metrics = {"loss_path": path_loss.detach()} if log_metrics else {}

        if self.num_components:
            rate, log_weights, means, widths = self._predict_diversity(initial_action, state, conditioning)
            spread = rate.cumsum(dim=1) / self.vel_scale + self.min_spread
            error = ((prediction - target) / self.vel_scale).detach()
            target_noise = -(spread.detach() * error).sum(dim=1) / (spread.detach() ** 2).sum(dim=1)
            log_density = (
                -0.5 * ((target_noise[:, None] - means) / widths) ** 2 - widths.log()
            ).sum(dim=-1)
            side_loss = (
                (0.5 * (error / spread) ** 2 + spread.log()).mean()
                - torch.logsumexp(log_weights + log_density, dim=1).mean()
            )
            loss = loss + side_loss
            if log_metrics:
                metrics["loss_diversity"] = side_loss.detach()

        if log_metrics:
            metrics["loss"] = loss.detach()
        return loss, metrics

    @torch.no_grad()
    def predict_action(self, condition, initial_action, state, inference_steps=None, *, generator=None):
        substeps = 1 if inference_steps is None else positive_int(inference_steps, "inference_steps")
        conditioning = self.model.prepare_conditioning(self.condition_encoder(condition))
        step_offset = None
        if self.num_components:
            rate, log_weights, means, widths = self._predict_diversity(initial_action, state, conditioning)
            noise = self._sample_noise(log_weights, means, widths, center=self.eps_center, generator=generator)
            step_offset = rate * noise[:, None]
        return self.integrate(initial_action, state, conditioning, substeps=substeps, step_offset=step_offset)
