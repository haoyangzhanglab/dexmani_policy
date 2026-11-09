"""Continuous-joint adaptation of NTUMARS/ODE-POLICY.

Source: bcea95a299dc3fa8b2a571603a5e8b50e1c449e9,
ode_policy/policies/ode/ode_image_policy.py.
Changes: normalized decoder boundary, FP32 accumulation, explicit generator,
shared FiLM projections, and removal of image/gripper/ablation branches.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_int


class _FiLMLayer(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, h, gb):
        g, b = gb
        return h + F.silu(self.lin(h) * (1 + g) + b)


class VectorField(nn.Module):
    def __init__(self, cond_dim, state_dim, hidden_dim, num_layers, action_dim):
        super().__init__()
        self.inp = nn.Linear(action_dim + state_dim, hidden_dim)
        self.layers = nn.ModuleList([_FiLMLayer(hidden_dim, cond_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def precompute(self, c):
        return [layer.film(c).chunk(2, dim=-1) for layer in self.layers]

    def forward(self, a, s, film, *, return_features=False):
        h = self.inp(torch.cat([a, s], dim=-1))
        for layer, gb in zip(self.layers, film):
            h = layer(h, gb)
        return h if return_features else self.out(h)


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
        self.cond_net = nn.Sequential(
            nn.Linear(obs_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
        )
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.solver = solver
        self.vel_scale = vel_scale
        self.perturb_prob = perturb_prob
        self.perturb_decay = perturb_decay
        self.side_k = side_head
        self.eps_center = eps_center
        self.spread_min, self.mix_min = 0.02, 0.05
        if side_head:
            n_amp = n_action_steps * action_dim
            self.side_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.SiLU(),
                nn.Linear(256, n_amp + side_head * (1 + 2 * action_dim)),
            )
            nn.init.zeros_(self.side_head[-1].weight)
            nn.init.zeros_(self.side_head[-1].bias)
            nn.init.normal_(self.side_head[-1].bias[n_amp + side_head:
                                                 n_amp + side_head * (1 + action_dim)])

    def integrate(self, a0, state, film, *, substeps=1, step_offset=None):
        a, out = a0.float(), []
        h = 1.0 / substeps
        for i in range(self.n_action_steps):
            def f(x):
                velocity = self.vel_scale * self.model(x, state, film).float()
                return velocity if step_offset is None else velocity + step_offset[:, i]

            for _ in range(substeps):
                if self.solver == "euler":
                    a = a + h * f(a)
                else:
                    k1 = f(a)
                    k2 = f(a + 0.5 * h * k1)
                    k3 = f(a + 0.5 * h * k2)
                    k4 = f(a + h * k3)
                    a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(a)
        return torch.stack(out, dim=1)

    def _side(self, a0, state, film):
        features = self.model(a0, state, film, return_features=True).detach()
        out = self.side_head(features).float()
        k, p, n = self.side_k, self.action_dim, self.n_action_steps
        rate = self.vel_scale * F.softplus(out[:, :n * p]).view(-1, n, p)
        mean, raw = out[:, n * p + k:].view(-1, 2, k, p).unbind(dim=1)
        return (rate, F.log_softmax(out[:, n * p:n * p + k], dim=-1), mean,
                self.mix_min + F.softplus(raw + 0.54))

    @staticmethod
    def _draw_eps(logw, mean, width, *, center, generator):
        k = torch.multinomial(logw.exp(), 1, generator=generator)[:, 0]
        i = torch.arange(len(k), device=k.device)
        if center:
            return mean[i, k]
        noise = torch.randn(mean[:, 0].shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean[i, k] + width[i, k] * noise

    def compute_loss(self, condition, a0, state, target, *, perturb_std, log_metrics=True):
        c = self.cond_net(condition)
        film = self.model.precompute(c)
        a0, target = a0.float(), target.float()
        if self.training and self.perturb_prob > 0:
            on = (torch.rand(a0.shape[0], 1, device=a0.device) < self.perturb_prob).to(a0.dtype)
            delta = torch.randn_like(a0) * perturb_std * on
            decay = self.perturb_decay ** torch.arange(
                1, self.n_action_steps + 1, device=a0.device, dtype=a0.dtype,
            )
            a0 = a0 + delta
            target = target + delta[:, None, :] * decay[None, :, None]
        prediction = self.integrate(a0, state, film)
        path_loss = F.mse_loss(prediction, target) / self.vel_scale ** 2
        loss = path_loss
        metrics = {"loss_path": path_loss.detach()} if log_metrics else {}
        if self.side_k:
            rate, logw, mean, width = self._side(a0, state, film)
            sig = rate.cumsum(dim=1) / self.vel_scale + self.spread_min
            err = ((prediction - target) / self.vel_scale).detach()
            eps1 = -(sig.detach() * err).sum(dim=1) / (sig.detach() ** 2).sum(dim=1)
            logp = (-0.5 * ((eps1[:, None] - mean) / width) ** 2 - width.log()).sum(dim=-1)
            side_loss = ((0.5 * (err / sig) ** 2 + sig.log()).mean()
                         - torch.logsumexp(logw + logp, dim=1).mean())
            loss = loss + side_loss
            if log_metrics:
                metrics["loss_diversity"] = side_loss.detach()
        if log_metrics:
            metrics["loss"] = loss.detach()
        return loss, metrics

    @torch.no_grad()
    def predict_action(self, condition, a0, state, inference_steps=None, *, generator=None):
        substeps = 1 if inference_steps is None else positive_int(inference_steps, "inference_steps")
        c = self.cond_net(condition)
        film = self.model.precompute(c)
        step_offset = None
        if self.side_k:
            rate, logw, mean, width = self._side(a0, state, film)
            eps = self._draw_eps(logw, mean, width, center=self.eps_center, generator=generator)
            step_offset = rate * eps[:, None]
        return self.integrate(a0, state, film, substeps=substeps, step_offset=step_offset)
