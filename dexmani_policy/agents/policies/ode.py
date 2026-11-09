from __future__ import annotations

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.ode import ODE
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet
from dexmani_policy.agents.policies.base import Policy


class ODEEncoder(nn.Module):
    def __init__(self, pc_dim, pc_out_dim, num_points, n_obs_steps, fps_random_config):
        super().__init__()
        self.pc_encoder = PointNet(input_channels=pc_dim, output_channels=pc_out_dim)
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config
        self.out_dim = n_obs_steps * pc_out_dim

    def forward(self, obs):
        pc = preprocess_point_cloud(
            obs["point_cloud"], self.num_points, self.fps_random_config, training=self.training,
        )
        return self.pc_encoder(pc).unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class ODEPolicy(Policy):
    supported_action_keys = ("action",)
    supports_rtc = False
    requires_prev_action = True

    def __init__(
        self, horizon, n_obs_steps, n_action_steps, action_dim,
        pc_dim=6, pc_out_dim=128, state_dim=19, num_points=1024,
        fps_random_config=None, cond_dim=512, hidden_dim=512, num_layers=4,
        solver="euler", vel_scale=0.05, perturb_prob=0.5, perturb_decay=0.7,
        perturb_std=0.01, side_head=0, eps_center=True,
    ):
        if horizon - n_obs_steps + 1 != n_action_steps:
            raise ValueError("ODE requires horizon - n_obs_steps + 1 == n_action_steps")
        obs_encoder = ODEEncoder(pc_dim, pc_out_dim, num_points, n_obs_steps, fps_random_config)
        decoder = ODE(
            obs_encoder.out_dim, action_dim, state_dim, n_action_steps,
            cond_dim=cond_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            solver=solver, vel_scale=vel_scale, perturb_prob=perturb_prob,
            perturb_decay=perturb_decay, side_head=side_head, eps_center=eps_center,
        )
        super().__init__(obs_encoder, decoder, horizon, n_obs_steps, n_action_steps, action_dim)
        std = torch.as_tensor(perturb_std, dtype=torch.float32).expand(action_dim).clone()
        if not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("perturb_std must be finite and nonnegative")
        self.register_buffer("perturb_std", std)

    def _condition_state(self, obs):
        normalized = self._preprocess_obs(obs)
        condition = self.obs_encoder(normalized)
        state = normalized["joint_state"].unflatten(0, (-1, self.n_obs_steps))[:, -1]
        return condition, state

    def compute_loss(self, batch, *, log_metrics=True, **kwargs):
        condition, state = self._condition_state(batch["obs"])
        field = self.normalizer["action"]
        a0 = field.normalize(batch["prev_action"])
        target = field.normalize(batch["action"][:, self.action_start_index:])
        return self.action_decoder.compute_loss(
            condition, a0, state, target, perturb_std=self.perturb_std * field.scale,
            log_metrics=log_metrics, **kwargs,
        )

    @torch.no_grad()
    def predict_action(
        self, obs_dict: dict, inference_steps: int | None = None, *,
        generator: torch.Generator | None = None, prev_action: torch.Tensor | None = None,
        rtc_prefix: torch.Tensor | None = None, delay_steps: int = 0, rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if rtc_prefix is not None:
            raise NotImplementedError("ODEPolicy does not support RTC")
        condition, state = self._condition_state(obs_dict)
        if prev_action is None:
            prev_action = obs_dict["joint_state"][:, -1]
        field = self.normalizer["action"]
        prediction = self.action_decoder.predict_action(
            condition, field.normalize(prev_action), state,
            inference_steps=inference_steps, generator=generator,
        )
        return {"future_actions": field.unnormalize(prediction)}
