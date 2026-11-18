from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_policy.common.config import TACTILE_FEATURE_SHAPES
from dexmani_policy.common.inference import prepare_numeric_tensor, prepare_observations, predict_actions


class SimRunner:
    @staticmethod
    def load_seed_pool(task_name):
        from dexmani_sim import DATA_DIR

        return [int(value) for value in (DATA_DIR / "eval_seeds" / f"{task_name}.txt").read_text().split()]

    def __init__(
        self,
        task_name: str,
        n_obs_steps: int,
        sensor_modalities: list[str],
        rgb_preprocessing: dict[str, Any],
        control_dt_s: float,
        env_kwargs: dict[str, Any] | None = None,
        env_rebuild_interval: int = 25,
    ):
        self.task_name = task_name
        self.control_dt_s = control_dt_s
        self.env_kwargs = env_kwargs or {}
        self.env_rebuild_interval = env_rebuild_interval
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = list(sensor_modalities)
        self.rgb_preprocessing = rgb_preprocessing

        self.obs_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)

    def update_obs(self, observation: dict[str, np.ndarray]):
        # Environments can reuse their observation arrays on the next control step.
        self.obs_buffer.append({name: observation[name].copy() for name in self.sensor_modalities})

    def get_stacked_obs(self) -> dict[str, np.ndarray]:
        history = list(self.obs_buffer)
        history = [history[0]] * (self.n_obs_steps - len(history)) + history
        return {name: np.stack([obs[name] for obs in history]) for name in self.sensor_modalities}

    def get_obs_batch(self, device, *, normalization_dims) -> dict[str, torch.Tensor]:
        return prepare_observations(
            self.get_stacked_obs(), normalization_dims=normalization_dims,
            n_obs_steps=self.n_obs_steps, rgb_preprocessing=self.rgb_preprocessing, device=device,
        )

    def reset(self):
        self.obs_buffer.clear()

    def predict_action_chunk(
        self, obs_batch, agent, inference_steps: int | None = None, *,
        action_steps: int, generator=None, prev_action=None,
    ) -> np.ndarray:
        kwargs = {}
        if prev_action is not None:
            kwargs["prev_action"] = prepare_numeric_tensor(
                prev_action, device=next(agent.parameters()).device, label="previous executed action",
            )
        return predict_actions(
            agent, obs_batch, inference_steps=inference_steps, generator=generator, **kwargs,
        )[:action_steps]

    def _run_episode(
        self,
        agent,
        env,
        episode_seed,
        inference_steps: int | None = None,
        *,
        action_steps: int,
    ) -> tuple[bool, int | None]:
        """Record first raw success separately from the final hold/grace success."""
        obs, _ = env.reset(seed=episode_seed)
        self.reset()
        self.update_obs(obs)
        device = next(agent.parameters()).device
        # Environment and policy own separate random streams.
        generator = torch.Generator(device=device).manual_seed(episode_seed)

        task_done_step = None
        prev_action = None

        while True:
            obs_batch = self.get_obs_batch(device=device, normalization_dims=agent.normalization_dims)
            action_chunk = self.predict_action_chunk(
                obs_batch, agent, inference_steps=inference_steps,
                action_steps=action_steps, generator=generator, prev_action=prev_action,
            )
            for action in action_chunk:
                obs, _, done, truncated, info = env.step(action)
                if agent.requires_prev_action:
                    prev_action = info["executed_joint_action"].copy()
                self.update_obs(obs)

                if info["success_condition"] and task_done_step is None:
                    task_done_step = env.action_cnt

                if done or truncated:
                    return bool(info["success"]), task_done_step

    def run(self, agent, *, seeds, inference_steps, action_steps, video_save_dir=None):
        if agent.requires_prev_action and action_steps != agent.n_action_steps:
            raise ValueError("ODE execution requires action_steps == n_action_steps")
        unsupported = (TACTILE_FEATURE_SHAPES.keys() & set(self.sensor_modalities)) - {"finger_tactile_force"}
        if unsupported:
            raise ValueError(f"Sim has no dense tactile producer for: {sorted(unsupported)}")
        env = None
        try:
            for index, seed in enumerate(seeds):
                if env is None or (
                    self.env_rebuild_interval > 0 and index % self.env_rebuild_interval == 0
                ):
                    if env is not None:
                        try:
                            env.close()
                        finally:
                            env = None
                    env = self.make_env(record_video=video_save_dir is not None)

                success, steps = self._run_episode(
                    agent, env, seed, inference_steps, action_steps=action_steps
                )
                result = {
                    "task_name": self.task_name,
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "total_steps": env.action_cnt,
                    "video": None,
                }
                if video_save_dir is not None:
                    from dexmani_policy.env_runner.video import save_episode_video

                    result["video"] = save_episode_video(
                        env.get_video(), 1 / env.dt, Path(video_save_dir) / f"episode_{seed}.mp4",
                    )
                yield result
        finally:
            if env is not None:
                env.close()

    def _environment_class(self):
        from dexmani_sim import resolve_env_class

        return resolve_env_class(self.task_name)

    def make_env(self, *, record_video: bool):
        env_class = self._environment_class()
        kwargs = {
            "render_mode": "rgb_array", **self.env_kwargs,
            "observation_keys": tuple(self.sensor_modalities),
        }
        env = env_class(record_video=record_video, **kwargs)
        dt = env.dt
        if not np.isclose(dt, self.control_dt_s, rtol=1e-6, atol=1e-9):
            env.close()
            raise ValueError(f"Control period mismatch: Sim dt={dt} s, Policy control_dt_s={self.control_dt_s} s")
        return env
