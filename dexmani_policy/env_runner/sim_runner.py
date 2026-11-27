from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_policy.common.config import TACTILE_FEATURE_SHAPES, positive_int
from dexmani_policy.common.io_util import cleanup_after_error, cleanup_on_exit
from dexmani_policy.deployment import prepare_observations, predict_actions


def resolve_action_steps(steps: int | None, *, n_action_steps: int, future_steps: int) -> int:
    steps = positive_int(n_action_steps if steps is None else steps, "action_steps")
    if steps > future_steps:
        raise ValueError(f"action_steps must not exceed future_steps ({future_steps})")
    return steps


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
        *,
        skip_intermediate_observations: bool = False,
    ):
        self.task_name = task_name
        self.control_dt_s = control_dt_s
        self.env_kwargs = env_kwargs or {}
        self.env_rebuild_interval = env_rebuild_interval
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = list(sensor_modalities)
        self.rgb_preprocessing = rgb_preprocessing
        self.skip_intermediate_observations = skip_intermediate_observations

        self.obs_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)

    def update_obs(self, observation: dict[str, np.ndarray]):
        # Environments can reuse their observation arrays on the next control step.
        self.obs_buffer.append({name: observation[name].copy() for name in self.sensor_modalities})

    def get_stacked_obs(self) -> dict[str, np.ndarray]:
        history = list(self.obs_buffer)
        history = [history[0]] * (self.n_obs_steps - len(history)) + history
        return {name: np.stack([obs[name] for obs in history]) for name in self.sensor_modalities}

    def reset(self):
        self.obs_buffer.clear()

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
        skip_observations = self.skip_intermediate_observations and action_steps > self.n_obs_steps
        if skip_observations:
            from dexmani_sim import TACTILE_OBSERVATION_KEYS

            collect_tactile = not set(self.sensor_modalities).isdisjoint(TACTILE_OBSERVATION_KEYS)

        while True:
            obs_batch = prepare_observations(
                self.get_stacked_obs(), normalization_dims=agent.normalization_dims,
                n_obs_steps=self.n_obs_steps, rgb_preprocessing=self.rgb_preprocessing, device=device,
            )
            action_chunk = predict_actions(
                agent, obs_batch, inference_steps=inference_steps, generator=generator,
            )[:action_steps]
            skip_steps = len(action_chunk) - self.n_obs_steps if skip_observations else 0
            for index, action in enumerate(action_chunk):
                if index < skip_steps:
                    _, _, done, truncated, info = env.step_physics(action, collect_tactile=collect_tactile)
                else:
                    obs, _, done, truncated, info = env.step(action)
                    self.update_obs(obs)
                if info["success_condition"] and task_done_step is None:
                    task_done_step = env.action_cnt

                if done or truncated:
                    return bool(info["success"]), task_done_step

    def run(self, agent, *, seeds, inference_steps, action_steps, video_save_dir=None):
        if self.skip_intermediate_observations and video_save_dir is not None:
            raise ValueError("Skipping intermediate observations requires video=False")
        action_steps = resolve_action_steps(
            action_steps, n_action_steps=agent.n_action_steps, future_steps=agent.future_steps,
        )
        unsupported = (TACTILE_FEATURE_SHAPES.keys() & set(self.sensor_modalities)) - {"finger_tactile_force"}
        if unsupported:
            raise ValueError(f"Sim has no dense tactile producer for: {sorted(unsupported)}")
        with ExitStack() as resources:
            for index, seed in enumerate(seeds):
                if index == 0 or (
                    self.env_rebuild_interval > 0 and index % self.env_rebuild_interval == 0
                ):
                    resources.close()
                    env = self.make_env(record_video=video_save_dir is not None)
                    resources.enter_context(cleanup_on_exit(env.close))

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

    def _make_env(self, *, record_video: bool):
        from dexmani_sim import resolve_env_class

        env_class = resolve_env_class(self.task_name)
        kwargs = {
            "render_mode": "rgb_array", **self.env_kwargs,
            "observation_keys": tuple(self.sensor_modalities),
        }
        return env_class(record_video=record_video, **kwargs)

    def make_env(self, *, record_video: bool):
        env = self._make_env(record_video=record_video)
        try:
            self._validate_env(env)
        except BaseException as error:
            cleanup_after_error(env.close, error)
            raise
        return env

    def _validate_env(self, env):
        dt = env.dt
        if not np.isclose(dt, self.control_dt_s, rtol=1e-6, atol=1e-9):
            raise ValueError(f"Control period mismatch: Sim dt={dt} s, Policy control_dt_s={self.control_dt_s} s")
