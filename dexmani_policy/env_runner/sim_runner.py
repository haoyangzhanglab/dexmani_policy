from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dexmani_policy.common.image_util import preprocess_rgb_eval
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.env_runner.video import save_episode_video


class SimRunner:
    def __init__(
        self,
        task_name: str,
        n_obs_steps: int,
        sensor_modalities: list[str],
        rgb_preprocessing: dict[str, Any],
        env_kwargs: dict[str, Any] | None = None,
        env_rebuild_interval: int = 25,
    ):
        self.task_name = task_name
        self.env_kwargs = env_kwargs or {}
        self.env_rebuild_interval = env_rebuild_interval
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = list(sensor_modalities)
        self.rgb_preprocessing = rgb_preprocessing

        self.obs_buffer: dict[str, np.ndarray] = {}
        self.obs_cursor = 0
        self.hand_history: np.ndarray | None = None

    def update_obs(self, observation: dict[str, np.ndarray]):
        position = self.obs_cursor % self.n_obs_steps
        for name in self.sensor_modalities:
            value = observation[name]
            if name not in self.obs_buffer:
                shape = (self.n_obs_steps, *value.shape)
                self.obs_buffer[name] = np.empty(shape, dtype=value.dtype)
            self.obs_buffer[name][position] = value
        self.obs_cursor += 1

    def get_stacked_obs(self) -> dict[str, np.ndarray]:
        if self.obs_cursor < self.n_obs_steps:
            # Pad missing history with the first observed frame.
            indices = np.maximum(
                np.arange(self.n_obs_steps) - self.n_obs_steps + self.obs_cursor, 0
            )
        else:
            indices = (self.obs_cursor + np.arange(self.n_obs_steps)) % self.n_obs_steps

        return {name: buffer[indices] for name, buffer in self.obs_buffer.items()}

    def get_obs_batch(self, device) -> dict[str, torch.Tensor]:
        batch = {}
        for name, observation in self.get_stacked_obs().items():
            value = (
                preprocess_rgb_eval(observation, **self.rgb_preprocessing)
                if name == "rgb" else observation
            )
            dtype = None if name == "rgb" else torch.float32
            batch[name] = torch.as_tensor(value, device=device, dtype=dtype).unsqueeze(0)
        return batch

    def reset(self):
        self.obs_buffer.clear()
        self.obs_cursor = 0
        self.hand_history = None

    @torch.no_grad()
    def predict_action_chunk(
        self, obs_batch, agent, inference_steps: int | None = None, *, action_steps: int, generator=None
    ) -> np.ndarray:
        kwargs: dict[str, Any] = {}
        if self.hand_history is not None:
            history = torch.as_tensor(
                self.hand_history, dtype=torch.float32, device=obs_batch["joint_state"].device
            )
            kwargs["action_context"] = {"hand_history": history.unsqueeze(0)}
        result = agent.predict_action(
            obs_dict=obs_batch, inference_steps=inference_steps, generator=generator, **kwargs
        )
        actions = result["future_actions"][0, :action_steps]
        return np.ascontiguousarray(actions.cpu().numpy())

    def run_episode(
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
        if getattr(agent, "history_steps", 0):
            initial_hand = obs["joint_state"][-agent.hand_dim :]
            self.hand_history = np.repeat(initial_hand[None], agent.history_steps, axis=0)

        device = next(agent.parameters()).device
        # Environment and policy own separate random streams.
        generator = torch.Generator(device=device).manual_seed(episode_seed)

        task_done_step = None

        while True:
            obs_batch = self.get_obs_batch(device=device)
            action_chunk = self.predict_action_chunk(
                obs_batch, agent, inference_steps=inference_steps,
                action_steps=action_steps, generator=generator,
            )
            for action in action_chunk:
                obs, _, done, truncated, info = env.step(action)
                if self.hand_history is not None:
                    # History follows the command actually applied after environment clipping.
                    self.hand_history[:-1] = self.hand_history[1:]
                    self.hand_history[-1] = info["executed_joint_action"][-agent.hand_dim :]
                self.update_obs(obs)

                if info["success_condition"] and task_done_step is None:
                    task_done_step = env.action_cnt

                if done or truncated:
                    return bool(info["success"]), task_done_step

    def run(self, agent, *, seeds, inference_steps, action_steps, video_save_dir=None):
        env = None
        try:
            for index, seed in enumerate(seeds):
                if env is None or (
                    self.env_rebuild_interval > 0 and index % self.env_rebuild_interval == 0
                ):
                    if env is not None:
                        previous, env = env, None
                        previous.close()
                    env = self.make_env(record_video=video_save_dir is not None)

                success, steps = self.run_episode(
                    agent, env, seed, inference_steps, action_steps=action_steps
                )
                video = (
                    save_episode_video(env, video_save_dir, f"episode_{seed}")
                    if video_save_dir is not None else None
                )
                yield {
                    "task_name": self.task_name,
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "total_steps": env.action_cnt,
                    "video": video,
                }
        finally:
            if env is not None:
                close_resources(env.close)

    def make_env(self, *, record_video: bool):
        from dexmani_sim import resolve_env_class

        env_class = resolve_env_class(self.task_name)
        kwargs = {"render_mode": "rgb_array", **self.env_kwargs}
        return env_class(record_video=record_video, **kwargs)
