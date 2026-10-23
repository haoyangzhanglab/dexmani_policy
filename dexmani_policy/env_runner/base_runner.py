"""Single-task rollout with observation windows and physical action execution."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
from termcolor import cprint

from dexmani_policy.common.config import positive_int, resolve_action_steps
from dexmani_policy.common.rgb_preprocessing import preprocess_validation_rgb
from dexmani_policy.env_runner.common.metrics import summarize_episodes, validate_episode_details
from dexmani_policy.env_runner.common.seeds import validate_task_seeds
from dexmani_policy.env_runner.common.video import save_episode_video


class BaseRunner:
    """Subclasses supply an environment and physical evaluation seeds."""

    def __init__(
        self,
        n_obs_steps: int,
        sensor_modalities: list[str] | None = None,
        clear_cache_freq: int = 25,
        env_video_fps: int | None = None,
    ):
        self.n_obs_steps = positive_int(n_obs_steps, "n_obs_steps")
        self.sensor_modalities = sensor_modalities or ["point_cloud", "joint_state"]
        self.obs_buffer: dict[str, np.ndarray | torch.Tensor] = {}
        self.obs_cursor = 0
        self.obs_count = 0
        self.hand_history = None

        self.env_video_fps = env_video_fps
        self.clear_cache_freq = clear_cache_freq
        self.rgb_preprocessing = None

    @property
    def task_runners(self):
        return {self.task_name: self}

    def validate_observation(self, observation: dict[str, Any]):
        missing = set(self.sensor_modalities) - observation.keys()
        if missing:
            raise ValueError(f"Missing required observation frame fields: {sorted(missing)}")
        # Check the whole frame before writes; NumPy assignment can silently broadcast.
        for name in self.sensor_modalities:
            value = observation[name]
            if name in self.obs_buffer:
                buffer = self.obs_buffer[name]
                if (
                    type(value) is not type(buffer)
                    or value.shape != buffer.shape[1:]
                    or value.dtype != buffer.dtype
                    or (
                        torch.is_tensor(value)
                        and (value.device != buffer.device or value.layout != torch.strided)
                    )
                ):
                    raise ValueError(f"Observation {name} changed type, shape, dtype or device")
            elif isinstance(value, np.ndarray):
                if value.dtype.kind not in "buif":
                    raise TypeError(f"Observation {name} must be a numeric array")
            elif torch.is_tensor(value):
                if value.layout != torch.strided or value.is_complex():
                    raise TypeError(f"Observation {name} must be a dense real tensor")
            else:
                raise TypeError(f"Unsupported observation type for {name}: {type(value).__name__}")

    def update_obs(self, observation: dict[str, Any]):
        self.validate_observation(observation)
        position = self.obs_cursor % self.n_obs_steps
        for name in self.sensor_modalities:
            value = observation[name]
            if name not in self.obs_buffer:
                shape = (self.n_obs_steps, *value.shape)
                self.obs_buffer[name] = (
                    np.empty(shape, dtype=value.dtype)
                    if isinstance(value, np.ndarray)
                    else value.new_empty(shape)
                )
            self.obs_buffer[name][position] = value
        self.obs_cursor += 1
        self.obs_count = min(self.obs_count + 1, self.n_obs_steps)

    def get_stacked_obs(self) -> dict[str, Any]:
        """Return independent chronological copies, padding the first observed frame."""
        if self.obs_count == 0:
            raise RuntimeError("No observation in buffer")
        if self.obs_count < self.n_obs_steps:
            indices = np.maximum(np.arange(self.n_obs_steps) - self.n_obs_steps + self.obs_count, 0)
        else:
            indices = (self.obs_cursor + np.arange(self.n_obs_steps)) % self.n_obs_steps
        stacked = {}
        tensor_indices = {}
        for name, buffer in self.obs_buffer.items():
            if isinstance(buffer, np.ndarray):
                stacked[name] = buffer[indices]
            else:
                if buffer.device not in tensor_indices:
                    tensor_indices[buffer.device] = torch.as_tensor(indices, device=buffer.device)
                stacked[name] = buffer[tensor_indices[buffer.device]]
        return stacked

    def get_obs_batch(self, device) -> dict[str, Any]:
        batch = {}
        for name, value in self.get_stacked_obs().items():
            if name == "rgb":
                if self.rgb_preprocessing is None:
                    raise ValueError("RGB runner requires the saved Dataset preprocessing recipe")
                value = preprocess_validation_rgb(value, **self.rgb_preprocessing)
            dtype = None if name in {"rgb", "tactile_valid"} else torch.float32
            batch[name] = torch.as_tensor(value, device=device, dtype=dtype).unsqueeze(0)
        return batch

    def reset(self):
        self.obs_buffer.clear()
        self.obs_cursor = 0
        self.obs_count = 0
        self.hand_history = None

    @torch.no_grad()
    def get_action_chunk(
        self, obs_batch, agent, inference_steps: int | None = None, *, action_steps: int
    ) -> np.ndarray:
        kwargs = {}
        if getattr(agent, "history_steps", 0):
            if self.hand_history is None:
                raise RuntimeError("Hand command history must be initialized at episode reset")
            history = torch.as_tensor(
                self.hand_history, dtype=torch.float32, device=obs_batch["joint_state"].device
            )
            kwargs["action_context"] = {"hand_history": history.unsqueeze(0)}
        result = agent.predict_action(obs_dict=obs_batch, inference_steps=inference_steps, **kwargs)
        prediction = result["pred_action"]
        if prediction.shape != (1, agent.horizon, agent.action_dim):
            raise ValueError("pred_action must contain one complete physical action trajectory")
        start = agent.n_obs_steps - 1
        actions = prediction[0, start : start + action_steps, : agent.control_action_dim]
        return np.ascontiguousarray(actions.detach().cpu().numpy())

    def run_one_episode(
        self,
        agent,
        env,
        episode_seed,
        inference_steps: int | None = None,
        *,
        action_steps: int,
    ):
        """Record first raw success separately from the final hold/grace success."""
        obs, _ = env.reset(seed=episode_seed)
        self.reset()
        self.update_obs(obs)
        if getattr(agent, "history_steps", 0):
            initial_hand = np.asarray(obs["joint_state"])[-agent.hand_dim :]
            if initial_hand.shape != (agent.hand_dim,) or not np.isfinite(initial_hand).all():
                raise ValueError("LAMP reset requires a finite measured hand pose")
            self.hand_history = np.repeat(initial_hand[None], agent.history_steps, axis=0)

        # Environment reset does not seed the policy sampling generator.
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)

        done = False
        truncated = False
        episode_success = False
        task_done_step = None
        device = next(agent.parameters()).device

        while not (done or truncated):
            obs_batch = self.get_obs_batch(device=device)
            action_chunk = self.get_action_chunk(
                obs_batch, agent, inference_steps=inference_steps, action_steps=action_steps
            )
            for action in action_chunk:
                obs, _, done, truncated, info = env.step(action)
                if self.hand_history is not None:
                    # History follows the command actually applied after environment clipping.
                    executed = np.asarray(info.get("executed_joint_action"), dtype=np.float32)
                    if executed.shape != (agent.action_dim,) or not np.isfinite(executed).all():
                        raise ValueError("LAMP requires env.step info['executed_joint_action']")
                    self.hand_history[:-1] = self.hand_history[1:]
                    self.hand_history[-1] = executed[-agent.hand_dim :]
                self.update_obs(obs)

                if info.get("success_condition") and task_done_step is None:
                    task_done_step = getattr(env, "action_cnt", None)

                if info.get("success", False):
                    episode_success = True

                if done or truncated:
                    break

        return episode_success, task_done_step

    def run(
        self,
        agent,
        *,
        task_seeds: dict[str, list[int]],
        inference_steps: int | None = None,
        action_steps: int | None = None,
        video_save_dir: Path | None = None,
    ):
        details = []
        videos = []
        env = None
        task = self.task_name

        try:
            action_steps = resolve_action_steps(agent, action_steps)
            if inference_steps is not None:
                positive_int(inference_steps, "inference_steps")
            count = validate_task_seeds(task_seeds, self.task_runners)
            seeds = task_seeds[task]

            for index, seed in enumerate(seeds):
                if env is None or (
                    self.clear_cache_freq > 0 and index % self.clear_cache_freq == 0
                ):
                    if env is not None:
                        with suppress(Exception):
                            env.close()
                        env = None
                    env = self.make_env()
                    video_fps = (
                        self.env_video_fps
                        if self.env_video_fps is not None
                        else getattr(env, "video_fps", 15)
                    )
                try:
                    success, steps = self.run_one_episode(
                        agent, env, seed, inference_steps, action_steps=action_steps
                    )
                except Exception:
                    cprint(f"Evaluation failed: task={task}, seed={seed}", "red")
                    save_episode_video(env, video_save_dir, f"episode_{seed}_crash", video_fps)
                    raise
                details.append(
                    {
                        "task_name": task,
                        "seed": seed,
                        "success": success,
                        "steps": steps,
                        "total_steps": getattr(env, "action_cnt", None),
                    }
                )
                video = save_episode_video(env, video_save_dir, f"episode_{seed}", video_fps)
                if video is not None:
                    videos.append(video)
                status = "success" if success else "fail"
                cprint(
                    f"[progress {index + 1}/{count}] env seed: {seed}, status: {status}, "
                    f"done step: {steps if steps is not None else 'N/A'}",
                    "cyan",
                )

            validate_episode_details(details, {task: seeds})
            stats = summarize_episodes(details)
            cprint(
                f"[result] Episodes: {count}, Success rate: {stats['success_rate']:.1%}, "
                f"Avg steps (success): {stats['avg_steps']}, Avg steps (all): {stats['avg_steps_all']}",
                "yellow",
            )
            return {
                "episode_details": details,
                "success_rate": stats["success_rate"],
                "avg_steps": stats["avg_steps"],
                "avg_steps_all": stats["avg_steps_all"],
                "videos": videos,
            }
        finally:
            if env is not None:
                with suppress(Exception):
                    env.close()

    def make_env(self):
        raise NotImplementedError

    def get_seed_list(self) -> list[int]:
        raise NotImplementedError
