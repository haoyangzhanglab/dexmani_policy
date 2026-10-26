"""Single-task rollout with observation windows and physical action execution."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.runtime import close_resources
from dexmani_policy.common.rgb_preprocessing import preprocess_validation_rgb
from dexmani_policy.env_runner.common.video import save_episode_video


class SimRunner:
    """Environment lifecycle and one task's explicit episode sequence."""

    def __init__(
        self,
        task_name: str,
        n_obs_steps: int,
        sensor_modalities: list[str],
        env_kwargs: dict[str, Any] | None = None,
        clear_cache_freq: int = 25,
        env_video_fps: int | None = None,
        record_video: bool = True,
        render_mode: str = "rgb_array",
        viewer_resolution: tuple[int, int] | None = None,
    ):
        self.n_obs_steps = positive_int(n_obs_steps, "n_obs_steps")
        self.sensor_modalities = list(sensor_modalities)
        self.obs_buffer: dict[str, np.ndarray | torch.Tensor] = {}
        self.obs_cursor = 0
        self.hand_history = None

        self.env_video_fps = env_video_fps
        self.clear_cache_freq = clear_cache_freq
        self.rgb_preprocessing = None

        self.task_name = task_name
        self.env_kwargs = env_kwargs or {}
        self.record_video = record_video
        self.render_mode = render_mode
        self.viewer_resolution = viewer_resolution

    def update_obs(self, observation: dict[str, Any]):
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

    def get_stacked_obs(self) -> dict[str, Any]:
        """Return independent chronological copies, padding the first observed frame."""
        if self.obs_cursor == 0:
            raise RuntimeError("No observation in buffer")
        if self.obs_cursor < self.n_obs_steps:
            indices = np.maximum(
                np.arange(self.n_obs_steps) - self.n_obs_steps + self.obs_cursor, 0
            )
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
        self.hand_history = None

    @torch.no_grad()
    def predict_action_chunk(
        self, obs_batch, agent, inference_steps: int | None = None, *, action_steps: int, generator=None
    ) -> np.ndarray:
        kwargs = {}
        if getattr(agent, "history_steps", 0):
            if self.hand_history is None:
                raise RuntimeError("Hand command history must be initialized at episode reset")
            history = torch.as_tensor(
                self.hand_history, dtype=torch.float32, device=obs_batch["joint_state"].device
            )
            kwargs["action_context"] = {"hand_history": history.unsqueeze(0)}
        result = agent.predict_action(obs_dict=obs_batch, inference_steps=inference_steps, generator=generator, **kwargs)
        prediction = result["future_actions"]
        if prediction.shape != (1, agent.future_steps, agent.control_action_dim):
            raise ValueError("future_actions must contain the complete physical future")
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite policy future_actions")
        actions = prediction[0, :action_steps]
        return np.ascontiguousarray(actions.detach().cpu().numpy())

    def run_episode(
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

        # Environment and policy own separate random streams.
        generator = torch.Generator(device=next(agent.parameters()).device).manual_seed(episode_seed)

        done = False
        truncated = False
        episode_success = False
        task_done_step = None
        device = next(agent.parameters()).device

        while not (done or truncated):
            obs_batch = self.get_obs_batch(device=device)
            action_chunk = self.predict_action_chunk(
                obs_batch, agent, inference_steps=inference_steps, action_steps=action_steps, generator=generator
            )
            for action in action_chunk:
                obs, _, done, truncated, info = env.step(action)
                if self.hand_history is not None:
                    # History follows the command actually applied after environment clipping.
                    executed = np.asarray(info["executed_joint_action"], dtype=np.float32)
                    if executed.shape != (agent.action_dim,) or not np.isfinite(executed).all():
                        raise ValueError("LAMP requires env.step info['executed_joint_action']")
                    self.hand_history[:-1] = self.hand_history[1:]
                    self.hand_history[-1] = executed[-agent.hand_dim :]
                self.update_obs(obs)

                if info["success_condition"] and task_done_step is None:
                    task_done_step = env.action_cnt

                if info["success"]:
                    episode_success = True

                if done or truncated:
                    break

        return episode_success, task_done_step

    def run(self, agent, *, seeds, inference_steps, action_steps, video_save_dir=None):
        env = None
        try:
            for index, seed in enumerate(seeds):
                if env is None or (
                    self.clear_cache_freq > 0 and index % self.clear_cache_freq == 0
                ):
                    if env is not None:
                        previous, env = env, None
                        previous.close()
                    env = self.make_env()
                fps = (
                    self.env_video_fps
                    if self.env_video_fps is not None
                    else getattr(env, "video_fps", 15)
                )
                try:
                    success, steps = self.run_episode(
                        agent, env, seed, inference_steps, action_steps=action_steps
                    )
                except BaseException:
                    try:
                        save_episode_video(env, video_save_dir, f"episode_{seed}_crash", fps)
                    except BaseException:
                        traceback.print_exc()
                    raise
                video = save_episode_video(env, video_save_dir, f"episode_{seed}", fps)
                yield {
                    "task_name": self.task_name,
                    "seed": seed,
                    "success": bool(success),
                    "steps": steps,
                    "total_steps": env.action_cnt,
                    "video": video,
                }
        finally:
            if env is not None:
                close_resources(env.close)

    @staticmethod
    def expand_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Expand episode randomization switches; explicit kwargs take precedence."""
        expanded = dict(env_kwargs)

        if expanded.pop("texture_random", False):
            expanded.setdefault("use_texture_randomization", True)
            expanded.setdefault("per_episode_skybox", True)
            expanded.setdefault("randomize_light", True)
            expanded.setdefault("randomize_object_texture", True)
            expanded.setdefault("randomize_camera_pose", True)

        if expanded.pop("instance_random", False):
            expanded.setdefault("randomize_model_id", True)

        if expanded.pop("table_random", False):
            expanded.setdefault("randomize_table_height", True)

        return expanded

    def make_env(self):
        from dexmani_sim import resolve_env_class

        env_class = resolve_env_class(self.task_name)
        env = env_class(
            render_mode=self.render_mode, record_video=self.record_video, **self.env_kwargs
        )

        if self.viewer_resolution is not None and env.viewer is not None:
            env.viewer.resolution = self.viewer_resolution

        return env
