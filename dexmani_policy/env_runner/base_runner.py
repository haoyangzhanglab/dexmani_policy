from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
from termcolor import cprint

from dexmani_policy.common.pytorch_util import dict_apply, format_success_rate
from dexmani_policy.common.temporal_ensembler import ChunkOverlapBlender


class BaseRunner:
    """Abstract environment runner for agent evaluation.

    Manages the evaluation loop for a single task:

    - Maintains an observation deque and stacks the last ``n_obs_steps``
      frames for the agent's observation window.
    - Runs ``num_episodes`` trials, each starting from ``env.reset()`` and
      stepping until termination or ``max_steps``.
    - Collects video frames and success/failure outcomes per episode.
    - Handles evaluation errors per-episode (continues to next episode on
      failure rather than aborting the entire run).

    Subclasses override ``run()`` to adapt to specific environment types
    (single-task sim, multi-task sim, real robot, etc.).
    """

    def __init__(
        self,
        n_obs_steps: int,
        default_eval_episodes: int,
        sensor_modalities: List[str] | None = None,
        clear_cache_freq: int = 25,
        env_video_fps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ):
        self.is_multi_task = False
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = sensor_modalities or ["point_cloud", "joint_state"]

        # Circular buffer: pre-allocated per-modality storage to avoid per-step
        # np.zeros allocations in the hot path.
        self._obs_buffer: Dict[str, np.ndarray] = {}
        self._obs_cursor = 0  # next write position in ring buffer
        self._obs_count = 0  # number of frames stored (0 .. n_obs_steps)
        self._obs_str_buffer: Dict[str, list] = {}  # for string modalities

        self.env_video_fps = env_video_fps  # may be None → auto-detect from env
        self.default_eval_episodes = default_eval_episodes
        self.clear_cache_freq = clear_cache_freq

        # ACT temporal ensembling (Zhao et al. 2023, arXiv:2304.13705).
        # When coeff is set (e.g. 0.01), consecutive overlapping chunks are
        # blended via exponential weighting for smoother action transitions.
        self._blender: ChunkOverlapBlender | None = None
        if temporal_ensemble_coeff is not None:
            self._blender = ChunkOverlapBlender(
                temporal_ensemble_coeff=temporal_ensemble_coeff,
            )

    def update_obs(self, observation: Dict[str, Any]):
        """Write one observation frame into the circular buffer.

        On the first call the buffer is allocated lazily from the frame shapes.
        """
        pos = self._obs_cursor % self.n_obs_steps
        for k, v in observation.items():
            if k not in self.sensor_modalities:
                continue
            if isinstance(v, np.ndarray):
                if k not in self._obs_buffer:
                    self._obs_buffer[k] = np.zeros((self.n_obs_steps,) + v.shape, dtype=v.dtype)
                self._obs_buffer[k][pos] = v
            elif isinstance(v, torch.Tensor):
                if k not in self._obs_buffer:
                    self._obs_buffer[k] = torch.zeros(
                        (self.n_obs_steps,) + tuple(v.shape), dtype=v.dtype, device=v.device
                    )
                self._obs_buffer[k][pos] = v
            elif isinstance(v, str):
                if k not in self._obs_str_buffer:
                    self._obs_str_buffer[k] = []
                buf = self._obs_str_buffer[k]
                buf.append(v)
                if len(buf) > self.n_obs_steps:
                    buf.pop(0)
        self._obs_cursor += 1
        self._obs_count = min(self._obs_count + 1, self.n_obs_steps)

    def get_stacked_obs(self) -> Dict[str, Any]:
        """Return a time-ordered stack of the last n_obs_steps frames.

        Uses pre-allocated circular buffer -- zero per-call allocation in the
        common case (count >= n_obs_steps).
        """
        if self._obs_count == 0:
            raise RuntimeError("No observation in buffer")
        out: Dict[str, Any] = {}
        for k, buf in self._obs_buffer.items():
            if self._obs_count < self.n_obs_steps:
                # Episode start: only _obs_count frames available.
                # Pad the beginning with the first frame.
                result = np.empty_like(buf) if isinstance(buf, np.ndarray) else torch.empty_like(buf)
                pad_len = self.n_obs_steps - self._obs_count
                result[:pad_len] = buf[0]
                result[pad_len:] = buf[: self._obs_count]
                out[k] = result
            else:
                # Normal case: return chronologically-ordered slice.
                start = self._obs_cursor % self.n_obs_steps
                idx = (start + np.arange(self.n_obs_steps)) % self.n_obs_steps
                if isinstance(buf, np.ndarray):
                    out[k] = buf[idx]
                else:
                    out[k] = buf[torch.as_tensor(idx, device=buf.device)]
        for k, buf in self._obs_str_buffer.items():
            out[k] = [buf[-1]] * self.n_obs_steps if buf else []
        if len(out) == 0:
            raise RuntimeError("Stacked observation dict is empty")
        return out

    def get_obs_batch(self, device) -> Dict[str, Any]:
        def to_torch(x, *, dtype=None, device=None):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)
            if isinstance(x, np.ndarray):
                return torch.as_tensor(x, device=device, dtype=dtype)
            return x

        stacked_obs = self.get_stacked_obs()
        obs_batch = dict_apply(stacked_obs, lambda x: to_torch(x, device=device))
        obs_batch = dict_apply(obs_batch, lambda x: x.unsqueeze(0) if torch.is_tensor(x) else x)

        return obs_batch

    def reset(self):
        self._obs_buffer.clear()
        self._obs_str_buffer.clear()
        self._obs_cursor = 0
        self._obs_count = 0
        if self._blender is not None:
            self._blender.reset()

    @torch.no_grad()
    def get_action_chunk(self, obs_batch, agent, denoise_timesteps: int = None) -> np.ndarray:
        result = agent.predict_action(obs_dict=obs_batch, denoise_timesteps=denoise_timesteps)

        if self._blender is not None:
            full_pred = result["pred_action"]  # (B, horizon, A)
            blended = self._blender.update(full_pred, n_action_steps=agent.n_action_steps)
            return blended.detach().cpu().numpy().squeeze(0)

        return result["control_action"].detach().cpu().numpy().squeeze(0)

    @staticmethod
    def _encode_video(path: Path, frames: np.ndarray, fps: int) -> None:
        """Encode frames to MP4, preferring ffmpeg pipe for streaming.

        Falls back to imageio.mimsave if ffmpeg is not available.
        """
        import shutil
        import subprocess

        if shutil.which("ffmpeg"):
            T, H, W, C = frames.shape
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{W}x{H}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(frame.astype(np.uint8).tobytes())
            proc.stdin.close()
            proc.wait(timeout=60)
        else:
            imageio.mimsave(str(path), frames.astype(np.uint8), fps=fps)

    def run_one_episode(self, agent, env, episode_seed, denoise_timesteps: int = None, **kwargs):
        """Run a single evaluation episode.

        Environment contract (required for accurate ``avg_steps`` metrics):
            - ``info["success_condition"]`` (bool): set on the **first** step
              where the task goal is met (raw signal, no hold delay).
            - ``env.action_cnt`` (int): current step counter on the env.
            - ``info["success"]`` (bool): set when the episode is considered
              successful (may include a hold/grace delay after
              ``success_condition``).

        If ``success_condition`` or ``action_cnt`` is missing, ``task_done_step``
        will be ``None`` and ``avg_steps`` will be reported as ``N/A``.
        """
        obs, info = env.reset(seed=episode_seed, options=kwargs.get("options", None))
        self.reset()
        self.update_obs(obs)

        done = False
        truncated = False
        episode_success = False
        task_done_step = None  # None = not yet succeeded (distinct from action_cnt=0)
        device = next(agent.parameters()).device

        while not (done or truncated):
            obs_batch = self.get_obs_batch(device=device)
            action_chunk = self.get_action_chunk(obs_batch, agent, denoise_timesteps=denoise_timesteps)
            for i in range(action_chunk.shape[0]):
                obs, reward, done, truncated, info = env.step(action_chunk[i])
                self.update_obs(obs)

                # Record first success step using the raw success_condition (no hold delay)
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
        denoise_timesteps: int = None,
        eval_episodes: int = None,
        video_save_dir: Optional[Path] = None,
    ):
        """Run *eval_episodes* evaluation trials.

        Exception isolation (three-layer defence):

        1. **Episode execution** (``run_one_episode``) — caught, recorded as
           ``False``, crash video saved if possible.
        2. **Frame extraction** (``env.get_video()``) — caught, falls back to
           ``None`` — never corrupts the episode result.
        3. **Video encoding** (``_encode_video``) — caught, warning printed —
           never corrupts the episode result.

        The ``episode_completed`` flag decouples execution from video I/O so
        that video failures cannot poison ``success_list``.
        """
        env = self.make_env()
        if self.env_video_fps is None:
            self.env_video_fps = getattr(env, "video_fps", 15)
        eval_seeds = self.get_seed_list()
        eval_episodes = eval_episodes if eval_episodes is not None else self.default_eval_episodes

        if eval_episodes > len(eval_seeds):
            cprint(
                f"⚠️ eval_episodes ({eval_episodes}) > available seeds ({len(eval_seeds)}), limiting to {len(eval_seeds)}",
                "yellow",
            )
            eval_episodes = len(eval_seeds)

        num_episodes = eval_episodes
        success_list = []
        task_done_step_list = []
        episode_video_list = []
        episode_details = []  # per-episode: {seed, success, steps}
        attempted = 0

        print("=" * 90)

        try:
            seed_idx = 0
            while len(success_list) < num_episodes and seed_idx < len(eval_seeds):
                eval_seed = eval_seeds[seed_idx]
                seed_idx += 1
                attempted += 1

                try:
                    episode_success, task_done_step = self.run_one_episode(
                        agent, env, eval_seed, denoise_timesteps
                    )
                    episode_completed = True
                except Exception as e:
                    episode_completed = False
                    cprint(f"Seed {eval_seed} failed: {e}", "red")
                    # Try to capture pre-crash video frames for diagnostics
                    try:
                        crash_video = env.get_video()
                    except Exception:
                        crash_video = None
                    success_list.append(False)
                    episode_details.append(
                        {
                            "seed": eval_seed,
                            "success": False,
                            "steps": None,
                            "total_steps": getattr(env, "action_cnt", None),
                            "error": str(e),
                        }
                    )
                    if crash_video is not None and video_save_dir is not None:
                        crash_path = video_save_dir / f"episode_{eval_seed}_crash.mp4"
                        crash_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            self._encode_video(crash_path, crash_video, self.env_video_fps)
                        except Exception:
                            pass
                        episode_video_list.append({f"episode_{eval_seed}_crash": str(crash_path)})

                if episode_completed:
                    total_steps = getattr(env, "action_cnt", None)

                    # get_video() is best-effort — failures must not corrupt episode metrics
                    try:
                        video = env.get_video()
                    except Exception:
                        video = None

                    if self.clear_cache_freq > 0 and attempted % self.clear_cache_freq == 0:
                        env.close()
                        env = self.make_env()

                    status = "success" if episode_success else "fail"
                    done_step_str = task_done_step if task_done_step is not None else "N/A"
                    cprint(
                        f"[progress {len(success_list) + 1}/{num_episodes}] env seed: {eval_seed}, status: {status}, done step: {done_step_str}",
                        "cyan",
                    )

                    success_list.append(episode_success)
                    if episode_success and task_done_step is not None:
                        task_done_step_list.append(task_done_step)
                    episode_details.append(
                        {
                            "seed": eval_seed,
                            "success": episode_success,
                            "steps": task_done_step,
                            "total_steps": total_steps,
                        }
                    )
                    # Video encoding is best-effort — failures must not corrupt metrics
                    if video is not None and video_save_dir is not None:
                        video_path = video_save_dir / f"episode_{eval_seed}.mp4"
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            self._encode_video(video_path, video, self.env_video_fps)
                            episode_video_list.append({f"episode_{eval_seed}": str(video_path)})
                        except Exception as e:
                            cprint(f"  ⚠ Video encoding failed for seed {eval_seed}: {e}", "yellow")

            if len(success_list) < num_episodes:
                cprint(
                    f"Warning: Only collected {len(success_list)}/{num_episodes} valid episodes (ran out of seeds)",
                    "red",
                )

            success_rate = float(np.mean(success_list)) if len(success_list) > 0 else None
            avg_steps = int(round(np.mean(task_done_step_list))) if len(task_done_step_list) > 0 else None

            # avg_steps_all includes all episodes (failures → full episode length)
            all_steps = [d["total_steps"] for d in episode_details if d.get("total_steps") is not None]
            avg_steps_all = int(round(np.mean(all_steps))) if all_steps else None

            sr_str = format_success_rate(success_rate)
            avg_steps_str = "N/A" if avg_steps is None else str(avg_steps)
            avg_all_str = "N/A" if avg_steps_all is None else str(avg_steps_all)
            cprint(
                f"[result] Valid: {len(success_list)}/{num_episodes}, Success rate: {sr_str}, "
                f"Avg steps (success): {avg_steps_str}, Avg steps (all): {avg_all_str}",
                "yellow",
            )
            print("=" * 90)

        finally:
            env.close()

        return {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_steps_all": avg_steps_all,
            "videos": episode_video_list,
            "episode_details": episode_details,
            "episodes_collected": len(success_list),
            "episodes_requested": num_episodes,
        }

    def make_env(self):
        raise NotImplementedError

    def get_seed_list(self) -> List[int]:
        raise NotImplementedError
