import torch
import numpy as np
from termcolor import cprint
from collections import deque
from typing import Any, Dict, List

from dexmani_policy.common.pytorch_util import dict_apply, format_success_rate


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
        env_video_fps: int,
        default_eval_episodes: int,
        sensor_modalities: List[str] | None = None,
        clear_cache_freq: int = 25,
    ):
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = sensor_modalities or ["point_cloud", "joint_state"]
        self.obs_deque = deque(maxlen=n_obs_steps)

        self.env_video_fps = env_video_fps
        self.default_eval_episodes = default_eval_episodes
        self.clear_cache_freq = clear_cache_freq

    @staticmethod
    def stack_last_n(all_items, n_steps):
        all_list = list(all_items)
        if len(all_list) == 0:
            raise RuntimeError("Empty input to stack_last_n()")
        head = all_list[0]

        if isinstance(head, np.ndarray):
            result = np.zeros((n_steps,) + all_list[-1].shape, dtype=all_list[-1].dtype)
            start_idx = -min(n_steps, len(all_list))
            result[start_idx:] = np.asarray(all_list[start_idx:])
            if n_steps > len(all_list):
                # Pad with the earliest available frame (same semantics as training-time episode-start padding)
                result[:start_idx] = result[start_idx]
            return result

        if isinstance(head, torch.Tensor):
            shape = (n_steps,) + tuple(all_list[-1].shape)
            result = torch.zeros(shape, dtype=all_list[-1].dtype, device=all_list[-1].device)
            start_idx = -min(n_steps, len(all_list))
            result[start_idx:] = torch.stack(list(all_list[start_idx:]), dim=0)
            if n_steps > len(all_list):
                result[:start_idx] = result[start_idx]
            return result

        if isinstance(head, str):
            last = all_list[-1]
            return [last] * n_steps

        raise RuntimeError(f"Unsupported observation field type: {type(head)}")    


    def update_obs(self, observation: Dict[str, Any]):
        self.obs_deque.append(observation)
    

    def get_stacked_obs(self) -> Dict[str, Any]:
        if len(self.obs_deque) == 0:
            raise RuntimeError("No observation in deque")
        keys = list(self.obs_deque[-1].keys())
        out: Dict[str, Any] = {}
        dropped_keys = []
        for k in keys:
            if k in self.sensor_modalities:
                out[k] = self.stack_last_n((frame[k] for frame in self.obs_deque), self.n_obs_steps)
            else:
                dropped_keys.append(k)
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
        self.obs_deque.clear()


    @torch.no_grad()
    def get_action_chunk(self, obs_batch, agent, denoise_timesteps:int=None) -> np.ndarray:
        action = agent.predict_action(obs_dict=obs_batch, denoise_timesteps=denoise_timesteps)
        return action["control_action"].detach().cpu().numpy().squeeze(0)
    

    def eval_one_episode(self, agent, env, episode_seed, denoise_timesteps:int=None, **kwargs):
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
                    task_done_step = getattr(env, 'action_cnt', None)

                if info.get("success", False):
                    episode_success = True

                if done or truncated:
                    break

        return episode_success, task_done_step
    

    def run(self, agent, denoise_timesteps:int=None, eval_episodes:int=None):
        env = self.make_env()
        eval_seeds = self.get_seed_list()
        eval_episodes = eval_episodes if eval_episodes is not None else self.default_eval_episodes

        if eval_episodes > len(eval_seeds):
            cprint(f"⚠️ eval_episodes ({eval_episodes}) > available seeds ({len(eval_seeds)}), limiting to {len(eval_seeds)}", "yellow")
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
                    episode_success, task_done_step = self.eval_one_episode(agent, env, eval_seed, denoise_timesteps)
                    video = env.get_video()

                    if self.clear_cache_freq > 0 and attempted % self.clear_cache_freq == 0:
                        env.close()
                        env = self.make_env()

                    status = "success" if episode_success else "fail"
                    done_step_str = task_done_step if task_done_step is not None else "N/A"
                    cprint(f"[progress {len(success_list)+1}/{num_episodes}] env seed: {eval_seed}, status: {status}, done step: {done_step_str}", "cyan")

                    success_list.append(episode_success)
                    if episode_success and task_done_step is not None:
                        task_done_step_list.append(task_done_step)
                    episode_details.append({
                        "seed": eval_seed,
                        "success": episode_success,
                        "steps": task_done_step,
                    })
                    if video is not None:
                        episode_video_list.append({
                            f"episode_{eval_seed}": video
                        })

                except Exception as e:
                    cprint(f"Seed {eval_seed} failed: {e}", "red")
                    # Try to capture pre-crash video frames for diagnostics
                    try:
                        crash_video = env.get_video()
                    except Exception:
                        crash_video = None
                    success_list.append(False)
                    episode_details.append({
                        "seed": eval_seed,
                        "success": False,
                        "steps": None,
                        "error": str(e),
                    })
                    if crash_video is not None:
                        episode_video_list.append({
                            f"episode_{eval_seed}_crash": crash_video
                        })

            if len(success_list) < num_episodes:
                cprint(f"Warning: Only collected {len(success_list)}/{num_episodes} valid episodes (ran out of seeds)", "red")

            success_rate = float(np.mean(success_list)) if len(success_list) > 0 else None
            avg_steps = int(round(np.mean(task_done_step_list))) if len(task_done_step_list) > 0 else None

            sr_str = format_success_rate(success_rate)
            avg_steps_str = 'N/A' if avg_steps is None else str(avg_steps)
            cprint(f"[result] Valid: {len(success_list)}/{num_episodes}, Success rate: {sr_str}, Avg steps (success only): {avg_steps_str}", "yellow")
            print("=" * 90)

        finally:
            env.close()

        return {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "videos": episode_video_list,
            "episode_details": episode_details,
        }
    

    def make_env(self):
        raise NotImplementedError


    def get_seed_list(self) -> List[int]:
        raise NotImplementedError
