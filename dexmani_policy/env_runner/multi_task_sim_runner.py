from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from termcolor import cprint

from dexmani_policy.common.pytorch_util import format_success_rate
from dexmani_policy.env_runner.base_runner import EvalEpisodeError, _classify_eval_exception
from dexmani_policy.env_runner.sim_runner import SimRunner


class TaskTextSimRunner(SimRunner):
    def __init__(self, task_text: str, **kwargs):
        super().__init__(**kwargs)
        self.task_text = task_text

    @torch.no_grad()
    def get_action_chunk(self, obs_batch, agent, denoise_timesteps: int = None) -> np.ndarray:
        obs_batch["task_text"] = [self.task_text]
        return super().get_action_chunk(obs_batch, agent, denoise_timesteps)


class MultiTaskSimRunner:
    def __init__(
        self,
        task_configs: List[Dict[str, Any]],
        n_obs_steps: int,
        default_eval_episodes: int,
        sensor_modalities: Optional[List[str]] = None,
        clear_cache_freq: int = 25,
        env_video_fps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ):
        if not task_configs:
            raise ValueError("task_configs cannot be empty")

        self.is_multi_task = True
        self.env_video_fps = env_video_fps
        # Shared seed pool propagated to every child runner (set by the eval
        # entry points before ``run``). None → each child uses its own default.
        self.eval_seeds: Optional[List[int]] = None
        self.runners: Dict[str, TaskTextSimRunner] = {}

        for cfg in task_configs:
            task_name = cfg["task_name"]
            task_text = cfg.get("task_text")
            if task_text is None:
                task_text = task_name
                cprint(
                    f"⚠️ task_text not set for {task_name}, falling back to task_name='{task_name}'. "
                    f"Ensure this matches dataset.task_texts to avoid train/eval text embedding mismatch.",
                    "yellow",
                )
            env_kwargs = cfg.get("env_kwargs")
            if env_kwargs is None:
                cprint(
                    f"⚠️  task '{task_name}' has no env_kwargs set — control_mode defaults to 'joint'",
                    "yellow",
                )

            self.runners[task_name] = TaskTextSimRunner(
                task_text=task_text,
                task_name=task_name,
                n_obs_steps=n_obs_steps,
                env_video_fps=env_video_fps,
                default_eval_episodes=default_eval_episodes,
                sensor_modalities=sensor_modalities,
                env_kwargs=env_kwargs,
                clear_cache_freq=clear_cache_freq,
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                rgb_preprocess_size=cfg.get("rgb_preprocess_size"),
                rgb_random_crop_size=cfg.get("rgb_random_crop_size"),
            )

    def get_seed_list(self) -> List[int]:
        """Return the shared seed pool applied to every child task runner.

        The multi-task evaluation unit is ``(task, seed)``: one seed pool is
        propagated to every task, so a single seed list is the correct
        interface for the eval entry points (``_select_eval_seeds`` /
        ``select_best_ckpt``).  Delegates to the first child runner (seed file
        / default pool) when no explicit pool has been set.
        """
        if self.eval_seeds is not None:
            return list(self.eval_seeds)
        return next(iter(self.runners.values())).get_seed_list()

    def print_summary(
        self,
        per_task,
        macro_success_rate,
        micro_success_rate,
        micro_n_success,
        micro_n_valid,
        avg_steps,
        rates,
        failed_tasks,
    ):
        cprint("\n" + "=" * 90, "yellow")
        cprint("[Multi-Task Summary]", "yellow")
        for task_name, result in per_task.items():
            sr = result["success_rate"]
            if sr is not None:
                sr_str = format_success_rate(sr)
                cprint(
                    f"  {task_name}: success_rate={sr_str}, avg_steps (success only)={result['avg_steps']}",
                    "yellow",
                )
            else:
                error_type = result.get("error_type", "Unknown")
                cprint(f"  {task_name}: FAILED - {error_type}", "red")

        if failed_tasks:
            cprint(f"  Failed tasks: {failed_tasks}", "red")

        macro_str = format_success_rate(macro_success_rate)
        micro_str = format_success_rate(micro_success_rate)
        n_tasks = len(self.runners)
        cprint(
            f"  Macro ({len(rates)}/{n_tasks} tasks): success_rate={macro_str}",
            "yellow",
        )
        cprint(
            f"  Micro ({micro_n_success}/{micro_n_valid} episodes): "
            f"success_rate={micro_str}, avg_steps (success only)={avg_steps}",
            "yellow",
        )
        cprint(f"{'=' * 90}", "yellow")

    def run(
        self,
        agent,
        denoise_timesteps: int = None,
        eval_episodes: int = None,
        video_save_dir=None,
    ) -> Dict[str, Any]:

        per_task: Dict[str, Any] = {}
        all_videos = []
        failed_tasks = []

        # Propagate eval_seeds to child runners (C3 fix).
        # select_best_ckpt.py and eval_best_ckpt.py set this attribute on the
        # MultiTaskSimRunner instance, but without this block each child
        # runner independently calls get_seed_list() with its own default.
        parent_seeds = getattr(self, "eval_seeds", None)
        for task_name, runner in self.runners.items():
            if parent_seeds is not None:
                runner.eval_seeds = list(parent_seeds)
            cprint(f"\n{'=' * 40} Evaluating task: {task_name} (text={runner.task_text}) {'=' * 40}", "cyan")
            # Each task gets its own sub-directory for videos
            task_video_dir = None
            if video_save_dir is not None:
                task_video_dir = video_save_dir / task_name
                task_video_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = runner.run(
                    agent,
                    denoise_timesteps=denoise_timesteps,
                    eval_episodes=eval_episodes,
                    video_save_dir=task_video_dir,
                )
                per_task[task_name] = result
                for v in result.get("videos", []):
                    for k, arr_or_path in v.items():
                        all_videos.append({f"{task_name}_{k}": arr_or_path})
            except KeyboardInterrupt:
                cprint(f"\n⚠️ Evaluation interrupted by user during task {task_name}", "yellow")
                raise
            except EvalEpisodeError:
                raise
            except (RuntimeError, ValueError, AttributeError) as e:
                cprint(f"Task {task_name} failed: {type(e).__name__}: {e}", "red")
                failed_tasks.append(task_name)
                per_task[task_name] = {
                    "success_rate": None,
                    "avg_steps": None,
                    "videos": [],
                    "episode_details": [],
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_category": _classify_eval_exception(e),
                }
            except Exception as e:
                cprint(f"\n❌ Unexpected error in task {task_name}: {type(e).__name__}: {e}", "red")
                traceback.print_exc()
                cprint("This is an unexpected error. Please report this issue.", "red")
                failed_tasks.append(task_name)
                per_task[task_name] = {
                    "success_rate": None,
                    "avg_steps": None,
                    "videos": [],
                    "episode_details": [],
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_category": _classify_eval_exception(e),
                }

        # ── Aggregate: macro (mean per-task SR) + micro (across all (task, seed)) ──
        # Failed tasks carry success_rate=None here and are filtered out of
        # `rates` below; the abort after aggregation keeps a reduced macro
        # average from being silently reported.
        rates = [r["success_rate"] for r in per_task.values() if r["success_rate"] is not None]
        steps = [r["avg_steps"] for r in per_task.values() if r["avg_steps"] is not None]

        macro_success_rate = sum(rates) / len(rates) if rates else None
        avg_steps = int(round(sum(steps) / len(steps))) if steps else None

        # Micro: denominator is the actual number of completed (task, seed)
        # episode units — not the per-task seed count (which would undercount
        # and could yield a success rate > 1).
        micro_n_success = sum(
            1 for r in per_task.values() for d in r.get("episode_details", []) if d.get("success")
        )
        micro_n_valid = sum(len(r.get("episode_details", [])) for r in per_task.values())
        micro_success_rate = (micro_n_success / micro_n_valid) if micro_n_valid > 0 else None

        self.print_summary(
            per_task, macro_success_rate, micro_success_rate,
            micro_n_success, micro_n_valid, avg_steps, rates, failed_tasks,
        )

        if failed_tasks:
            raise RuntimeError(
                f"Multi-task evaluation failed for tasks: {failed_tasks}. "
                f"Refusing to report a reduced aggregate success rate."
            )

        results = {
            "success_rate": macro_success_rate,   # macro (mean per-task); see micro_success_rate
            "macro_success_rate": macro_success_rate,
            "micro_success_rate": micro_success_rate,
            "avg_steps": avg_steps,
            "videos": all_videos,
            "per_task": per_task,
            "failed_tasks": failed_tasks,
            "n_success": micro_n_success,
            "n_valid_episodes": micro_n_valid,
        }
        return results
