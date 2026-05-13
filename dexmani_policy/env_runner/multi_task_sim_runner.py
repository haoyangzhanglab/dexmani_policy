import torch
import numpy as np
from typing import List, Optional, Dict, Any
from termcolor import cprint

from dexmani_policy.env_runner.sim_runner import SimRunner


class TaskTextSimRunner(SimRunner):
    """为特定 task_text 定制的 SimRunner，在 predict_action 时将 task_text 注入 obs dict"""

    def __init__(self, task_text: str, **kwargs):
        super().__init__(**kwargs)
        self.task_text = task_text

    @torch.no_grad()
    def get_action_chunk(self, obs_batch, agent, denoise_timesteps: int = None) -> np.ndarray:
        obs_batch["task_text"] = [self.task_text]
        return super().get_action_chunk(obs_batch, agent, denoise_timesteps)


class MultiTaskSimRunner:
    """
    多任务仿真评估 Runner，按任务分别评估并汇总结果。

    为每个 task 创建独立的 TaskTextSimRunner 实例，分别运行评估，
    最终汇总各 task 的 success_rate 取平均作为总指标。
    """

    def __init__(
        self,
        task_configs: List[Dict[str, Any]],
        n_obs_steps: int,
        env_video_fps: int,
        default_eval_episodes: int,
        sensor_modalities: Optional[List[str]] = None,
        clear_cache_freq: int = 25,
    ):
        if not task_configs:
            raise ValueError("task_configs cannot be empty")

        self.env_video_fps = env_video_fps
        self.runners: Dict[str, TaskTextSimRunner] = {}

        for cfg in task_configs:
            task_name = cfg["task_name"]
            task_text = cfg.get("task_text", task_name)
            env_kwargs = cfg.get("env_kwargs")

            self.runners[task_name] = TaskTextSimRunner(
                task_text=task_text,
                task_name=task_name,
                n_obs_steps=n_obs_steps,
                env_video_fps=env_video_fps,
                default_eval_episodes=default_eval_episodes,
                sensor_modalities=sensor_modalities,
                env_kwargs=env_kwargs,
                clear_cache_freq=clear_cache_freq,
            )

    def print_summary(self, per_task, avg_success_rate, avg_steps, rates, failed_tasks):
        cprint("\n" + "="*90, "yellow")
        cprint(f"[Multi-Task Summary]", "yellow")
        for task_name, result in per_task.items():
            sr = result["success_rate"]
            if sr is not None:
                sr_str = f"{sr*100:.1f}%"
                cprint(f"  {task_name}: success_rate={sr_str}, avg_steps={result['avg_steps']}", "yellow")
            else:
                error_type = result.get("error_type", "Unknown")
                cprint(f"  {task_name}: FAILED - {error_type}", "red")

        if failed_tasks:
            cprint(f"  Failed tasks: {failed_tasks}", "red")

        total_sr_str = f"{avg_success_rate*100:.1f}%" if avg_success_rate is not None else "N/A"
        success_count = len(rates)
        total_count = len(self.runners)
        cprint(f"  Overall ({success_count}/{total_count} tasks): success_rate={total_sr_str}, avg_steps={avg_steps}", "yellow")
        cprint(f"{'='*90}", "yellow")

    def run(
        self,
        agent,
        denoise_timesteps: int = None,
        eval_episodes: int = None,
    ) -> Dict[str, Any]:
        per_task: Dict[str, Any] = {}
        all_videos = []
        failed_tasks = []

        for task_name, runner in self.runners.items():
            cprint(f"\n{'='*40} Evaluating task: {task_name} (text={runner.task_text}) {'='*40}", "cyan")
            try:
                result = runner.run(agent, denoise_timesteps=denoise_timesteps, eval_episodes=eval_episodes)
                per_task[task_name] = result
                for v in result.get("videos", []):
                    for k, arr in v.items():
                        all_videos.append({f"{task_name}/{k}": arr})
            except Exception as e:
                cprint(f"Task {task_name} failed: {type(e).__name__}: {e}", "red")
                failed_tasks.append(task_name)
                per_task[task_name] = {
                    "success_rate": None,
                    "avg_steps": None,
                    "videos": [],
                    "episode_details": [],
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

        rates = [r["success_rate"] for r in per_task.values() if r["success_rate"] is not None]
        steps = [r["avg_steps"] for r in per_task.values() if r["avg_steps"] is not None]

        avg_success_rate = sum(rates) / len(rates) if rates else None
        avg_steps = int(round(sum(steps) / len(steps))) if steps else None

        self.print_summary(per_task, avg_success_rate, avg_steps, rates, failed_tasks)

        return {
            "success_rate": avg_success_rate,
            "avg_steps": avg_steps,
            "videos": all_videos,
            "per_task": per_task,
            "failed_tasks": failed_tasks,
        }
