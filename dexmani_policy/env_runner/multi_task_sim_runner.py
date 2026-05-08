import torch
import numpy as np
from typing import List, Optional, Dict, Any
from termcolor import cprint

from dexmani_policy.env_runner.sim_runner import SimRunner


class _TaskAwareSimRunner(SimRunner):
    """为特定 task_id 定制的 SimRunner，在 predict_action 时注入 task_id"""

    def __init__(self, task_id: int, **kwargs):
        super().__init__(**kwargs)
        self.task_id = task_id

    @torch.no_grad()
    def get_action_chunk(self, nobs, agent, denoise_timesteps: int = None) -> np.ndarray:
        action = agent.predict_action(
            obs_dict=nobs, task_id=self.task_id, denoise_timesteps=denoise_timesteps
        )
        return action["control_action"].detach().cpu().numpy().squeeze(0)


class MultiTaskSimRunner:
    """
    多任务仿真评估 Runner，按任务分别评估并汇总结果。

    为每个 task 创建独立的 _TaskAwareSimRunner 实例，分别运行评估，
    最终汇总各 task 的 success_rate 取平均作为总指标。
    """

    def __init__(
        self,
        task_configs: List[Dict[str, Any]],
        n_obs_steps: int,
        env_video_fps: int,
        default_eval_episodes: int,
        sensor_modalities: Optional[List[str]] = None,
    ):
        self.env_video_fps = env_video_fps
        self.runners: Dict[str, _TaskAwareSimRunner] = {}

        for cfg in task_configs:
            task_name = cfg["task_name"]
            task_id = cfg["task_id"]
            env_kwargs = cfg.get("env_kwargs")

            self.runners[task_name] = _TaskAwareSimRunner(
                task_id=task_id,
                task_name=task_name,
                n_obs_steps=n_obs_steps,
                env_video_fps=env_video_fps,
                default_eval_episodes=default_eval_episodes,
                sensor_modalities=sensor_modalities,
                env_kwargs=env_kwargs,
            )

    def run(
        self,
        agent,
        denoise_timesteps: int = None,
        eval_episodes: int = None,
    ) -> Dict[str, Any]:
        per_task: Dict[str, Any] = {}
        all_videos = []

        for task_name, runner in self.runners.items():
            cprint(f"\n{'='*40} Evaluating task: {task_name} (id={runner.task_id}) {'='*40}", "cyan")
            result = runner.run(agent, denoise_timesteps=denoise_timesteps, eval_episodes=eval_episodes)
            per_task[task_name] = result
            all_videos.extend(result.get("videos", []))

        rates = [r["success_rate"] for r in per_task.values() if r["success_rate"] is not None]
        steps = [r["avg_steps"] for r in per_task.values() if r["avg_steps"] is not None]

        avg_success_rate = sum(rates) / len(rates) if rates else None
        avg_steps = int(round(sum(steps) / len(steps))) if steps else None

        cprint(f"\n{'='*90}", "yellow")
        cprint(f"[Multi-Task Summary]", "yellow")
        for task_name, result in per_task.items():
            sr = result["success_rate"]
            sr_str = f"{sr*100:.1f}%" if sr is not None else "N/A"
            cprint(f"  {task_name}: success_rate={sr_str}, avg_steps={result['avg_steps']}", "yellow")
        total_sr_str = f"{avg_success_rate*100:.1f}%" if avg_success_rate is not None else "N/A"
        cprint(f"  Overall: success_rate={total_sr_str}, avg_steps={avg_steps}", "yellow")
        cprint(f"{'='*90}", "yellow")

        return {
            "success_rate": avg_success_rate,
            "avg_steps": avg_steps,
            "videos": all_videos,
            "per_task": per_task,
        }
