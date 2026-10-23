"""Sequential language-conditioned evaluation over explicit task seed plans."""

from __future__ import annotations

from typing import Any

from termcolor import cprint

from dexmani_policy.env_runner.common.metrics import compute_evaluation_statistics
from dexmani_policy.env_runner.common.seeds import validate_task_seeds
from dexmani_policy.env_runner.sim_runner import SimRunner


class MultiTaskSimRunner:
    def __init__(
        self,
        task_configs: list[dict[str, Any]],
        n_obs_steps: int,
        sensor_modalities: list[str] | None = None,
        clear_cache_freq: int = 25,
        env_video_fps: int | None = None,
    ):
        if not task_configs:
            raise ValueError("task_configs cannot be empty")

        self.task_runners: dict[str, SimRunner] = {}
        defaults = {
            "n_obs_steps": n_obs_steps,
            "sensor_modalities": sensor_modalities,
            "clear_cache_freq": clear_cache_freq,
            "env_video_fps": env_video_fps,
        }
        for cfg in task_configs:
            task_name = cfg["task_name"]
            if task_name in self.task_runners:
                raise ValueError(f"Duplicate task: {task_name}")
            task_text = cfg.get("task_text")
            options = {
                **defaults,
                **cfg,
                "task_text": task_name if task_text is None else task_text,
            }
            self.task_runners[task_name] = SimRunner(**options)

    def run(
        self,
        agent,
        *,
        task_seeds: dict[str, list[int]],
        inference_steps: int | None = None,
        action_steps: int | None = None,
        video_save_dir=None,
    ) -> dict[str, Any]:
        validate_task_seeds(task_seeds, self.task_runners)
        per_task = {}
        videos = []
        for task, runner in self.task_runners.items():
            cprint(f"Evaluating task: {task} (text={runner.task_text})", "cyan")
            task_video_dir = video_save_dir / task if video_save_dir is not None else None
            result = runner.run(
                agent,
                inference_steps=inference_steps,
                action_steps=action_steps,
                task_seeds={task: task_seeds[task]},
                video_save_dir=task_video_dir,
            )
            per_task[task] = result
            videos.extend(
                {f"{task}_{name}": path}
                for video in result["videos"]
                for name, path in video.items()
            )

        stats = compute_evaluation_statistics({"per_task": per_task})
        steps = [
            child["avg_steps"] for child in per_task.values() if child["avg_steps"] is not None
        ]
        result = {
            "success_rate": stats["macro_success_rate"],
            "macro_success_rate": stats["macro_success_rate"],
            "micro_success_rate": stats["micro_success_rate"],
            "avg_steps": int(round(sum(steps) / len(steps))) if steps else None,
            "videos": videos,
            "per_task": per_task,
            "n_success": stats["n_success"],
            "n_valid_episodes": stats["n_valid_episodes"],
        }
        self.print_summary(result)
        return result

    def print_summary(self, result):
        cprint("\n[Multi-Task Summary]", "yellow")
        for task, child in result["per_task"].items():
            cprint(
                f"  {task}: success_rate={child['success_rate']:.1%}, "
                f"avg_steps (success only)={child['avg_steps']}",
                "yellow",
            )
        cprint(
            f"  Macro ({len(self.task_runners)} tasks): {result['macro_success_rate']:.1%}",
            "yellow",
        )
        cprint(
            f"  Micro ({result['n_success']}/{result['n_valid_episodes']} episodes): "
            f"{result['micro_success_rate']:.1%}, avg_steps={result['avg_steps']}",
            "yellow",
        )
