import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
from termcolor import cprint

from dexmani_policy.common.pytorch_util import format_success_rate
from dexmani_policy.common.checkpoint_io import CheckpointStore


def _save_json(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


class SimEvaluator:
    """Offline simulation evaluator for checkpoint assessment.

    Loads a trained agent from a checkpoint and runs it in the simulation
    environment. Supports:

    - Arbitrary checkpoint tags or direct paths via ``CheckpointStore``.
    - EMA vs non-EMA model selection.
    - Per-task success rate tracking and incremental JSONL logging.
    - Summary printing with task-level and average success rates.

    Evaluation seeds are fixed (``eval.seed: 0``) for reproducibility.
    """
    def __init__(
        self,
        device,
        agent,
        env_runner,
        checkpoint_store: CheckpointStore,
        eval_root_dir: Path,
    ):
        self.device = device
        self.agent = agent
        self.env_runner = env_runner
        self.checkpoint_store = checkpoint_store
        self.eval_root_dir = eval_root_dir
        self.eval_root_dir.mkdir(parents=True, exist_ok=True)

    def create_eval_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.eval_root_dir / f"{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _load_for_inference(self, tag_or_path: str, use_ema: bool):
        from dexmani_policy.common.pytorch_util import fix_state_dict
        path = self.checkpoint_store.resolve_path(tag_or_path)
        checkpoint = self.checkpoint_store.load(path)

        train_params = checkpoint.train_params
        if train_params is not None:
            for key in ('n_obs_steps', 'n_action_steps', 'action_dim', 'horizon', 'action_key'):
                expected = train_params.get(key)
                actual = getattr(self.agent, key, None)
                if expected is not None and actual is not None and expected != actual:
                    raise ValueError(
                        f"Checkpoint train_params.{key}={expected} does not match "
                        f"agent.{key}={actual}. The config.yaml used for eval may be "
                        f"from a different training run than this checkpoint."
                    )

        if use_ema and checkpoint.ema_model_state is not None:
            print("Using EMA weights for inference.")
            self.agent.load_state_dict(fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False), strict=True)
        else:
            if use_ema and checkpoint.ema_model_state is None:
                print("WARNING: EMA weights requested but not found in checkpoint. Using model weights.")
            self.agent.load_state_dict(fix_state_dict(checkpoint.model_state, is_current_ddp=False), strict=True)

        if not self.agent.normalizer.is_fitted(required_keys=['action']):
            raise RuntimeError(
                "Normalizer is missing required key 'action' after loading checkpoint. "
                "The checkpoint may be corrupted or saved without normalizer params."
            )

    @torch.no_grad()
    def run(
        self,
        eval_episodes: int,
        denoise_timesteps_list: List[int],
        ckpt_tag_or_path: str = "latest",
        use_ema_for_eval: bool = True,
        eval_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not denoise_timesteps_list:
            raise ValueError(
                "denoise_timesteps_list cannot be empty. "
                "Please provide at least one denoise timestep value."
            )

        cprint(f"Loading checkpoint: {ckpt_tag_or_path} (EMA={use_ema_for_eval})", "cyan")
        self._load_for_inference(ckpt_tag_or_path, use_ema_for_eval)
        cprint("✅ Checkpoint loaded successfully", "green")

        self.agent.to(self.device)
        self.agent.eval()

        eval_run_dir = self.create_eval_run_dir()
        video_fps = int(getattr(self.env_runner, 'env_video_fps', 15))

        if eval_config is not None:
            _save_json(eval_config, eval_run_dir / "eval_config.json")

        summary = {
            "meta": {
                "ckpt_tag": ckpt_tag_or_path,
                "eval_episodes": eval_episodes,
                "use_ema_for_eval": use_ema_for_eval,
                "denoise_timesteps": denoise_timesteps_list,
            },
            "metrics": {},
        }

        is_multi_task = hasattr(self.env_runner, 'runners')

        for denoise_timesteps in denoise_timesteps_list:
            case_dir = eval_run_dir / f"denoise_timesteps{denoise_timesteps}"
            case_dir.mkdir(parents=True, exist_ok=True)

            result = self.env_runner.run(
                self.agent,
                denoise_timesteps=denoise_timesteps,
                eval_episodes=eval_episodes,
                video_save_dir=case_dir,
            )

            # Videos are already saved to disk by the runner when video_save_dir is set.
            # Only handle the legacy path (raw arrays) if someone passes video_save_dir=None.
            for item in result.get("videos", []):
                for key, video_array_or_path in item.items():
                    if isinstance(video_array_or_path, (str, Path)):
                        continue  # already saved to disk
                    video_path = case_dir / f"{key}.mp4"
                    imageio.mimsave(
                        str(video_path),
                        video_array_or_path.astype(np.uint8),
                        fps=video_fps,
                    )

            case_metrics: Dict[str, Any] = {
                "success_rate": result["success_rate"],
                "avg_steps": result["avg_steps"],
            }
            if is_multi_task:
                per_task = result.get("per_task", {})
                if per_task:
                    case_metrics["per_task"] = per_task
            else:
                episode_details = result.get("episode_details", [])
                if episode_details:
                    case_metrics["episode_details"] = episode_details
            _save_json(case_metrics, case_dir / "metrics.json")
            summary["metrics"][f"denoise_timesteps{denoise_timesteps}"] = case_metrics

        _save_json(summary, eval_run_dir / "eval_metrics.json")

        # --- print final summary table ---
        cprint("=" * 60, "cyan")
        cprint("  Evaluation Summary", "cyan", attrs=["bold"])
        cprint("=" * 60, "cyan")
        cprint(f"  Checkpoint : {ckpt_tag_or_path} (EMA={use_ema_for_eval})", "cyan")
        cprint(f"  Episodes   : {eval_episodes}" + (" (per task)" if is_multi_task else ""), "cyan")
        for denoise_timesteps in denoise_timesteps_list:
            case_key = f"denoise_timesteps{denoise_timesteps}"
            case = summary["metrics"].get(case_key, {})
            sr = case.get("success_rate")
            avg = case.get("avg_steps")
            sr_str = format_success_rate(sr)
            avg_str = f"{avg}" if avg is not None else "N/A"
            cprint(f"  --- denoise_timesteps={denoise_timesteps} ---", "cyan")
            cprint(f"  Success Rate : {sr_str}", "green" if (sr or 0) >= 0.5 else "red")
            cprint(f"  Avg Steps    : {avg_str}", "cyan")

            per_task = case.get("per_task", {})
            for task_name, task_result in per_task.items():
                task_sr = task_result.get("success_rate")
                task_sr_str = format_success_rate(task_sr)
                cprint(f"    {task_name}: {task_sr_str}", "cyan")

        cprint(f"  Results saved to: {eval_run_dir}", "cyan")
        cprint("=" * 60, "cyan")

        return summary
