import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
from termcolor import cprint

from dexmani_policy.training.common.checkpoint_io import CheckpointStore


def _save_json(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


class SimEvaluator:
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.eval_root_dir / f"{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
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
        video_fps = int(self.env_runner.env_video_fps)

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

        for denoise_timesteps in denoise_timesteps_list:
            case_dir = eval_run_dir / f"denoise_timesteps{denoise_timesteps}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # Run episodes one-by-one and save results incrementally,
            # so partial results survive even if evaluation crashes mid-run.
            env = self.env_runner.make_env()
            eval_seeds = self.env_runner.get_seed_list()
            num_episodes = min(eval_episodes, len(eval_seeds))
            if eval_episodes > len(eval_seeds):
                cprint(
                    f"⚠️ eval_episodes ({eval_episodes}) > available seeds "
                    f"({len(eval_seeds)}), limiting to {num_episodes}", "yellow")

            success_list: List[bool] = []
            task_done_step_list: List[int] = []
            episode_details: List[Dict[str, Any]] = []
            video_count = 0

            try:
                for episode_idx in range(num_episodes):
                    eval_seed = eval_seeds[episode_idx]
                    try:
                        done, task_done_step = self.env_runner.eval_one_episode(
                            self.agent, env, eval_seed,
                            denoise_timesteps=denoise_timesteps,
                        )
                        video_array = env.get_video()
                        video_count += 1

                        if (self.env_runner.clear_cache_freq > 0
                                and video_count % self.env_runner.clear_cache_freq == 0):
                            env.close()
                            env = self.env_runner.make_env()

                        status = "success" if done else "fail"
                        done_step_str = task_done_step if task_done_step is not None else "N/A"
                        cprint(
                            f"[progress {episode_idx + 1}/{num_episodes}] "
                            f"env seed: {eval_seed}, status: {status}, "
                            f"done step: {done_step_str}", "cyan")

                        success_list.append(done)
                        if done and task_done_step is not None:
                            task_done_step_list.append(task_done_step)
                        episode_details.append({
                            "seed": eval_seed,
                            "success": done,
                            "steps": task_done_step,
                        })

                        if video_array is not None:
                            video_path = case_dir / f"eval_episode_{eval_seed}.mp4"
                            imageio.mimsave(
                                str(video_path),
                                video_array.astype(np.uint8),
                                fps=video_fps,
                            )

                    except Exception as e:
                        cprint(f"Seed {eval_seed} failed: {e}", "red")
                        success_list.append(False)
                        episode_details.append({
                            "seed": eval_seed,
                            "success": False,
                            "steps": None,
                            "error": str(e),
                        })

                    # --- incremental save after every episode ---
                    _save_json({
                        "success_rate": float(np.mean(success_list)) if success_list else None,
                        "avg_steps": int(round(np.mean(task_done_step_list))) if task_done_step_list else None,
                        "episode_details": episode_details,
                        "completed": len(success_list),
                        "total": num_episodes,
                    }, case_dir / "metrics.json")

            finally:
                env.close()

            case_metrics: Dict[str, Any] = {
                "success_rate": float(np.mean(success_list)) if success_list else None,
                "avg_steps": int(round(np.mean(task_done_step_list))) if task_done_step_list else None,
            }
            if episode_details:
                case_metrics["episode_details"] = episode_details
            _save_json(case_metrics, case_dir / "metrics.json")
            summary["metrics"][f"denoise_timesteps{denoise_timesteps}"] = case_metrics

        _save_json(summary, eval_run_dir / "eval_metrics.json")
        return summary
