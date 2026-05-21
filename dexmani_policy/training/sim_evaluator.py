import json
import torch
import imageio
import numpy as np
from pathlib import Path
from termcolor import cprint
from datetime import datetime
from typing import Dict, List, Any, Optional

from dexmani_policy.training.common.checkpoint_io import CheckpointStore


class SimEvalRecorder:
    def __init__(self, output_dir: Path, video_fps: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = int(video_fps)

    def save_case_result(self, result: Dict[str, Any], denoise_timesteps: int) -> Dict[str, Any]:
        success_rate = result["success_rate"]
        avg_steps = result["avg_steps"]

        case_dir = self.output_dir / f"denoise_timesteps{denoise_timesteps}"
        case_dir.mkdir(parents=True, exist_ok=True)

        for video_dict in result.get("videos", []):
            for video_name, video_array in video_dict.items():
                video_path = case_dir / f"eval_{video_name}.mp4"
                imageio.mimsave(
                    str(video_path),
                    video_array.astype(np.uint8),
                    fps=self.video_fps,
                )

        case_metrics = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
        }
        episode_details = result.get("episode_details", [])
        if episode_details:
            case_metrics["episode_details"] = episode_details
        self.save_json(case_metrics, case_dir / "metrics.json")
        return case_metrics

    def save_config(self, eval_config: Dict[str, Any]):
        self.save_json(eval_config, self.output_dir / "eval_config.json")

    def save_summary(self, summary: Dict[str, Any]):
        self.save_json(summary, self.output_dir / "eval_metrics.json")

    @staticmethod
    def save_json(data: Dict[str, Any], path: Path):
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
            for key in ('n_obs_steps', 'n_action_steps', 'action_dim', 'horizon'):
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
        recorder = SimEvalRecorder(
            output_dir=eval_run_dir,
            video_fps=self.env_runner.env_video_fps,
        )
        if eval_config is not None:
            recorder.save_config(eval_config)

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
            result = self.env_runner.run(
                self.agent,
                denoise_timesteps=denoise_timesteps,
                eval_episodes=eval_episodes,
            )
            case_metrics = recorder.save_case_result(result, denoise_timesteps=denoise_timesteps)
            summary["metrics"][f"denoise_timesteps{denoise_timesteps}"] = case_metrics

        recorder.save_summary(summary)
        return summary
