import json
import torch
import imageio
import numpy as np
from pathlib import Path
from termcolor import cprint
from datetime import datetime
from typing import Dict, List, Any, Optional

from dexmani_policy.training.common.workspace import TrainWorkspace


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
        self.save_json(case_metrics, case_dir / "metrics.json")
        return case_metrics

    def save_config(self, eval_config: Dict[str, Any]):
        self.save_json(eval_config, self.output_dir / "eval_config.json")

    def save_summary(self, summary: Dict[str, Any]):
        self.save_json(summary, self.output_dir / "eval_metrics.json")

    @staticmethod
    def save_json(data: Dict[str, Any], path: Path):
        def _default(obj):
            try:
                if isinstance(obj, np.generic):
                    return obj.item()
            except Exception:
                pass
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


class SimEvaluator:
    def __init__(
        self,
        device,
        agent,
        env_runner,
        workspace: TrainWorkspace,
    ):
        self.device = device
        self.agent = agent
        self.env_runner = env_runner
        self.workspace = workspace
        self.eval_root_dir = self.workspace.output_dir / "eval"
        self.eval_root_dir.mkdir(parents=True, exist_ok=True)

    def create_eval_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = self.eval_root_dir / f"{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @torch.no_grad()
    def run(
        self,
        eval_episodes: int,
        denoise_timesteps_list: List[int],
        ckpt_tag_or_path: str = "latest",
        use_ema_for_eval: bool = True,
        eval_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.workspace.load_for_inference(
            model=self.agent,
            tag_or_path=ckpt_tag_or_path,
            use_ema=use_ema_for_eval,
        )
        self.agent.to(self.device)
        self.agent.eval()

        eval_run_dir = self.create_eval_run_dir()
        recorder = SimEvalRecorder(
            output_dir=eval_run_dir,
            video_fps=self.env_runner.video_fps,
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

        cprint("=" * 30 + " eval in sim ... " + "=" * 30, "red")
        for steps in denoise_timesteps_list:
            result = self.env_runner.run(
                self.agent,
                denoise_timesteps=steps,
                episodes=eval_episodes,
            )
            case_metrics = recorder.save_case_result(result, denoise_timesteps=steps)
            summary["metrics"][f"denoise_timesteps{steps}"] = case_metrics
        cprint("=" * 30 + " saving data ... " + "=" * 30, "red")

        recorder.save_summary(summary)
        return summary
