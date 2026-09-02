import atexit
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    TopKCheckpointTracker,
    TrainCheckpoint,
)
from dexmani_policy.training.logging import (
    JsonlLogger,
    WandbLogger,
)


@dataclass
class CheckpointConfig:
    monitor_key: str
    mode: str
    topk: int


@dataclass
class WandbConfig:
    project: str
    group: str
    name: str
    id: str
    resume: str
    mode: str
    video_fps: int = 15


class TrainWorkspace:
    def __init__(self, output_dir: str, wandb_cfg: WandbConfig, checkpoint_cfg: CheckpointConfig):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"

        self.checkpoint_store = CheckpointStore(self.checkpoint_dir)

        self.topk_tracker = TopKCheckpointTracker(
            checkpoint_dir=self.checkpoint_dir,
            monitor_key=checkpoint_cfg.monitor_key,
            mode=checkpoint_cfg.mode,
            k=checkpoint_cfg.topk,
        )

        self.json_logger = JsonlLogger(output_dir=self.output_dir)
        # Derive the W&B id from the experiment identity (the timestamped
        # output_dir basename) rather than the policy+task+seed triple alone.
        # A fixed triple collides across re-runs and would re-attach a fresh
        # run to a stale remote run under resume="allow".
        wandb_id = f"{wandb_cfg.id}_{self.output_dir.name}"
        self.wandb_logger = WandbLogger(
            output_dir=self.output_dir,
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            group=wandb_cfg.group,
            id=wandb_id,
            resume=wandb_cfg.resume,
            mode=wandb_cfg.mode,
            video_fps=wandb_cfg.video_fps,
        )

        self._closed = False
        atexit.register(self.close)

    def save_hydra_config(self, hydra_config):
        OmegaConf.save(hydra_config, self.output_dir / "config.yaml", resolve=True)
        cfg_dict = OmegaConf.to_container(hydra_config, resolve=True)
        self.wandb_logger.log_config(cfg_dict, self.output_dir)

    def resolve_checkpoint_path(self, tag_or_path: str) -> Path:
        return self.checkpoint_store.resolve_path(tag_or_path, best_fn=self.topk_tracker.best_path)

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        self.json_logger.log(data, step=step)
        self.wandb_logger.log(data, step=step)

    def save_checkpoint(self, tag: str, checkpoint: TrainCheckpoint) -> Path:
        filename = tag if str(tag).endswith(".pt") else f"{tag}.pt"
        return self.checkpoint_store.save(filename, checkpoint)

    def save_latest(self, checkpoint_path: Path) -> Path:
        latest_path = self.checkpoint_dir / "latest.pt"
        tmp_path = latest_path.with_suffix(".tmp.pt")
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink()
        tmp_path.symlink_to(checkpoint_path.name)
        os.replace(tmp_path, latest_path)
        return latest_path

    def save_topk(self, checkpoint_path: Path, checkpoint: TrainCheckpoint) -> Optional[Path]:
        return self.topk_tracker.update(checkpoint_path, checkpoint)

    def load_checkpoint(self, tag_or_path: str) -> TrainCheckpoint:
        path = self.resolve_checkpoint_path(tag_or_path)
        print("Loading checkpoint from:", path)
        return self.checkpoint_store.load(path)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.json_logger.close()
        self.wandb_logger.close()
