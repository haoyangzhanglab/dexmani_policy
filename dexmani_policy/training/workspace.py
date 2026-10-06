import atexit
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from dexmani_policy.training.run_identity import claim_run, check_run_claim
from dexmani_policy.training.checkpoint import CheckpointStore, TrainCheckpoint
from dexmani_policy.training.logging import (
    JsonlLogger,
    WandbLogger,
)


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
    def __init__(self, output_dir: str, wandb_cfg: WandbConfig | None = None, claim_token=None):
        self.output_dir = Path(output_dir)
        if claim_token is None:
            claim_token = claim_run(self.output_dir)
        check_run_claim(self.output_dir, claim_token)
        from dexmani_policy.training.source_snapshot import save_source_snapshot
        save_source_snapshot(self.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"

        self.checkpoint_store = CheckpointStore(self.checkpoint_dir)

        self.json_logger = JsonlLogger(output_dir=self.output_dir)
        # Include the permanent claim identity: sweep basenames such as "0"
        # repeat across launches and cannot identify a W&B run on their own.
        self.wandb_logger = None
        if wandb_cfg is not None:
            wandb_id = f"{wandb_cfg.id}_{self.output_dir.name}_{claim_token[:8]}"
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
        if self.wandb_logger is not None:
            self.wandb_logger.log_config(cfg_dict, self.output_dir)

    def resolve_checkpoint_path(self, tag_or_path: str) -> Path:
        return self.checkpoint_store.resolve_path(tag_or_path)

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        self.json_logger.log(data, step=step)
        if self.wandb_logger is not None:
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

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.json_logger.close()
        if self.wandb_logger is not None:
            self.wandb_logger.close()
