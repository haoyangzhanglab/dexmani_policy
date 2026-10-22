from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.training.common.logging import JsonlLogger, WandbLogger


def prepare_output_dir(output_dir):
    """Create the output directory, refusing existing training artifacts."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        if (root / name).exists():
            raise FileExistsError(
                f"Training output already exists: {root / name}; use a new output directory"
            )


@dataclass
class WandbConfig:
    project: str
    group: str
    name: str
    mode: str


class Workspace:
    def __init__(self, output_dir: str, wandb_cfg: WandbConfig | None = None):
        self.output_dir = Path(output_dir)
        prepare_output_dir(self.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"

        self.checkpoint_store = CheckpointStore(self.checkpoint_dir)

        self.json_logger = JsonlLogger(output_dir=self.output_dir)
        self.wandb_logger = None
        if wandb_cfg is not None:
            try:
                self.wandb_logger = WandbLogger(
                    output_dir=self.output_dir,
                    project=wandb_cfg.project,
                    name=wandb_cfg.name,
                    group=wandb_cfg.group,
                    mode=wandb_cfg.mode,
                )
            except Exception:
                self.json_logger.close()
                raise

        self.closed = False

    def save_hydra_config(self, hydra_config):
        with atomic_path(self.output_dir / "config.yaml") as temporary:
            OmegaConf.save(hydra_config, temporary, resolve=True)
        if self.wandb_logger is not None:
            cfg_dict = OmegaConf.to_container(hydra_config, resolve=True)
            self.wandb_logger.log_config(cfg_dict)

    def log(self, data: dict[str, Any], step: int | None = None):
        self.json_logger.log(data, step=step)
        if self.wandb_logger is not None:
            self.wandb_logger.log(data, step=step)

    def save_checkpoint(self, tag: str, checkpoint: TrainCheckpoint) -> Path:
        filename = tag if str(tag).endswith(".pt") else f"{tag}.pt"
        return self.checkpoint_store.save(filename, checkpoint)

    def save_latest(self, checkpoint_path: Path) -> Path:
        latest_path = self.checkpoint_dir / "latest.pt"
        with atomic_path(latest_path) as temporary:
            temporary.unlink()
            temporary.symlink_to(checkpoint_path.name)
        return latest_path

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.json_logger.close()
        finally:
            if self.wandb_logger is not None:
                self.wandb_logger.close()
