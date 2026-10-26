import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import TrainingCheckpoint, save_training_checkpoint
from dexmani_policy.common.runtime import atomic_path, close_resources


def prepare_output_dir(output_dir):
    """Create the output directory, refusing existing training artifacts."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        if (root / name).exists():
            raise FileExistsError(
                f"Training output already exists: {root / name}; use a new output directory"
            )


class Workspace:
    def __init__(self, output_dir: str, wandb_cfg: Mapping | None = None):
        self.output_dir = Path(output_dir)
        prepare_output_dir(self.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"

        self.metrics_file = (self.output_dir / "metrics.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )
        self.wandb_run = None
        if wandb_cfg is not None:
            try:
                os.environ.setdefault("WANDB_SILENT", "true")
                import wandb

                # Continuation creates a new logging run, including repeated sweep names.
                self.wandb_run = wandb.init(
                    dir=str(self.output_dir), id=uuid.uuid4().hex, resume="never", **wandb_cfg
                )
            except BaseException:
                self.close()
                raise

    def save_hydra_config(self, hydra_config):
        for index, recipe in enumerate(hydra_config["data_recipe"]):
            path = self.output_dir / f"split_manifest_{index}.json"
            with atomic_path(path, overwrite=False) as temporary:
                manifest = OmegaConf.to_container(recipe["split_manifest"])
                temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        with atomic_path(self.output_dir / "config.yaml", overwrite=False) as temporary:
            OmegaConf.save(hydra_config, temporary, resolve=True)
        if self.wandb_run is not None:
            cfg_dict = OmegaConf.to_container(hydra_config, resolve=True)
            self.wandb_run.config.update({**cfg_dict, "output_dir": str(self.output_dir)})

    def log(self, data: dict[str, Any], step: int | None = None):
        record = {"step": int(step) if step is not None else None, **data}
        self.metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(data, step=step)

    def save_checkpoint(self, tag: str, checkpoint: TrainingCheckpoint) -> Path:
        return save_training_checkpoint(self.checkpoint_dir / f"{tag}.pt", checkpoint)

    def save_latest(self, checkpoint_path: Path) -> Path:
        latest_path = self.checkpoint_dir / "latest.pt"
        with atomic_path(latest_path) as temporary:
            temporary.unlink()
            temporary.symlink_to(checkpoint_path.name)
        return latest_path

    def close(self):
        callbacks = [self.metrics_file.close]
        if self.wandb_run is not None:
            callbacks.append(self.wandb_run.finish)
        close_resources(*callbacks)
