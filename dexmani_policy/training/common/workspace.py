import os
import signal
import atexit
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from dexmani_policy.training.common.logging import (
    JsonlLogger, 
    WandbLogger, 
    MultiLogger
)
from dexmani_policy.training.common.checkpoint_io import (
    TrainCheckpoint,
    CheckpointStore,
    TopKCheckpointTracker,
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
    video_fps: int


class TrainWorkspace:
    def __init__(
        self,
        output_dir: str,
        wandb_cfg: WandbConfig,
        checkpoint_cfg: CheckpointConfig
    ):
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

        json_logger = JsonlLogger(output_dir=self.output_dir)
        wandb_logger = WandbLogger(
            output_dir=self.output_dir,
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            group=wandb_cfg.group,
            id=wandb_cfg.id,
            resume=wandb_cfg.resume,
            mode=wandb_cfg.mode,
            video_fps=wandb_cfg.video_fps
        )
        self.logger = MultiLogger([json_logger, wandb_logger])

        self._install_shutdown_hooks()


    def save_hydra_config(self, hydra_config):
        OmegaConf.save(hydra_config, self.output_dir / "config.yaml", resolve=True)


    def _resolve_checkpoint_path(self, tag_or_path: str) -> Path:
        if tag_or_path == "latest":
            path = self.checkpoint_dir / "latest.pt"
        elif tag_or_path == "best":
            path = self.topk_tracker.best_path()
            if path is None:
                raise FileNotFoundError(f"No best checkpoint found in {self.checkpoint_dir}.")
        else:
            path = Path(tag_or_path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path


    def _load_model_weights(self, checkpoint: TrainCheckpoint, model, ema_model=None):
        model.load_state_dict(checkpoint.model_state, strict=True)
        if ema_model is not None and checkpoint.ema_model_state is not None:
            ema_model.load_state_dict(checkpoint.ema_model_state, strict=True)


    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        self.logger.log(data, step=step)


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
        path = self._resolve_checkpoint_path(tag_or_path)
        print("Loading checkpoint from:", path)
        return self.checkpoint_store.load(path)


    def load_for_inference(
        self,
        model,
        tag_or_path: str,
        use_ema: bool = False,
    ):
        checkpoint = self.load_checkpoint(tag_or_path)
        if use_ema and checkpoint.ema_model_state is not None:
            print("Using EMA weights for inference.")
            model.load_state_dict(checkpoint.ema_model_state, strict=True)
        else:
            model.load_state_dict(checkpoint.model_state, strict=True)


    def load_for_resume(
        self,
        model,
        ema_model,
        optimizer,
        scheduler,
        tag_or_path: str,
    ) -> Tuple[int, int]:
        try:
            checkpoint = self.load_checkpoint(tag_or_path)
        except FileNotFoundError:
            return 0, 0

        self._load_model_weights(
            checkpoint=checkpoint,
            model=model,
            ema_model=ema_model,
        )
        optimizer.load_state_dict(checkpoint.optimizer_state)
        scheduler.load_state_dict(checkpoint.scheduler_state)

        # next_step: 已完成的训练步数，训练循环会从此继续递增
        # next_epoch: 下一个要训练的 epoch（checkpoint.epoch + 1）
        next_step = checkpoint.global_step
        next_epoch = checkpoint.epoch + 1
        return next_step, next_epoch


    def close(self):
        self.logger.close()


    def _install_shutdown_hooks(self):
        atexit.register(self.close)
        def _handle_signal(signum, frame):
            try:
                self.close()
            finally:
                raise KeyboardInterrupt
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)


class DummyWorkspace:
    def save_hydra_config(self, hydra_config):
        pass

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        pass

    def save_checkpoint(self, tag: str, checkpoint: TrainCheckpoint) -> Path:
        return Path("dummy")

    def save_latest(self, checkpoint_path: Path) -> Path:
        return Path("dummy")

    def save_topk(self, checkpoint_path: Path, checkpoint: TrainCheckpoint) -> Optional[Path]:
        return None

    def load_checkpoint(self, tag_or_path: str) -> TrainCheckpoint:
        raise NotImplementedError("DummyWorkspace.load_checkpoint should not be called")

    def load_for_resume(
        self,
        model,
        ema_model,
        optimizer,
        scheduler,
        tag_or_path: str,
    ) -> Tuple[int, int]:
        import warnings
        warnings.warn("DummyWorkspace.load_for_resume called - this should only happen in non-main DDP processes")
        return 0, 0

    def close(self):
        pass