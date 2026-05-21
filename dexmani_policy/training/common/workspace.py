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
    _instances: set = set()
    _hooks_installed: bool = False

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

        self._closed = False
        self.__class__._instances.add(self)
        self.install_shutdown_hooks()


    def save_hydra_config(self, hydra_config):
        OmegaConf.save(hydra_config, self.output_dir / "config.yaml", resolve=True)


    def resolve_checkpoint_path(self, tag_or_path: str) -> Path:
        return self.checkpoint_store.resolve_path(tag_or_path, best_fn=self.topk_tracker.best_path)


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
        path = self.resolve_checkpoint_path(tag_or_path)
        print("Loading checkpoint from:", path)
        return self.checkpoint_store.load(path)


    def load_for_inference(
        self,
        model,
        tag_or_path: str,
        use_ema: bool = False,
    ):
        from dexmani_policy.common.pytorch_util import fix_state_dict
        checkpoint = self.load_checkpoint(tag_or_path)
        if use_ema and checkpoint.ema_model_state is not None:
            print("Using EMA weights for inference.")
            model.load_state_dict(fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False), strict=True)
        else:
            if use_ema and checkpoint.ema_model_state is None:
                print("WARNING: EMA weights requested but not found in checkpoint. Using model weights.")
            model.load_state_dict(fix_state_dict(checkpoint.model_state, is_current_ddp=False), strict=True)


    def load_for_resume(
        self,
        model,
        ema_model,
        ema_updater,
        optimizer,
        scheduler,
        tag_or_path: str,
    ) -> Tuple[int, int]:
        try:
            checkpoint = self.load_checkpoint(tag_or_path)
        except FileNotFoundError:
            return 0, 0

        from dexmani_policy.common.pytorch_util import fix_state_dict
        from torch.nn.parallel import DistributedDataParallel as DDP

        is_current_ddp = isinstance(model, DDP)
        model.load_state_dict(fix_state_dict(checkpoint.model_state, is_current_ddp), strict=True)

        if ema_model is not None and checkpoint.ema_model_state is not None:
            ema_model.load_state_dict(fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False), strict=True)

        optimizer.load_state_dict(checkpoint.optimizer_state)
        scheduler.load_state_dict(checkpoint.scheduler_state)

        if ema_updater is not None and checkpoint.ema_updater_state is not None:
            ema_updater.load_state_dict(checkpoint.ema_updater_state)

        next_step = checkpoint.global_step
        next_epoch = checkpoint.epoch + 1
        return next_step, next_epoch


    def close(self):
        if self._closed:
            return
        self._closed = True
        self._instances.discard(self)
        self.logger.close()

    @classmethod
    def _close_all(cls):
        for inst in list(cls._instances):
            inst.close()

    @classmethod
    def _handle_signal(cls, signum, frame):
        cls._close_all()
        raise KeyboardInterrupt

    def install_shutdown_hooks(self):
        # atexit: per-instance 注册无副作用，close() 已幂等
        atexit.register(self.close)
        # signal: 使用类方法统一处理，仅安装一次
        if not self.__class__._hooks_installed:
            self.__class__._hooks_installed = True
            signal.signal(signal.SIGINT, self.__class__._handle_signal)
            signal.signal(signal.SIGTERM, self.__class__._handle_signal)

