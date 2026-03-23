import os
import shutil
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

def load_cfg_from_experiment(exp_dir: Path):
    exp_dir = exp_dir.resolve()
    cfg_path = exp_dir / "config.yaml"
    assert cfg_path.is_file(), f"Can't find config.yaml: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)
    return cfg, cfg_path


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


    # 将一条训练日志写入当前工作区的日志后端。
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        self.logger.log(data, step=step)


    # 将当前训练状态保存为一个 checkpoint 文件。
    def save_checkpoint(self, tag: str, checkpoint: TrainCheckpoint) -> Path:
        filename = tag if str(tag).endswith(".pt") else f"{tag}.pt"
        return self.checkpoint_store.save(filename, checkpoint)


    # 将指定 checkpoint 更新为 latest.pt，便于后续断点续训。
    def save_latest(self, checkpoint_path: Path) -> Path:
        latest_path = self.checkpoint_dir / "latest.pt"
        tmp_path = latest_path.with_suffix(".tmp")
        try:
            os.link(checkpoint_path, tmp_path)
        except OSError:
            shutil.copy2(checkpoint_path, tmp_path)
        os.replace(tmp_path, latest_path)
        return latest_path


    # 根据监控指标更新 top-k checkpoint 清单，并返回当前 best 路径。
    def save_topk(self, checkpoint_path: Path, checkpoint: TrainCheckpoint) -> Optional[Path]:
        return self.topk_tracker.update(checkpoint_path, checkpoint)


    # 按标签或路径加载一个 checkpoint。
    def load_checkpoint(self, tag_or_path: str) -> TrainCheckpoint:
        path = self._resolve_checkpoint_path(tag_or_path)
        print("Loading checkpoint from:", path)
        return self.checkpoint_store.load(path)


    # 为推理加载模型权重，可按需优先使用 EMA 权重。
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


    # 为断点续训恢复模型、优化器和调度器状态，并返回起始 step 和 epoch。
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

        start_step = checkpoint.global_step
        start_epoch = checkpoint.epoch + 1
        return start_step, start_epoch


    # 关闭日志后端并释放当前工作区资源。
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