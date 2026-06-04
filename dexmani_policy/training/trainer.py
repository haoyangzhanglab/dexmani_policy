import time
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dexmani_policy.training.common.logging import to_log_scalars
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.checkpoint_io import TrainCheckpoint
from dexmani_policy.common.pytorch_util import optimizer_to, dict_apply


@dataclass
class TrainLoopConfig:
    num_epochs: int
    log_interval_steps: int
    val_interval_epochs: int
    eval_interval_epochs: int
    sample_interval_epochs: int
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0


class Trainer:
    def __init__(
        self,
        device,
        model,
        ema_model,
        ema_updater,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        env_runner,
        workspace: Optional[TrainWorkspace],
        train_loop_cfg: TrainLoopConfig,
        use_ema_teacher_for_consistency: bool,
        is_main_process: bool = True,
    ):
        self.device = device

        self.model = model
        self.ema_model = ema_model
        self.ema_updater = ema_updater

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.env_runner = env_runner
        self.workspace = workspace

        self.num_epochs = train_loop_cfg.num_epochs
        self.log_interval_steps = train_loop_cfg.log_interval_steps
        self.val_interval_epochs = train_loop_cfg.val_interval_epochs
        self.eval_interval_epochs = train_loop_cfg.eval_interval_epochs
        self.sample_interval_epochs = train_loop_cfg.sample_interval_epochs
        self.grad_clip_norm = train_loop_cfg.grad_clip_norm
        self.enable_env_eval = (self.env_runner is not None) and (self.eval_interval_epochs > 0)
        self.checkpoint_interval_epochs = self.eval_interval_epochs if self.enable_env_eval else self.val_interval_epochs

        self.use_ema = self.ema_model is not None
        self.use_ema_teacher_for_consistency = use_ema_teacher_for_consistency and self.use_ema

        self.grad_accum_steps = train_loop_cfg.grad_accum_steps
        self.accum_step = 0
        self.is_main_process = is_main_process
        self.train_sampling_batch = None
        self.current_epoch = -1
        self.global_step = 0

    def plan_epoch_end_tasks(self, epoch: int) -> Dict[str, bool]:
        epoch_idx = epoch + 1
        is_last = epoch_idx == self.num_epochs

        return {
            "sample": is_last or (self.sample_interval_epochs > 0 and epoch_idx % self.sample_interval_epochs == 0),
            "validate": is_last or (self.val_interval_epochs > 0 and epoch_idx % self.val_interval_epochs == 0),
            "evaluate": self.enable_env_eval and (is_last or epoch_idx % self.eval_interval_epochs == 0),
            "save_checkpoint": is_last or (self.checkpoint_interval_epochs > 0 and epoch_idx % self.checkpoint_interval_epochs == 0),
            "save_latest": is_last or (self.val_interval_epochs > 0 and epoch_idx % self.val_interval_epochs == 0),
        }

    def apply_gradient_step(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
        if not torch.isfinite(total_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite gradient norm at epoch={self.current_epoch}, step={self.global_step}: "
                f"total_norm={total_norm}"
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_ema and self.ema_updater is not None:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            self.ema_updater.step(model)

    def flush_gradient_accumulation(self):
        if self.accum_step % self.grad_accum_steps == 0:
            return

        scale = self.grad_accum_steps / (self.accum_step % self.grad_accum_steps)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

        self.apply_gradient_step()

    def _save_nan_debug(self, raw_loss):
        if self.workspace is None:
            return
        output_dir = self.workspace.output_dir
        if output_dir is None:
            return

        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"nan_debug_epoch={self.current_epoch:04d}_step={self.global_step:08d}_{ts}.pt"

        payload = {
            "state": {
                "epoch": int(self.current_epoch),
                "global_step": int(self.global_step),
                "nan_loss": float(raw_loss),
            },
            "weights": {
                "model": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                "ema_model": self.ema_model.state_dict() if self.use_ema else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            "_format": "simple.v1",
            "_saved_at": time.time(),
        }
        torch.save(payload, ckpt_dir / filename)

        return ckpt_dir / filename

    def train_one_step(self, batch: Dict[str, Any]):
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        loss_kwargs = {'ema_backbone': self.ema_model.action_decoder.model} if self.use_ema_teacher_for_consistency else {}
        raw_loss, log_dict = self.model.compute_loss(batch, **loss_kwargs)

        if not torch.isfinite(raw_loss):
            debug_path = self._save_nan_debug(raw_loss)
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite loss at epoch={self.current_epoch}, step={self.global_step}: "
                f"raw_loss={raw_loss.item()}. Debug checkpoint saved to {debug_path}"
            )

        scaled_loss = raw_loss / self.grad_accum_steps
        scaled_loss.backward()

        self.accum_step += 1
        if self.accum_step % self.grad_accum_steps == 0:
            self.apply_gradient_step()

        return batch, log_dict


    @torch.no_grad()
    def validate(self, agent, ema_backbone=None) -> Optional[float]:
        if self.val_loader is None:
            return None

        count = 0
        loss_sum = torch.zeros((), device=self.device)

        for batch in self.val_loader:
            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            loss_kwargs = {'ema_backbone': ema_backbone} if ema_backbone is not None else {}
            loss, log_dict = agent.compute_loss(batch, **loss_kwargs)

            n = batch['action'].shape[0]
            loss_sum += loss.detach() * n
            count += n

        if count == 0:
            return None
        return (loss_sum / count).item()


    @torch.no_grad()
    def evaluate(self, agent) -> Dict[str, Any]:
        result = self.env_runner.run(agent)
        success_rate = result["success_rate"]
        metrics = {
            "eval/success_rate": success_rate * 100 if success_rate is not None else None,
            "eval/avg_steps": result["avg_steps"],
        }
        for item in result.get("videos", []):
            for key, value in item.items():
                metrics[f"eval/{key}_video"] = value
        for key, value in result.items():
            if key.startswith("per_task/") and isinstance(value, (int, float)):
                if key.endswith("success_rate"):
                    value = value * 100
                metrics[f"eval/{key}"] = value
        return metrics


    def finish_epoch(self, epoch, global_step, last_batch=None, eval_model=None, checkpoint_model=None):
        if eval_model is None:
            eval_model = self.ema_model if self.use_ema else self.model
        if checkpoint_model is None:
            checkpoint_model = self.model
        if last_batch is None:
            last_batch = next(iter(self.train_loader))

        epoch_metrics = {}
        epoch_end_tasks = self.plan_epoch_end_tasks(epoch)

        if epoch_end_tasks["sample"]:
            sample_batch = self.train_sampling_batch
            if sample_batch is None:
                sample_batch = dict_apply(last_batch, lambda x: x.to(self.device, non_blocking=True))
            epoch_metrics["sample/action_mse_error"] = eval_model.compute_action_mse(sample_batch)

        if epoch_end_tasks["validate"]:
            if self.use_ema_teacher_for_consistency:
                # FlowMatch: consistency loss requires student (training model) != teacher (EMA)
                model_for_val = self.model
                ema_backbone = self.ema_model.action_decoder.model
            else:
                # DP/DP3: use EMA for val when no env eval, otherwise monitor training model
                model_for_val = self.ema_model if (self.use_ema and not self.enable_env_eval) else self.model
                ema_backbone = None
            val_loss = self.validate(model_for_val, ema_backbone=ema_backbone)
            if val_loss is not None:
                epoch_metrics["val/loss"] = val_loss

        if epoch_end_tasks["evaluate"]:
            epoch_metrics.update(self.evaluate(eval_model))

        if self.workspace is not None:
            self.workspace.log(epoch_metrics, step=global_step)

        if self.workspace is not None and (epoch_end_tasks["save_checkpoint"] or epoch_end_tasks["save_latest"]):
            test_mean_score = None
            if epoch_end_tasks["save_checkpoint"]:
                if self.enable_env_eval:
                    test_mean_score = epoch_metrics.get("eval/success_rate")
                elif "val/loss" in epoch_metrics:
                    test_mean_score = -epoch_metrics["val/loss"]

            # save_latest without save_checkpoint: only update symlink, no new .pt file
            if not epoch_end_tasks["save_checkpoint"]:
                existing = sorted(self.workspace.checkpoint_dir.glob("epoch=*.pt"))
                if existing:
                    self.workspace.save_latest(existing[-1])
                return

            monitor = {}
            if test_mean_score is not None:
                monitor["test_mean_score"] = test_mean_score

            checkpoint = TrainCheckpoint(
                epoch=epoch,
                global_step=global_step,
                model_state=checkpoint_model.state_dict(),
                ema_model_state=self.ema_model.state_dict() if self.use_ema else None,
                ema_updater_state=self.ema_updater.state_dict() if self.use_ema and self.ema_updater else None,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                monitor=monitor,
                train_params={
                    'n_obs_steps': self.model.n_obs_steps,
                    'n_action_steps': self.model.n_action_steps,
                    'action_dim': self.model.action_dim,
                    'horizon': self.model.horizon,
                    'action_mode': getattr(self.model, 'action_mode', 'absolute_joint'),
                },
            )

            tag = f"epoch={epoch:04d}-step={global_step:08d}" if test_mean_score is None \
                  else f"epoch={epoch:04d}-step={global_step:08d}-score={test_mean_score:.4f}"
            checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)

            if test_mean_score is not None:
                self.workspace.save_topk(checkpoint_path, checkpoint)

            latest_target = checkpoint_path
            if test_mean_score is not None and not latest_target.exists():
                latest_target = self.workspace.topk_tracker.best_path()
            self.workspace.save_latest(latest_target)
            if test_mean_score is None and epoch_end_tasks["save_checkpoint"]:
                print(f"Skipping topk update at epoch {epoch} (no monitor metric available)")

    def on_epoch_start(self, epoch: int):
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        self.train_sampling_batch = None

    def train(self, resume_tag: str = "latest"):
        torch.set_float32_matmul_precision('high')
        global_step, start_epoch = self.workspace.load_for_resume(
            model=self.model,
            ema_model=self.ema_model,
            ema_updater=self.ema_updater,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tag_or_path=resume_tag,
        )

        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}, step {global_step}")

        self.model.to(self.device)
        if self.use_ema:
            self.ema_model.to(self.device)
            self.ema_model.eval()

        optimizer_to(self.optimizer, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        epoch_pbar = tqdm(
            range(start_epoch, self.num_epochs),
            desc="Epoch",
            position=0,
            mininterval=1.0,
        )

        for epoch in epoch_pbar:
            self.model.train()
            self.on_epoch_start(epoch)

            self.accum_step = 0
            self.optimizer.zero_grad(set_to_none=True)

            last_batch = None
            for batch in self.train_loader:
                if self.train_sampling_batch is None:
                    self.train_sampling_batch = dict_apply(
                        batch, lambda x: x.to(self.device, non_blocking=True))
                last_batch = batch
                self.current_epoch = epoch
                self.global_step = global_step
                _, log_dict = self.train_one_step(batch)
                global_step += 1

                if self.is_main_process and (global_step % self.log_interval_steps) == 0:
                    step_metrics = {"train/lr": self.scheduler.get_last_lr()[0]}
                    for key, value in to_log_scalars(log_dict).items():
                        step_metrics[f"train/{key}"] = value

                    epoch_pbar.set_postfix(
                        global_step=global_step,
                        loss=step_metrics.get("train/loss", None),
                    )
                    self.workspace.log(step_metrics, step=global_step)

            self.flush_gradient_accumulation()

            self.model.eval()

            if self.is_main_process:
                self.finish_epoch(epoch, global_step, last_batch=last_batch)

