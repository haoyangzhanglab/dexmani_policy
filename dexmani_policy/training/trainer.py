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
        workspace: TrainWorkspace,
        train_loop_cfg: TrainLoopConfig,
        use_ema_teacher_for_consistency: bool,
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
        self.is_main_process = True

    def plan_epoch_end_tasks(self, epoch: int) -> Dict[str, bool]:
        epoch_idx = epoch + 1
        is_last = epoch_idx == self.num_epochs

        return {
            "sample": is_last or epoch_idx % self.sample_interval_epochs == 0,
            "validate": is_last or epoch_idx % self.val_interval_epochs == 0,
            "evaluate": self.enable_env_eval and (is_last or epoch_idx % self.eval_interval_epochs == 0),
            "save_checkpoint": is_last or epoch_idx % self.checkpoint_interval_epochs == 0,
            "save_latest": is_last or epoch_idx % self.val_interval_epochs == 0,
        }

    def apply_gradient_step(self):
        """执行一次梯度更新"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_ema and self.ema_updater is not None:
            self.ema_updater.step(self.model)

    def flush_gradient_accumulation(self):
        if self.accum_step % self.grad_accum_steps == 0:
            return

        scale = self.grad_accum_steps / (self.accum_step % self.grad_accum_steps)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

        self.apply_gradient_step()

    def train_one_step(self, batch: Dict[str, Any]):
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        loss_kwargs = {'ema_model': self.ema_model} if self.use_ema_teacher_for_consistency else {}
        raw_loss, log_dict = self.model.compute_loss(batch, **loss_kwargs)

        scaled_loss = raw_loss / self.grad_accum_steps
        scaled_loss.backward()

        self.accum_step += 1
        if self.accum_step % self.grad_accum_steps == 0:
            self.apply_gradient_step()

        return batch, log_dict


    @torch.no_grad()
    def validate(self, agent) -> Optional[float]:
        if self.val_loader is None:
            return None

        count = 0
        loss_sum = torch.zeros((), device=self.device)

        for batch in self.val_loader:
            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            loss_kwargs = {'ema_model': self.ema_model} if self.use_ema_teacher_for_consistency else {}
            loss, log_dict = agent.compute_loss(batch, **loss_kwargs)

            n = batch['action'].shape[0]
            loss_sum += log_dict['loss_action'].detach() * n
            count += n

        if count == 0:
            return None
        return (loss_sum / count).item()


    @torch.no_grad()
    def evaluate(self, agent) -> Dict[str, Any]:
        # 训练期 eval 使用 env_runner.default_eval_episodes 和 action_decoder 默认 denoise 步数，
        # 不受 cfg.eval.sim 控制（该配置段仅用于独立 eval_sim.py）。
        result = self.env_runner.run(agent)
        success_rate = result["success_rate"]
        metrics = {
            "eval/success_rate": success_rate * 100 if success_rate is not None else None,
            "eval/avg_steps": result["avg_steps"],
        }
        for item in result.get("videos", []):
            for key, value in item.items():
                metrics[f"eval/{key}_video"] = value
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
            sample_batch = dict_apply(last_batch, lambda x: x.to(self.device, non_blocking=True))
            epoch_metrics["train/action_mse_error"] = eval_model.compute_action_mse(sample_batch)

        if epoch_end_tasks["validate"]:
            val_loss = self.validate(self.model)  # 始终用训练模型：ManiFlow 需要其 backbone 做 flow 预测，EMA 作为 teacher 通过 kwargs 传入
            if val_loss is not None:
                epoch_metrics["val/loss"] = val_loss

        if epoch_end_tasks["evaluate"]:
            epoch_metrics.update(self.evaluate(eval_model))

        self.workspace.log(epoch_metrics, step=global_step)

        if epoch_end_tasks["save_checkpoint"] or epoch_end_tasks["save_latest"]:
            test_mean_score = None
            if epoch_end_tasks["save_checkpoint"]:
                if self.enable_env_eval:
                    test_mean_score = epoch_metrics.get("eval/success_rate")
                elif "val/loss" in epoch_metrics:
                    test_mean_score = -epoch_metrics["val/loss"]

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
            )

            tag = f"epoch={epoch:04d}-step={global_step:08d}" if test_mean_score is None \
                  else f"epoch={epoch:04d}-step={global_step:08d}-score={test_mean_score:.4f}"
            checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
            self.workspace.save_latest(checkpoint_path)

            if test_mean_score is not None:
                self.workspace.save_topk(checkpoint_path, checkpoint)
            elif epoch_end_tasks["save_checkpoint"]:
                print(f"Skipping topk update at epoch {epoch} (no monitor metric available)")

    def on_epoch_start(self, epoch: int):
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)

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
                last_batch = batch
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

