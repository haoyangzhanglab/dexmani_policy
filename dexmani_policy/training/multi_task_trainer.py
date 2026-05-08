import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any

from dexmani_policy.training.trainer import Trainer
from dexmani_policy.training.common.logging import to_log_scalars
from dexmani_policy.training.common.checkpoint_io import TrainCheckpoint
from dexmani_policy.common.pytorch_util import optimizer_to, dict_apply


class MultiTaskTrainer(Trainer):
    """
    多任务训练器，继承 Trainer 并 override train() 方法。

    与 Trainer.train() 的差异：
        - 每个 epoch 开始时调用 dataset.set_epoch(epoch)，确保多任务采样的随机性跨 epoch 变化
    """

    @torch.no_grad()
    def compute_action_mse_for_one_batch(self, agent, batch: Dict[str, Any]) -> float:
        obs = batch["obs"]
        gt_action = batch["action"]
        task_ids = batch["task_id"]

        pred_action = torch.zeros_like(gt_action)
        for tid in task_ids.unique():
            mask = (task_ids == tid)
            obs_subset = {k: v[mask] if torch.is_tensor(v) else v for k, v in obs.items()}
            pred = agent.predict_action(obs_subset, task_id=tid.item())["pred_action"]
            pred_action[mask] = pred

        return nn.functional.mse_loss(pred_action, gt_action).item()

    def train(self, resume_tag: str = "latest"):
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

            # 多任务关键：更新 epoch 以改变任务采样的随机种子
            self.train_loader.dataset.set_epoch(epoch)

            self._accum_step = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch in self.train_loader:
                _, log_dict = self.train_one_step(batch)
                global_step += 1

                if (global_step % self.log_interval_steps) == 0:
                    step_metrics = {"train/lr": self.scheduler.get_last_lr()[0]}
                    for key, value in to_log_scalars(log_dict).items():
                        step_metrics[f"train/{key}"] = value

                    epoch_pbar.set_postfix(
                        global_step=global_step,
                        loss=step_metrics.get("train/loss", None),
                    )
                    self.workspace.log(step_metrics, step=global_step)

            # flush 未完成的梯度累积，避免浪费计算
            if self._accum_step % self.gradient_accumulate_every != 0:
                remainder = self._accum_step % self.gradient_accumulate_every
                scale = self.gradient_accumulate_every / remainder
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_ema and self.ema_updater is not None:
                    self.ema_updater.step(self.model)

            self.model.eval()

            epoch_metrics = {}
            epoch_end_tasks = self.plan_epoch_end_tasks(epoch)

            if epoch_end_tasks["sample"]:
                sample_batch = dict_apply(next(iter(self.train_loader)), lambda x: x.to(self.device, non_blocking=True))
                epoch_metrics["train/action_mse_error"] = self.compute_action_mse_for_one_batch(self.model, sample_batch)

            if epoch_end_tasks["validate"]:
                val_agent = self.ema_model if self.use_ema else self.model
                val_loss = self.validate(val_agent)
                if val_loss is not None:
                    epoch_metrics["val/loss"] = val_loss

            if epoch_end_tasks["evaluate"]:
                eval_model = self.ema_model if self.use_ema else self.model
                epoch_metrics.update(self.evaluate(eval_model))

            self.workspace.log(epoch_metrics, step=global_step)

            if epoch_end_tasks["save_checkpoint"]:
                if self.enable_env_eval:
                    test_mean_score = epoch_metrics.get("eval/success_rate")
                elif "val/loss" in epoch_metrics:
                    test_mean_score = -epoch_metrics["val/loss"]
                else:
                    test_mean_score = None

                monitor = {}
                if test_mean_score is not None:
                    monitor["test_mean_score"] = test_mean_score

                checkpoint = TrainCheckpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model_state=self.model.state_dict(),
                    ema_model_state=self.ema_model.state_dict() if self.use_ema else None,
                    ema_updater_state=self.ema_updater.state_dict() if self.use_ema and self.ema_updater else None,
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict(),
                    monitor=monitor,
                )

                if test_mean_score is None:
                    tag = f"epoch={epoch:04d}-step={global_step:08d}"
                    print(f"Warning: No monitor metric at epoch {epoch}, saving as latest only")
                else:
                    tag = f"epoch={epoch:04d}-step={global_step:08d}-score={test_mean_score:.4f}"

                checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
                self.workspace.save_latest(checkpoint_path)

                if test_mean_score is not None:
                    self.workspace.save_topk(checkpoint_path, checkpoint)

            elif epoch_end_tasks["save_latest"]:
                checkpoint = TrainCheckpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model_state=self.model.state_dict(),
                    ema_model_state=self.ema_model.state_dict() if self.use_ema else None,
                    ema_updater_state=self.ema_updater.state_dict() if self.use_ema and self.ema_updater else None,
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict(),
                    monitor={},
                )
                tag = f"epoch={epoch:04d}-step={global_step:08d}"
                checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
                self.workspace.save_latest(checkpoint_path)
