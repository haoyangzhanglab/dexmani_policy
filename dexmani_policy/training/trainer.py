import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dexmani_policy.training.common.logging import to_log_scalars
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.checkpoint_io import TrainCheckpoint
from dexmani_policy.common.pytorch_util import optimizer_to, dict_apply
from dexmani_policy.training.common.ddp_util import unwrap_model


@dataclass
class TrainLoopConfig:
    num_epochs: int
    log_interval_steps: int
    val_interval_epochs: int
    eval_interval_epochs: int
    sample_interval_epochs: int
    gradient_accumulate_every: int = 1
    grad_clip_norm: float = 1.0

    def __post_init__(self):
        if self.gradient_accumulate_every < 1:
            raise ValueError(f"gradient_accumulate_every must be >= 1, got {self.gradient_accumulate_every}")
    
    
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
        self.gradient_accumulate_every = train_loop_cfg.gradient_accumulate_every
        self.grad_clip_norm = train_loop_cfg.grad_clip_norm
        self.enable_env_eval = (self.env_runner is not None) and (self.eval_interval_epochs > 0)
        self.checkpoint_interval_epochs = self.eval_interval_epochs if self.enable_env_eval else self.val_interval_epochs

        self.use_ema = self.ema_model is not None
        self.use_ema_teacher_for_consistency = use_ema_teacher_for_consistency and self.use_ema
        self.accum_step = 0
        

    def plan_epoch_end_tasks(self, epoch: int) -> Dict[str, bool]:
        epoch_idx = epoch + 1
        is_last_epoch = epoch_idx == self.num_epochs

        should_sample = is_last_epoch or (epoch_idx % self.sample_interval_epochs == 0)
        should_validate = is_last_epoch or (epoch_idx % self.val_interval_epochs == 0)
        should_evaluate = self.enable_env_eval and (
            is_last_epoch or (epoch_idx % self.eval_interval_epochs == 0)
        )
        should_save = is_last_epoch or (epoch_idx % self.checkpoint_interval_epochs == 0)

        return {
            "sample": should_sample,
            "validate": should_validate,
            "evaluate": should_evaluate,
            "save_checkpoint": should_save,
        }


    def train_one_step(self, batch: Dict[str, Any]):
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        loss_kwargs = {'ema_model': self.ema_model} if self.use_ema_teacher_for_consistency else {}
        raw_loss, log_dict = self.model.compute_loss(batch, **loss_kwargs)

        loss = raw_loss / self.gradient_accumulate_every
        loss.backward()

        self.accum_step += 1

        if self.accum_step % self.gradient_accumulate_every == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_ema and self.ema_updater is not None:
                model_to_update = unwrap_model(self.model)
                self.ema_updater.step(model_to_update)

        return batch, log_dict


    @torch.no_grad()
    def validate(self, agent) -> Optional[float]:
        if self.val_loader is None:
            return None

        count = 0
        loss_sum = torch.zeros((), device=self.device)

        for batch in self.val_loader:
            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            # 不传入 ema_model：验证时 agent 已经是 ema_model（见 train() 中 val_agent 的赋值）
            # 如果传入会导致 teacher=student=ema_model，consistency loss 退化为自我一致性检查，失去 distillation 意义
            # 当前只计算 flow loss，用于监控 flow matching 的泛化（checkpoint 选择基于 success_rate，不依赖 val_loss）
            loss_kwargs = {}
            loss, _ = agent.compute_loss(batch, **loss_kwargs)

            n = batch['action'].shape[0]
            loss_sum += loss.detach() * n
            count += n

        if count == 0:
            return None
        return (loss_sum / count).item()


    @torch.no_grad()
    def compute_action_mse_for_one_batch(self, agent, batch: Dict[str, Any]) -> float:
        obs = batch["obs"]
        gt_action = batch["action"]
        pred_action = agent.predict_action(obs)["pred_action"]
        return nn.functional.mse_loss(pred_action, gt_action).item()


    @torch.no_grad()
    def evaluate(self, agent) -> Dict[str, Any]:
        result = self.env_runner.run(agent)
        metrics = {
            "eval/success_rate": result["success_rate"] * 100,
            "eval/avg_steps": result["avg_steps"],
        }
        for item in result.get("videos", []):
            for key, value in item.items():
                metrics[f"eval/{key}_video"] = value
        return metrics


    def train(self, resume_tag: str = "latest"):
        global_step, start_epoch = self.workspace.load_for_resume(
            model=self.model,
            ema_model=self.ema_model,
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
