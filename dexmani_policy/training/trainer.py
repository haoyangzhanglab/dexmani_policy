import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable

from dexmani_policy.training.common.logging import to_log_scalars
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.checkpoint_io import TrainCheckpoint
from dexmani_policy.common.pytorch_util import optimizer_to, dict_apply


# 默认的 loss 参数构造函数，不向 compute_loss 额外传入参数。
def default_loss_kwargs(
    stage: str,
    batch: Dict[str, Any],
    policy,
    model,
    ema_model,
) -> Dict[str, Any]:
    return {}


# 为一致性损失提供 EMA teacher 参数。
def consistency_loss_kwargs(
    stage: str,
    batch: Dict[str, Any],
    policy,
    model,
    ema_model,
) -> Dict[str, Any]:
    if ema_model is None:
        raise ValueError("consistency_loss_kwargs requires ema_model")
    return {"ema_teacher": ema_model}


class Trainer:
    def __init__(
        self,
        device,
        model: nn.Module,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        workspace: TrainWorkspace,
        total_epochs: int,
        val_every: int,
        eval_every: int,
        sample_every: int,
        log_every_steps: int = 20,
        ema_model: Optional[nn.Module] = None,
        ema_updater=None,
        loss_kwargs_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        env_runner=None,
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

        self.total_epochs = int(total_epochs)
        self.val_every = int(val_every)
        self.eval_every = int(eval_every)
        self.sample_every = int(sample_every)
        self.log_every_steps = int(log_every_steps)

        self.loss_kwargs_fn = loss_kwargs_fn or default_loss_kwargs

        self.use_ema = self.ema_model is not None
        self.online_eval = (self.env_runner is not None) and (self.eval_every > 0)
        self.save_every = self.eval_every if self.online_eval else self.val_every


    def _plan_epoch_end_operations(self, epoch: int) -> Dict[str, bool]:
        is_last_epoch = (epoch == self.total_epochs - 1)
        return {
            "sample": ((epoch + 1) % self.sample_every == 0) or is_last_epoch,
            "validate": ((epoch + 1) % self.val_every == 0) or is_last_epoch,
            "evaluate": self.online_eval and (((epoch + 1) % self.eval_every == 0) or is_last_epoch),
            "save": ((epoch + 1) % self.save_every == 0) or is_last_epoch,
        }


    def _train_one_step(self, batch: Dict[str, Any]):
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        loss_kwargs = self.loss_kwargs_fn(
            stage="train",
            batch=batch,
            policy=self.model,
            model=self.model,
            ema_model=self.ema_model,
        )
        loss, log_dict = self.model.compute_loss(batch, **loss_kwargs)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_ema and self.ema_updater is not None:
            self.ema_updater.step(self.model)

        return batch, log_dict


    @torch.no_grad()
    def _validate(self, policy) -> Optional[float]:
        loss_sum = torch.zeros((), device=self.device)
        count = 0

        for batch in self.val_loader:
            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            loss_kwargs = self.loss_kwargs_fn(
                stage="val",
                batch=batch,
                policy=policy,
                model=self.model,
                ema_model=self.ema_model,
            )
            loss, _ = policy.compute_loss(batch, **loss_kwargs)

            loss_sum += loss.detach()
            count += 1

        if count == 0:
            return None

        return (loss_sum / count).item()


    @torch.no_grad()
    def _sample_train_batch_mse(self, policy, batch: Dict[str, Any]) -> float:
        obs = batch["obs"]
        gt_action = batch["action"]
        pred_action = policy.predict_action(obs)
        return nn.functional.mse_loss(pred_action, gt_action).item()


    @torch.no_grad()
    def _evaluate(self, policy) -> Dict[str, Any]:
        result = self.env_runner.run(policy)
        metrics = {
            "eval/success_rate": result["success_rate"],
            "eval/avg_done_steps": result["avg_done_steps"],
        }
        for item in result.get("video", []):
            for key, value in item.items():
                metrics[f"eval/rollout_{key}_video"] = value
        return metrics


    # 执行完整训练流程，并在需要时从已有 checkpoint 断点续训。
    def train(self, resume_tag: Optional[str] = "latest"):
        global_step, start_epoch = self.workspace.load_for_resume(
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tag_or_path=resume_tag,
        )

        self.model.to(self.device)
        if self.use_ema:
            self.ema_model.to(self.device)
            self.ema_model.eval()

        optimizer_to(self.optimizer, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        epoch_pbar = tqdm(
            range(start_epoch, self.total_epochs),
            desc="Epoch",
            position=0,
            mininterval=1.0,
        )

        for epoch in epoch_pbar:
            self.model.train()
            last_batch = None

            for batch in self.train_loader:
                batch, log_dict = self._train_one_step(batch)

                if (global_step % self.log_every_steps) == 0:
                    step_metrics = {"train/lr": self.scheduler.get_last_lr()[0]}
                    for key, value in to_log_scalars(log_dict).items():
                        step_metrics[f"train/{key}"] = value

                    self.workspace.log(step_metrics, step=global_step)
                    epoch_pbar.set_postfix(
                        global_step=global_step,
                        loss=step_metrics.get("train/loss", None),
                    )

                global_step += 1
                last_batch = batch

            self.model.eval()
            policy = self.ema_model if self.use_ema else self.model
            epoch_end_ops = self._plan_epoch_end_operations(epoch)
            epoch_metrics = {}

            if epoch_end_ops["sample"] and last_batch is not None:
                epoch_metrics["train/action_mse_error"] = self._sample_train_batch_mse(policy, last_batch)

            if epoch_end_ops["validate"]:
                val_loss = self._validate(policy)
                if val_loss is not None:
                    epoch_metrics["val/loss"] = val_loss

            if epoch_end_ops["evaluate"]:
                epoch_metrics.update(self._evaluate(policy))

            self.workspace.log(epoch_metrics, step=global_step)

            if epoch_end_ops["save"]:
                if self.online_eval:
                    test_mean_score = epoch_metrics["eval/success_rate"]
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
                else:
                    tag = f"epoch={epoch:04d}-step={global_step:08d}-score={test_mean_score:.4f}"

                checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
                self.workspace.save_latest(checkpoint_path)

                if test_mean_score is not None:
                    self.workspace.save_topk(checkpoint_path, checkpoint)
