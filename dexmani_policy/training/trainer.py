import contextlib
import time
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dexmani_policy.common.pytorch_util import compile_models, optimizer_to, dict_apply, fix_state_dict, to_log_scalars
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.checkpoint_io import TrainCheckpoint


@dataclass
class TrainLoopConfig:
    num_epochs: int
    log_interval_steps: int
    val_interval_epochs: int
    eval_interval_epochs: int
    sample_interval_epochs: int
    gradient_accumulation_steps: int = 1


class Trainer:
    """Main training loop with EMA, validation, evaluation, and checkpointing.

    Orchestrates the full training lifecycle across ``num_epochs``:

    - **Training**: ``train_one_step()`` with mixed precision (bfloat16 AMP),
      gradient clipping, and three-layer NaN protection (loss NaN → skip,
      grad NaN → skip, DDP all_reduce sentinel).
    - **Validation**: Run validation loss every ``val_interval_epochs``.
    - **Evaluation**: Run sim evaluation every ``eval_interval_epochs`` via
      ``env_runner`` (if configured).
    - **Checkpointing**: Save at epoch end; top-K tracking by
      ``test_mean_score``; latest symlink for resume.
    - **EMA**: Exponential moving average of model weights, updated each step.

    Supports single-GPU and DDP (via ``distributed=True``). In DDP, only rank
    0 performs logging, checkpointing, and evaluation.
    """
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
        max_grad_norm: float = 1.0,
        use_bfloat16: bool = False,
        use_compile: bool = False,
        is_main_process: bool = True,
        distributed: bool = False,
        train_sampler = None,
        num_training_steps: Optional[int] = None,
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
        self.enable_env_eval = (self.env_runner is not None) and (self.eval_interval_epochs > 0)
        self.checkpoint_interval_epochs = self.eval_interval_epochs if self.enable_env_eval else self.val_interval_epochs
        self.max_grad_norm = max_grad_norm

        self.use_ema = self.ema_model is not None
        self.use_ema_teacher_for_consistency = use_ema_teacher_for_consistency and self.use_ema

        self.use_bfloat16 = use_bfloat16
        self.use_compile = use_compile

        self.gradient_accumulation_steps = max(1, int(train_loop_cfg.gradient_accumulation_steps))
        # Pre-compute AMP device_type string to avoid repeated str.split on every step
        self.amp_device_type = str(self.device).split(':')[0]

        self.is_main_process = is_main_process
        self.distributed = distributed
        self.train_sampler = train_sampler
        self.train_sampling_batch = None
        self.current_epoch = -1
        self.global_step = 0
        self.num_training_steps = num_training_steps

    @property
    def raw_model(self):
        """Return the unwrapped base model (no DDP, no torch.compile).

        NOTE: ``isinstance(self.raw_model, DDP)`` will always be False
        because this property already unwraps DDP.  When a DDP‑wrapped
        model is in use, check ``self.distributed`` or the original
        ``self.model`` attribute instead.
        """
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def apply_gradient_step(self):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(), max_norm=self.max_grad_norm
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_ema and self.ema_updater is not None:
            self.ema_updater.step(self.raw_model)

    def load_for_resume(self, tag_or_path: str = "latest"):
        """Restore model/EMA/optimizer/scheduler from a checkpoint. Returns (global_step, start_epoch)."""
        try:
            checkpoint = self.workspace.load_checkpoint(tag_or_path)
        except FileNotFoundError:
            return 0, 0

        is_current_ddp = isinstance(self.raw_model, DDP)
        self.raw_model.load_state_dict(fix_state_dict(checkpoint.model_state, is_current_ddp), strict=True)

        if self.use_ema and checkpoint.ema_model_state is not None:
            self.ema_model.load_state_dict(fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False), strict=True)

        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.scheduler.load_state_dict(checkpoint.scheduler_state)

        # Validate num_training_steps consistency on resume: a mismatch means the
        # dataloader config changed between runs (e.g. batch_size, num_workers),
        # which silently shifts the LR schedule curve even after load_state_dict.
        saved_steps = checkpoint.train_params.get('num_training_steps') if checkpoint.train_params else None
        current_steps = self.num_training_steps
        if saved_steps is not None and current_steps is not None and saved_steps != current_steps:
            import warnings
            warnings.warn(
                f"Resume: num_training_steps mismatch — saved={saved_steps}, current={current_steps}. "
                f"The LR schedule was originally configured for {saved_steps} total steps; "
                f"the current config would produce {current_steps}. "
                f"The scheduler state_dict has been restored from the checkpoint, but the "
                f"underlying schedule curve may be distorted. "
                f"Consider matching the original dataloader configuration to avoid LR drift.",
                UserWarning,
            )

        return checkpoint.global_step, checkpoint.epoch + 1

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
            "state": {"epoch": int(self.current_epoch), "global_step": int(self.global_step), "nan_loss": float(raw_loss)},
            "weights": {
                "model": self.raw_model.state_dict(),
                "ema_model": self.ema_model.state_dict() if self.use_ema else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            "_format": "simple.v1",
            "_saved_at": time.time(),
        }
        torch.save(payload, ckpt_dir / filename)
        return ckpt_dir / filename

    def train_one_step(self, batch: Dict[str, Any], *, is_accumulation_boundary: bool = True):
        """Forward + backward on one micro-batch.

        When ``gradient_accumulation_steps > 1`` the loss is scaled by
        ``1 / gradient_accumulation_steps`` and ``optimizer.step()`` /
        ``scheduler.step()`` / EMA are deferred until the accumulation
        boundary (``is_accumulation_boundary=True``).

        Parameters:
            batch: Data dict from the DataLoader.
            is_accumulation_boundary: If ``True``, apply gradient step after
                backward.  Set to ``False`` for intermediate micro-batches
                when accumulating gradients.
        """
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        loss_kwargs = {'ema_backbone': self.ema_model.action_decoder.model} if self.use_ema_teacher_for_consistency else {}
        with torch.amp.autocast(device_type=self.amp_device_type, dtype=torch.bfloat16, enabled=self.use_bfloat16):
            raw_loss, log_dict = self.model.compute_loss(batch, **loss_kwargs)

        if self.distributed:
            nan_flag = torch.tensor(
                [0 if torch.isfinite(raw_loss) else 1],
                dtype=torch.int, device=self.device,
            )
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            is_nan = bool(nan_flag.item())
        else:
            is_nan = not torch.isfinite(raw_loss)

        if is_nan:
            debug_path = self._save_nan_debug(raw_loss)
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite loss at epoch={self.current_epoch}, step={self.global_step}: "
                f"raw_loss={raw_loss.item()}. Debug checkpoint saved to {debug_path}"
            )

        # Scale loss so that the *sum* of micro-batch gradients equals the
        # gradient of the full batch (loss averaged across micro-batches).
        (raw_loss / self.gradient_accumulation_steps).backward()

        if is_accumulation_boundary:
            self.apply_gradient_step()

        return batch, log_dict


    @torch.no_grad()
    def validate(self, agent, ema_backbone=None):
        """Return dict with at least ``"loss"``; ``"loss_flow"`` and
        ``"loss_consistency"`` are included when the action decoder reports
        them (e.g. FlowMatchWithConsistency)."""
        if self.val_loader is None:
            return None

        count = 0
        loss_sum = torch.zeros((), device=self.device)
        flow_sum = torch.zeros((), device=self.device)
        cons_sum = torch.zeros((), device=self.device)
        has_components = False

        for batch in self.val_loader:
            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            loss_kwargs = {'ema_backbone': ema_backbone} if ema_backbone is not None else {}
            with torch.amp.autocast(device_type=self.amp_device_type, dtype=torch.bfloat16, enabled=self.use_bfloat16):
                loss, log_dict = agent.compute_loss(batch, **loss_kwargs)

            n = batch['action'].shape[0]
            loss_sum += loss.detach() * n
            if 'loss_flow' in log_dict:
                flow_sum += log_dict['loss_flow'].detach() * n
                cons_sum += log_dict['loss_consistency'].detach() * n
                has_components = True
            count += n

        if count == 0:
            return None

        if self.distributed:
            stats = torch.tensor(
                [loss_sum.item(), flow_sum.item(), cons_sum.item(), float(count)],
                device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            c = stats[3].item()
            result = {"loss": (stats[0] / c).item()}
            if has_components:
                result["loss_flow"] = (stats[1] / c).item()
                result["loss_consistency"] = (stats[2] / c).item()
            return result

        result = {"loss": (loss_sum / count).item()}
        if has_components:
            result["loss_flow"] = (flow_sum / count).item()
            result["loss_consistency"] = (cons_sum / count).item()
        return result


    @torch.no_grad()
    def evaluate(self, agent) -> Dict[str, Any]:
        result = self.env_runner.run(agent)
        success_rate = result["success_rate"]
        metrics = {
            "eval/success_rate": success_rate * 100 if success_rate is not None else None,
            "eval/avg_steps": result["avg_steps"],
            "eval/avg_steps_all": result.get("avg_steps_all"),
        }
        for item in result.get("videos", []):
            for key, value in item.items():
                metrics[f"eval/{key}_video"] = value
        per_task = result.get("per_task", {})
        for task_name, task_result in per_task.items():
            sr = task_result.get("success_rate")
            if sr is not None:
                metrics[f"eval/per_task/{task_name}/success_rate"] = sr * 100
                metrics[f"eval/per_task/{task_name}/avg_steps"] = task_result.get("avg_steps")
        return metrics


    def _should_run(self, epoch, interval):
        """Return True if task should run at this epoch: last epoch or interval-aligned."""
        is_last = epoch + 1 == self.num_epochs
        return is_last or (interval > 0 and (epoch + 1) % interval == 0)

    def _sample_and_log(self, epoch, eval_model, last_batch):
        sample_batch = self.train_sampling_batch
        if sample_batch is None:
            sample_batch = dict_apply(last_batch, lambda x: x.to(self.device, non_blocking=True))
        with torch.amp.autocast(device_type=self.amp_device_type, dtype=torch.bfloat16, enabled=self.use_bfloat16):
            return {"sample/action_mse_error": eval_model.compute_action_mse(sample_batch)}

    def _validate_and_log(self, epoch):
        if self.val_loader is None:
            return {}
        if self.use_ema_teacher_for_consistency:
            model_for_val = self.model
            ema_backbone = self.ema_model.action_decoder.model
        else:
            model_for_val = self.ema_model if (self.use_ema and not self.enable_env_eval) else self.model
            ema_backbone = None
        result = self.validate(model_for_val, ema_backbone=ema_backbone)
        if result is None:
            return {}
        metrics = {"val/loss": result["loss"]}
        if "loss_flow" in result:
            metrics["val/loss_flow"] = result["loss_flow"]
            metrics["val/loss_consistency"] = result["loss_consistency"]
        return metrics

    def _evaluate_and_log(self, eval_model):
        return self.evaluate(eval_model) if self.env_runner is not None else {}

    def _save_epoch_checkpoint(self, epoch, global_step, checkpoint_model, test_mean_score):
        """Build and save a checkpoint, update topk tracker, and refresh latest symlink."""
        monitor = {"test_mean_score": test_mean_score} if test_mean_score is not None else {}
        # checkpoint_model is already self.raw_model (unwrapped), but EMA may still
        # be wrapped by torch.compile – unwrap before serialising to keep keys clean.
        checkpoint = TrainCheckpoint(
            epoch=epoch,
            global_step=global_step,
            model_state=checkpoint_model.state_dict(),
            ema_model_state=fix_state_dict(self.ema_model.state_dict(), is_current_ddp=False) if self.use_ema else None,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            monitor=monitor,
            train_params={
                'n_obs_steps': self.model.n_obs_steps,
                'n_action_steps': self.model.n_action_steps,
                'action_dim': self.model.action_dim,
                'horizon': self.model.horizon,
                'action_key': getattr(self.model, 'action_key', 'action'),
                'num_training_steps': self.num_training_steps,
            },
        )
        tag = f"epoch={epoch:04d}-step={global_step:08d}"
        if test_mean_score is not None:
            tag += f"-score={test_mean_score:.4f}"
        checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)

        if test_mean_score is not None:
            self.workspace.save_topk(checkpoint_path, checkpoint)

        latest_target = checkpoint_path
        if test_mean_score is not None:
            best = self.workspace.topk_tracker.best_path()
            if best is not None:
                latest_target = best
        self.workspace.save_latest(latest_target)

    def finish_epoch(self, epoch, global_step, last_batch=None, eval_model=None, checkpoint_model=None):
        eval_model = eval_model or (self.ema_model if self.use_ema else self.raw_model)
        checkpoint_model = checkpoint_model or self.raw_model
        last_batch = last_batch or next(iter(self.train_loader))

        epoch_metrics = {}

        if self._should_run(epoch, self.sample_interval_epochs):
            epoch_metrics.update(self._sample_and_log(epoch, eval_model, last_batch))

        if self._should_run(epoch, self.val_interval_epochs):
            epoch_metrics.update(self._validate_and_log(epoch))

        should_eval = self._should_run(epoch, self.eval_interval_epochs)
        if should_eval:
            epoch_metrics.update(self._evaluate_and_log(eval_model))

        if self.workspace is not None:
            self.workspace.log(epoch_metrics, step=global_step)

        should_save = self._should_run(epoch, self.checkpoint_interval_epochs)
        should_save_latest = self._should_run(epoch, self.val_interval_epochs)

        if self.workspace is not None and (should_save or should_save_latest):
            if should_save:
                test_mean_score = (
                    epoch_metrics.get("eval/success_rate") if self.enable_env_eval
                    else -epoch_metrics["val/loss"] if "val/loss" in epoch_metrics
                    else None
                )
                self._save_epoch_checkpoint(epoch, global_step, checkpoint_model, test_mean_score)
            else:
                # should_save_latest only: refresh the latest symlink to point
                # at the most recent existing checkpoint (no new save).
                existing = sorted(self.workspace.checkpoint_dir.glob("epoch=*.pt"))
                if existing:
                    self.workspace.save_latest(existing[-1])

    def on_epoch_start(self, epoch: int):
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        self.train_sampling_batch = None

    def train(self, resume_tag: str = "latest", resume_state=None):
        torch.set_float32_matmul_precision('high')

        if resume_state is not None:
            global_step, start_epoch = resume_state
            if start_epoch > 0:
                print(f"Resuming training from epoch {start_epoch}, step {global_step}")
        else:
            global_step, start_epoch = self.load_for_resume(resume_tag)
            if start_epoch > 0:
                print(f"Resuming training from epoch {start_epoch}, step {global_step}")

        self.model.to(self.device)
        if self.use_ema:
            self.ema_model.to(self.device)
            self.ema_model.eval()

        if self.use_compile:
            compile_models(self.model, self.ema_model)

        optimizer_to(self.optimizer, self.device)

        epoch_iter = range(start_epoch, self.num_epochs)
        if self.is_main_process:
            epoch_pbar = tqdm(epoch_iter, desc="Epoch", position=0, mininterval=1.0)
        else:
            epoch_pbar = epoch_iter

        for epoch in epoch_pbar:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            self.on_epoch_start(epoch)

            self.optimizer.zero_grad(set_to_none=True)

            last_batch = None
            for micro_step, batch in enumerate(self.train_loader):
                if self.train_sampling_batch is None:
                    self.train_sampling_batch = dict_apply(
                        batch, lambda x: x.to(self.device, non_blocking=True))
                last_batch = batch
                self.current_epoch = epoch

                is_boundary = (micro_step + 1) % self.gradient_accumulation_steps == 0

                # DDP: suppress gradient all-reduce for non-boundary micro-batches
                # so that gradients accumulate locally, then sync once on the boundary.
                if self.distributed and not is_boundary:
                    sync_ctx = self.model.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()

                with sync_ctx:
                    _, log_dict = self.train_one_step(batch, is_accumulation_boundary=is_boundary)

                if is_boundary:
                    global_step += 1
                    self.global_step = global_step

                    if self.is_main_process and (global_step % self.log_interval_steps) == 0:
                        step_metrics = {"train/lr": self.scheduler.get_last_lr()[0]}
                        for key, value in to_log_scalars(log_dict).items():
                            step_metrics[f"train/{key}"] = value

                        if hasattr(epoch_pbar, 'set_postfix'):
                            epoch_pbar.set_postfix(
                                global_step=global_step,
                                loss=step_metrics.get("train/loss", None),
                            )
                        self.workspace.log(step_metrics, step=global_step)

            self.model.eval()

            finish_error = None
            if self.is_main_process:
                try:
                    self.finish_epoch(epoch, global_step, last_batch=last_batch)
                except Exception as e:
                    if self.distributed:
                        traceback.print_exc()
                        finish_error = e
                    else:
                        raise

            if self.distributed:
                error_flag = torch.tensor(
                    [1 if finish_error is not None else 0],
                    dtype=torch.int, device=self.device,
                )
                dist.broadcast(error_flag, src=0)
                if error_flag.item():
                    raise RuntimeError(
                        "Training aborted due to error in finish_epoch on rank 0"
                    ) from finish_error


