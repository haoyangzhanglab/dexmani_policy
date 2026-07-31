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
from dexmani_policy.training.workspace import TrainWorkspace
from dexmani_policy.common.checkpoint_io import TrainCheckpoint, build_train_params

@dataclass
class TrainLoopConfig:
    total_train_steps: int = 80000
    log_interval_steps: int = 50
    gradient_accumulation_steps: int = 1
    max_val_steps: int | None = None  # kept for utility validate() method

MILESTONE_RATIOS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

class Trainer:
    """Step-driven training loop with milestone checkpointing.

    Trains for exactly ``total_train_steps`` optimizer steps (not epochs).
    No online validation or simulation evaluation — just training + milestone
    checkpoint saves at 20/40/60/80/100% progress.

    - **Training**: ``train_one_step()`` with mixed precision (bfloat16 AMP),
      gradient clipping, and two-layer NaN protection (loss NaN, grad NaN).
    - **Checkpointing**: Milestone saves (5 total) at progress thresholds;
      ``latest.pt`` symlink tracks the most recent milestone for resume.
    - **EMA**: Exponential moving average of model weights, updated each step.

    Supports single-GPU and DDP (via ``distributed=True``). In DDP, only rank
    0 performs logging and checkpointing.
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
        val_loader = None,
        env_runner = None,
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

        self.total_train_steps = train_loop_cfg.total_train_steps
        self.log_interval_steps = train_loop_cfg.log_interval_steps
        self.max_val_steps = train_loop_cfg.max_val_steps
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

        # Layer 2 NaN protection: detect gradient NaN/Inf before optimizer.step().
        # Loss NaN (layer 1) is caught in train_one_step(), but a gradient NaN
        # could slip through clip_grad_norm_ and silently corrupt optimizer state.
        # This fills the documented gap — see CLAUDE.md "NaN 三层防护".
        grad_nan_params = []
        for name, param in self.raw_model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_nan_params.append(name)
        if grad_nan_params:
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite gradient at epoch={self.current_epoch}, step={self.global_step} "
                f"in {len(grad_nan_params)} parameter(s): {grad_nan_params[:5]}"
                f"{'...' if len(grad_nan_params) > 5 else ''}"
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
                "model": fix_state_dict(self.raw_model.state_dict(), is_current_ddp=False),
                "ema_model": fix_state_dict(self.ema_model.state_dict(), is_current_ddp=False) if self.use_ema else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            "_format": "simple.v1",
            "_saved_at": time.time(),
        }
        # Atomic write pattern: save to .tmp then os.replace() so a crash
        # mid-save never produces a corrupted .pt file.
        tmp_path = ckpt_dir / (filename + ".tmp")
        final_path = ckpt_dir / filename
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)

        # Keep only the last 5 NaN debug checkpoints to avoid unbounded disk usage.
        nan_ckpts = sorted(ckpt_dir.glob("nan_debug_epoch=*.pt"))
        for p in nan_ckpts[:-5]:
            try:
                p.unlink()
            except OSError:
                pass

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

            if self.max_val_steps is not None and count >= self.max_val_steps:
                break

        if count == 0:
            return None

        if self.distributed:
            stats = torch.tensor(
                [loss_sum.item(), flow_sum.item(), cons_sum.item(), float(count)],
                device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_count = stats[3].item()
            result = {"loss": (stats[0] / global_count).item()}
            if has_components:
                result["loss_flow"] = (stats[1] / global_count).item()
                result["loss_consistency"] = (stats[2] / global_count).item()
            return result

        result = {"loss": (loss_sum / count).item()}
        if has_components:
            result["loss_flow"] = (flow_sum / count).item()
            result["loss_consistency"] = (cons_sum / count).item()
        return result

    @torch.no_grad()
    def evaluate(self, agent) -> Dict[str, Any]:
        try:
            result = self.env_runner.run(agent)
        except torch.cuda.OutOfMemoryError:
            raise  # OOM is fatal — training cannot continue, do not swallow
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARNING] Evaluation failed at epoch={self.current_epoch}, "
                  f"step={self.global_step}: {type(e).__name__}: {e}.")
            return {"eval/error": str(e)}
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

    def _init_milestone_state(self) -> set[float]:
        """Derive passed milestones from ``self.global_step`` — the single source of truth.

        A milestone is considered passed if its target step has been reached.
        This is more robust than filesystem scanning: manual file deletions won't
        cause re-saving at incorrect steps, and resumed training at exactly the
        final step correctly skips all milestones.
        """
        return {ratio for ratio in MILESTONE_RATIOS
                if self.global_step >= int(self.total_train_steps * ratio)}

    def _save_milestone_checkpoint(self, epoch: int, global_step: int, ratio: float):
        """Save a milestone checkpoint and point ``latest.pt`` at it.

        No score, no TopK tracking — we only care about progress milestones.
        """
        checkpoint_model = self.raw_model
        checkpoint = TrainCheckpoint(
            epoch=epoch,
            global_step=global_step,
            model_state=fix_state_dict(checkpoint_model.state_dict(), is_current_ddp=False),
            ema_model_state=fix_state_dict(self.ema_model.state_dict(), is_current_ddp=False) if self.use_ema else None,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            monitor={},
            train_params=build_train_params(self.raw_model, self.num_training_steps),
        )
        pct = int(ratio * 100)
        tag = f"epoch={epoch:04d}-step={global_step:08d}-milestone={pct:02d}pct"
        checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
        self.workspace.save_latest(checkpoint_path)

    def _check_milestone(self, epoch: int, global_step: int):
        """Check and save the first un-passed milestone whose threshold is met.

        Called after each accumulation-boundary step.  Because
        ``MILESTONE_RATIOS`` are spaced 20 percentage points apart and
        ``total_train_steps`` is typically much larger, at most one milestone
        is crossed per step under normal operation.
        """
        if self.workspace is None or not self.is_main_process:
            return
        for ratio in MILESTONE_RATIOS:
            if ratio in self._passed_milestones:
                continue
            if global_step / self.total_train_steps >= ratio:
                self._save_milestone_checkpoint(epoch, global_step, ratio)
                self._passed_milestones.add(ratio)
                break

    def on_epoch_start(self, epoch: int):
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)

    def train(self, resume_tag: str = "latest", resume_state=None):
        torch.set_float32_matmul_precision('high')

        if resume_state is not None:
            global_step, start_epoch = resume_state
        else:
            global_step, start_epoch = self.load_for_resume(resume_tag)

        self.global_step = global_step
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}, step {global_step}")

        self.model.to(self.device)
        if self.use_ema:
            self.ema_model.to(self.device)
            self.ema_model.eval()

        if self.use_compile:
            compile_models(self.model, self.ema_model)

        optimizer_to(self.optimizer, self.device)

        # Initialize milestone tracking AFTER global_step is established.
        self._passed_milestones = self._init_milestone_state()

        epoch = start_epoch
        if self.is_main_process:
            step_pbar = tqdm(
                initial=global_step, total=self.total_train_steps,
                desc="Steps", position=0, mininterval=1.0,
            )

        while global_step < self.total_train_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            self.on_epoch_start(epoch)

            self.optimizer.zero_grad(set_to_none=True)

            for micro_step, batch in enumerate(self.train_loader):
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

                        if hasattr(step_pbar, 'set_postfix'):
                            step_pbar.set_postfix(
                                step=global_step,
                                loss=step_metrics.get("train/loss", None),
                            )
                        if self.workspace is not None:
                            self.workspace.log(step_metrics, step=global_step)

                    # Check for milestone checkpoint.
                    self._check_milestone(epoch, global_step)

                if global_step >= self.total_train_steps:
                    break

            self.model.eval()
            epoch += 1

        if self.is_main_process and hasattr(step_pbar, 'close'):
            step_pbar.close()
