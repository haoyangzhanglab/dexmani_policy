from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dexmani_policy.common.checkpoint_io import (
    TrainCheckpoint,
)
from dexmani_policy.common.pytorch_util import (
    compile_models,
    dict_apply,
    fix_state_dict,
    get_rng_state,
    optimizer_to,
    to_log_scalars,
)
from dexmani_policy.training.build_utils import validate_gradient_accumulation
from dexmani_policy.training.workspace import TrainWorkspace
from dexmani_policy.training.resume import restore_training_state


@dataclass
class TrainLoopConfig:
    total_train_steps: int = 80000
    log_interval_steps: int = 100
    gradient_accumulation_steps: int = 1


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
        num_training_steps: int,
        resume_contract: dict,
        max_grad_norm: float = 1.0,
        fast_grad_finite_check: bool = False,
        use_bfloat16: bool = False,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        is_main_process: bool = True,
        distributed: bool = False,
        train_sampler=None,
    ):
        self.device = device

        self.model = model
        self.ema_model = ema_model
        self.ema_updater = ema_updater

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.workspace = workspace

        self.total_train_steps = train_loop_cfg.total_train_steps
        self.log_interval_steps = train_loop_cfg.log_interval_steps
        self.max_grad_norm = max_grad_norm
        self.fast_grad_finite_check = fast_grad_finite_check
        self._last_grad_norm: float | None = None
        self._last_clip_ratio: float | None = None

        self.use_ema = self.ema_model is not None
        self.use_ema_teacher_for_consistency = (
            use_ema_teacher_for_consistency and self.use_ema
        )

        self.use_bfloat16 = use_bfloat16
        self.use_compile = use_compile
        self.compile_mode = compile_mode

        self.gradient_accumulation_steps = train_loop_cfg.gradient_accumulation_steps
        validate_gradient_accumulation(
            len(self.train_loader), self.gradient_accumulation_steps
        )
        # Pre-compute AMP device_type string to avoid repeated str.split on every step
        self.amp_device_type = str(self.device).split(":")[0]

        self.is_main_process = is_main_process
        self.distributed = distributed
        self._ddp_backward_initialized = False
        self.train_sampler = train_sampler if train_sampler is not None else train_loader.sampler
        self.resume_contract = resume_contract
        self.next_micro_step = 0
        self.current_epoch = 0
        self.global_step = 0
        self.num_training_steps = num_training_steps

        self._interrupted = False
        self._stop_requested = False
        self._step_pbar = None

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
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _find_nonfinite_gradients(self) -> list:
        """Return the names of parameters whose ``.grad`` is non-finite (or None)."""
        return [
            name
            for name, param in self.raw_model.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]

    def apply_gradient_step(self):
        grad_norm = None
        grad_nan_params = []

        # Fast path fuses the norm computation into clip_grad_norm_ via
        # error_if_nonfinite=True, which raises before touching any parameter when
        # a gradient is non-finite. On the happy path this skips the per-parameter
        # torch.isfinite loop + GPU→CPU sync below. Only meaningful when we clip.
        use_fast = self.fast_grad_finite_check and self.max_grad_norm > 0

        if use_fast:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(),
                    max_norm=self.max_grad_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                # error_if_nonfinite raises RuntimeError for non-finite grads,
                # but a bare except also swallows unrelated errors (device
                # mismatch, internal clip failure).  Only treat it as a
                # non-finite-gradient case when non-finite params are confirmed;
                # otherwise re-raise the original error.
                grad_nan_params = self._find_nonfinite_gradients()
                if not grad_nan_params:
                    raise
        else:
            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), max_norm=self.max_grad_norm
                )

            # Layer 2 NaN protection: detect gradient NaN/Inf before optimizer.step().
            # Loss NaN (layer 1) is caught in train_one_step(), but a gradient NaN
            # could slip through clip_grad_norm_ and silently corrupt optimizer state.
            # This fills the documented gap — see CLAUDE.md "NaN 两层防护".
            grad_nan_params = self._find_nonfinite_gradients()

        if grad_nan_params:
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite gradient at epoch={self.current_epoch}, step={self.global_step} "
                f"in {len(grad_nan_params)} parameter(s): {grad_nan_params[:5]}"
                f"{'...' if len(grad_nan_params) > 5 else ''}"
            )

        # Record grad-norm diagnostics for logging (§8.8): total norm and how far
        # it exceeded the clip threshold (clip_ratio ≥ 1 means clipping engaged).
        if grad_norm is not None:
            self._last_grad_norm = float(grad_norm)
            self._last_clip_ratio = float(grad_norm) / float(self.max_grad_norm)
        else:
            self._last_grad_norm = None
            self._last_clip_ratio = None

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_ema and self.ema_updater is not None:
            self.ema_updater.step(self.raw_model)

    def load_for_resume(self, tag_or_path: str):
        """Restore the shared v3 state before compilation."""
        checkpoint = self.workspace.load_checkpoint(tag_or_path)
        return restore_training_state(
            checkpoint, resume_contract=self.resume_contract, model=self.raw_model,
            ema_model=self.ema_model, ema_updater=self.ema_updater,
            optimizer=self.optimizer, scheduler=self.scheduler, device=self.device,
            rank=dist.get_rank() if self.distributed else 0,
        )

    def _save_nan_debug(self, raw_loss, nan_rank=None):
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
                "nan_rank": nan_rank,
            },
            "weights": {
                "model": fix_state_dict(
                    self.raw_model.state_dict(), is_current_ddp=False
                ),
                "ema_model": (
                    fix_state_dict(self.ema_model.state_dict(), is_current_ddp=False)
                    if self.use_ema
                    else None
                ),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            "_format": "dexmani.nan-debug.v1",
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

    def train_one_step(
        self,
        batch: Dict[str, Any],
        *,
        is_accumulation_boundary: bool = True,
        loss_divisor: int = 1,
    ):
        """Forward + backward on one micro-batch.

        The epoch loop supplies the logical group's ``loss_divisor`` and defers
        ``optimizer.step()`` / ``scheduler.step()`` / EMA until the accumulation
        boundary (``is_accumulation_boundary=True``).

        Parameters:
            batch: Data dict from the DataLoader.
            is_accumulation_boundary: If ``True``, apply gradient step after
                backward.  Set to ``False`` for intermediate micro-batches
                when accumulating gradients.
            loss_divisor: Number of micro-batches in the logical accumulation
                group.
        """
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        loss_kwargs = (
            {"ema_backbone": self.ema_model.action_decoder.model}
            if self.use_ema_teacher_for_consistency
            else {}
        )
        with torch.amp.autocast(
            device_type=self.amp_device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16,
        ):
            raw_loss, log_dict = self.model(batch, **loss_kwargs)

        if self.distributed:
            # Gather every rank's detached loss so the debug checkpoint records
            # the rank that actually produced the non-finite loss (not rank 0's
            # own, possibly finite, micro-batch).
            loss_tensor = raw_loss.detach().reshape(1)
            gathered = [
                torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered, loss_tensor)
            gathered_losses = torch.cat(gathered)
            finite_mask = torch.isfinite(gathered_losses)
            is_nan = not bool(finite_mask.all().item())
            nan_rank = int((~finite_mask).nonzero()[0].item()) if is_nan else None
        else:
            gathered_losses = None
            is_nan = not torch.isfinite(raw_loss)
            nan_rank = None

        if is_nan:
            nan_loss = (
                gathered_losses[nan_rank].item()
                if gathered_losses is not None and nan_rank is not None
                else float(raw_loss.detach())
            )
            debug_path = self._save_nan_debug(nan_loss, nan_rank=nan_rank)
            self.optimizer.zero_grad(set_to_none=True)
            rank_str = f"rank={nan_rank}, " if nan_rank is not None else ""
            raise RuntimeError(
                f"Non-finite loss at epoch={self.current_epoch}, step={self.global_step} "
                f"({rank_str}raw_loss={nan_loss}). Debug checkpoint saved to {debug_path}"
            )

        # Scale loss so that the *sum* of micro-batch gradients equals the
        # gradient of the full batch (loss averaged across micro-batches).
        (raw_loss / loss_divisor).backward()

        if is_accumulation_boundary:
            self.apply_gradient_step()

        return batch, log_dict

    def _init_milestone_state(self) -> set[float]:
        """Derive passed milestones from ``self.global_step`` — the single source of truth.

        A milestone is considered passed if its target step has been reached.
        This is more robust than filesystem scanning: manual file deletions won't
        cause re-saving at incorrect steps, and resumed training at exactly the
        final step correctly skips all milestones.
        """
        return {
            ratio
            for ratio in MILESTONE_RATIOS
            if self.global_step / self.total_train_steps >= ratio
        }

    def _save_checkpoint(self, epoch: int, global_step: int, tag_suffix: str):
        """Save a checkpoint with the given tag suffix and point ``latest.pt`` at it."""
        rng_states = [get_rng_state()]
        if self.distributed:
            local_rng = rng_states[0]
            rng_states = [None] * dist.get_world_size()
            dist.all_gather_object(rng_states, local_rng)
        if self.workspace is None or not self.is_main_process:
            return
        checkpoint = TrainCheckpoint(
            epoch=self.current_epoch,
            global_step=global_step,
            next_micro_step=self.next_micro_step,
            model_state=fix_state_dict(
                self.raw_model.state_dict(), is_current_ddp=False
            ),
            ema_model_state=(
                fix_state_dict(self.ema_model.state_dict(), is_current_ddp=False)
                if self.use_ema
                else None
            ),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            monitor={},
            resume_contract=self.resume_contract,
            # Persist the full training state machine: EMA decay warmup counter
            # and the process RNG stream, so resume reproduces the same schedule.
            ema_updater_step=(
                self.ema_updater.optimization_step
                if self.use_ema and self.ema_updater is not None
                else None
            ),
            ema_decay=(
                self.ema_updater.decay
                if self.use_ema and self.ema_updater is not None
                else None
            ),
            rng_states=rng_states,
        )
        tag = f"epoch={self.current_epoch:04d}-step={global_step:08d}-{tag_suffix}"
        checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
        self.workspace.save_latest(checkpoint_path)

    def _save_milestone_checkpoint(self, epoch: int, global_step: int, ratio: float):
        """Save a milestone checkpoint and point ``latest.pt`` at it.

        No score, no TopK tracking — we only care about progress milestones.
        """
        pct = int(ratio * 100)
        self._save_checkpoint(epoch, global_step, f"milestone={pct:02d}pct")

    def _save_interrupt_checkpoint(self, epoch: int, global_step: int):
        """Save a checkpoint on signal-triggered interruption."""
        print(f"\nSaving interrupt checkpoint at step {global_step}...", flush=True)
        self._save_checkpoint(epoch, global_step, "interrupt")

    def _signal_handler(self, signum, frame):
        """Minimal signal handler: set flag on first signal, force-exit on second."""
        if self._stop_requested:
            signame = signal.Signals(signum).name
            print(f"\nSecond {signame} — forcing exit.", flush=True)
            os._exit(1)
        signame = signal.Signals(signum).name
        print(
            f"\n=== {signame} — finishing current step, then saving checkpoint... ===",
            flush=True,
        )
        self._stop_requested = True

    def _check_milestone(self, epoch: int, global_step: int):
        """Check and save the first un-passed milestone whose threshold is met.

        Called after each accumulation-boundary step.  Because
        ``MILESTONE_RATIOS`` are spaced 20 percentage points apart and
        ``total_train_steps`` is typically much larger, at most one milestone
        is crossed per step under normal operation.
        """
        for ratio in MILESTONE_RATIOS:
            if ratio in self._passed_milestones:
                continue
            if global_step / self.total_train_steps >= ratio:
                self._save_milestone_checkpoint(epoch, global_step, ratio)
                self._passed_milestones.add(ratio)
                break

    def on_epoch_start(self, epoch: int):
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch)
        if hasattr(self.raw_model, "set_epoch"):
            self.raw_model.set_epoch(epoch)

    def train(self, resume_tag: str | None = None, resume_state=None):
        torch.set_float32_matmul_precision("high")

        if resume_state is not None:
            global_step, start_epoch, self.next_micro_step = resume_state
        elif resume_tag is None:
            global_step, start_epoch = 0, 0
        else:
            global_step, start_epoch, self.next_micro_step = self.load_for_resume(resume_tag)

        self.global_step = global_step
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}, step {global_step}")

        self.model.to(self.device)
        if self.use_ema:
            self.ema_model.to(self.device)
            self.ema_model.eval()

        if self.use_compile:
            compile_models(self.model, self.ema_model, mode=self.compile_mode)

        optimizer_to(self.optimizer, self.device)

        # Initialize milestone tracking AFTER global_step is established.
        self._passed_milestones = self._init_milestone_state()

        self._interrupted = False
        self._stop_requested = False
        prev_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        epoch = start_epoch
        self.current_epoch = epoch
        if self.is_main_process:
            self._step_pbar = tqdm(
                initial=global_step,
                total=self.total_train_steps,
                desc="Steps",
                position=0,
                mininterval=1.0,
            )

        try:
            while global_step < self.total_train_steps:
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch, self.next_micro_step)

                self.model.train()
                self.on_epoch_start(epoch)

                self.optimizer.zero_grad(set_to_none=True)

                num_batches = self.resume_contract["batches_per_epoch"]
                group_metric_sums = {}
                group_metric_count = 0
                for micro_step, batch in enumerate(self.train_loader, start=self.next_micro_step):
                    self.current_epoch = epoch

                    group_start = (
                        micro_step // self.gradient_accumulation_steps
                    ) * self.gradient_accumulation_steps
                    group_size = min(
                        self.gradient_accumulation_steps,
                        num_batches - group_start,
                    )
                    group_pos = micro_step - group_start
                    is_boundary = group_pos + 1 == group_size

                    # DDP: suppress gradient all-reduce for non-boundary micro-batches
                    # so that gradients accumulate locally, then sync once on the boundary.
                    # PyTorch 2.4 static_graph initializes its reducer on the
                    # first synchronized backward. A first-ever no_sync()
                    # backward trips expect_autograd_hooks_ at the boundary.
                    # Averaging the first micro-gradient early is linear and
                    # preserves the accumulated global-mean gradient.
                    if self.distributed and not is_boundary and self._ddp_backward_initialized:
                        sync_ctx = self.model.no_sync()
                    else:
                        sync_ctx = contextlib.nullcontext()

                    with sync_ctx:
                        _, log_dict = self.train_one_step(
                            batch,
                            is_accumulation_boundary=is_boundary,
                            loss_divisor=group_size,
                        )

                    self._ddp_backward_initialized = True
                    for key, value in to_log_scalars(log_dict).items():
                        group_metric_sums[key] = group_metric_sums.get(key, 0.0) + value
                    group_metric_count += 1

                    if is_boundary:
                        group_metrics = {
                            key: value / group_metric_count
                            for key, value in group_metric_sums.items()
                        }
                        group_metric_sums = {}
                        group_metric_count = 0
                        global_step += 1
                        self.global_step = global_step
                        self.next_micro_step = micro_step + 1
                        if self.next_micro_step == num_batches:
                            self.current_epoch = epoch + 1
                            self.next_micro_step = 0
                        if self.distributed:
                            stop = torch.tensor(int(self._stop_requested), device=self.device)
                            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                            self._interrupted = bool(stop.item())
                        else:
                            self._interrupted = self._stop_requested

                        if (global_step % self.log_interval_steps) == 0:
                            step_metrics = {"train/lr": self.scheduler.get_last_lr()[0]}
                            if self._last_grad_norm is not None:
                                step_metrics["train/grad_norm"] = self._last_grad_norm
                                step_metrics["train/clip_ratio"] = self._last_clip_ratio
                            for key, value in group_metrics.items():
                                step_metrics[f"train/{key}"] = value

                            if self.distributed:
                                keys = sorted(step_metrics)
                                packed = torch.tensor(
                                    [step_metrics[key] for key in keys],
                                    dtype=torch.float64, device=self.device,
                                )
                                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                                packed /= dist.get_world_size()
                                step_metrics = dict(zip(keys, packed.cpu().tolist()))

                            if self.is_main_process and self._step_pbar is not None:
                                self._step_pbar.update(self.log_interval_steps)
                                if hasattr(self._step_pbar, "set_postfix"):
                                    self._step_pbar.set_postfix(
                                        loss=f"{step_metrics['train/loss']:.5f}",
                                        step=f"{global_step:,}",
                                        lr=f"{step_metrics['train/lr']:.2e}",
                                    )
                            if self.is_main_process and self.workspace is not None:
                                self.workspace.log(step_metrics, step=global_step)

                        # Check for milestone checkpoint.
                        self._check_milestone(epoch, global_step)

                    if is_boundary and (global_step >= self.total_train_steps or self._interrupted):
                        break

                self.model.eval()
                epoch = self.current_epoch
                if self._interrupted:
                    break

            if self._interrupted and global_step > 0:
                print(
                    f"Training interrupted at step {global_step}/{self.total_train_steps}",
                    flush=True,
                )
                try:
                    self._save_interrupt_checkpoint(epoch, global_step)
                except Exception as e:
                    print(f"WARNING: interrupt checkpoint failed: {e}", flush=True)

        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

            if self._step_pbar is not None:
                try:
                    self._step_pbar.close()
                except Exception:
                    pass

            if self.workspace is not None and self.is_main_process:
                try:
                    self.workspace.close()
                except Exception:
                    pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
