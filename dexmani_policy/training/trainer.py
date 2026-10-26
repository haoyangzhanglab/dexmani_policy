from __future__ import annotations

import copy
import os
import signal
import time
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dexmani_policy.common.checkpoint import TrainingCheckpoint, policy_state_dict
from dexmani_policy.common.experiments import milestone_steps
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.runtime import close_resources, dict_apply, get_rng_state
from dexmani_policy.training.common.logging import accumulate_metrics, to_log_scalars
from dexmani_policy.training.resume import build_resume_contract


@dataclass
class TrainLoopConfig:
    total_train_steps: int
    log_interval_steps: int
    gradient_accumulation_steps: int


class Trainer:
    """Optimizer-step training; entry points own the component lifetime."""

    def __init__(
        self,
        components,
        train_loop_cfg: TrainLoopConfig,
        effective_config,
        max_grad_norm: float = 1.0,
        use_bfloat16: bool = False,
    ):
        self.device = components.device

        self.model = components.model
        self.ema_model = components.ema_model
        self.ema_updater = components.ema_updater

        self.optimizer = components.optimizer
        self.scheduler = components.scheduler

        self.train_loader = components.train_loader
        self.workspace = components.workspace

        self.total_train_steps = train_loop_cfg.total_train_steps
        self.log_interval_steps = train_loop_cfg.log_interval_steps
        self.max_grad_norm = max_grad_norm
        self.last_grad_norm: torch.Tensor | None = None
        self.last_clip_ratio: torch.Tensor | None = None

        self.use_bfloat16 = use_bfloat16

        self.gradient_accumulation_steps = train_loop_cfg.gradient_accumulation_steps
        self.resume_contract = build_resume_contract(self.train_loader)
        self.batches_per_epoch = len(self.train_loader)
        self.amp_device_type = torch.device(self.device).type

        self.effective_config = copy.deepcopy(
            OmegaConf.to_container(effective_config, resolve=True)
            if OmegaConf.is_config(effective_config)
            else effective_config
        )
        self.next_micro_step = 0
        self.current_epoch = 0
        self.global_step = 0

        self.stop_requested = False
        self.step_pbar = None
        self.last_checkpoint_step = None
        # Integer targets round up; colliding steps retain the largest percentage.
        self.milestone_steps = milestone_steps(self.total_train_steps)

    def find_nonfinite_gradients(self) -> list:
        return [
            name
            for name, param in self.model.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]

    def optimizer_step(self):
        grad_norm = None
        grad_nan_params = []

        if self.max_grad_norm > 0:
            if any("clip_group" in group for group in self.optimizer.param_groups):
                clip_groups = {}
                for group in self.optimizer.param_groups:
                    clip_groups.setdefault(group["clip_group"], []).extend(group["params"])
                norms = [
                    torch.nn.utils.clip_grad_norm_(
                        params, self.max_grad_norm, error_if_nonfinite=True
                    )
                    for params in clip_groups.values()
                ]
                # The largest independently clipped group determines the logged clip ratio.
                grad_norm = torch.stack(norms).max()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm,
                    error_if_nonfinite=True,
                )
        else:
            flags = [
                torch.isfinite(p.grad).all()
                for p in self.model.parameters()
                if p.grad is not None
            ]
            if flags and not bool(torch.stack(flags).all()):
                grad_nan_params = self.find_nonfinite_gradients()

        if grad_nan_params:
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite gradient at epoch={self.current_epoch}, step={self.global_step} "
                f"in {len(grad_nan_params)} parameter(s): {grad_nan_params[:5]}"
                f"{'...' if len(grad_nan_params) > 5 else ''}"
            )

        if grad_norm is not None:
            self.last_grad_norm = grad_norm.detach()
            self.last_clip_ratio = self.last_grad_norm / self.max_grad_norm
        else:
            self.last_grad_norm = None
            self.last_clip_ratio = None

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.ema_updater is not None:
            self.ema_updater.step(self.model)

    def train_microbatch(
        self,
        batch: dict[str, Any],
        *,
        is_accumulation_boundary: bool = True,
        loss_divisor: int = 1,
        log_metrics: bool = True,
    ):
        """Backpropagate one micro-batch, updating weights at accumulation boundaries."""
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        loss_kwargs = self.model.get_training_loss_kwargs(self.ema_model)
        with torch.amp.autocast(
            device_type=self.amp_device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16,
        ):
            raw_loss, log_dict = self.model(batch, log_metrics=log_metrics, **loss_kwargs)

        value = raw_loss.detach()
        if not torch.isfinite(value):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite loss at epoch={self.current_epoch}, step={self.global_step} "
                f"(raw_loss={value.item()})"
            )
        (raw_loss / loss_divisor).backward()

        if is_accumulation_boundary:
            self.optimizer_step()

        return log_dict

    def save_checkpoint(self, tag_suffix: str):
        global_step = self.global_step
        if self.last_checkpoint_step == global_step:
            return
        checkpoint = TrainingCheckpoint(
            epoch=self.current_epoch,
            global_step=global_step,
            next_micro_step=self.next_micro_step,
            model_state=policy_state_dict(self.model),
            ema_model_state=(
                policy_state_dict(self.ema_model) if self.ema_model is not None else None
            ),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            resume_contract=self.resume_contract,
            effective_config=self.effective_config,
            ema_updater_step=(
                self.ema_updater.optimization_step if self.ema_updater is not None else None
            ),
            rng_state=get_rng_state(self.device),
        )
        tag = f"epoch={self.current_epoch:04d}-step={global_step:08d}-{tag_suffix}"
        checkpoint_path = self.workspace.save_checkpoint(tag, checkpoint)
        self.workspace.save_latest(checkpoint_path)
        self.last_checkpoint_step = global_step

    def signal_handler(self, signum, frame):
        """Minimal signal handler: set flag on first signal, force-exit on second."""
        if self.stop_requested:
            signame = signal.Signals(signum).name
            print(f"\nSecond {signame} — forcing exit.", flush=True)
            os._exit(1)
        signame = signal.Signals(signum).name
        print(
            f"\n=== {signame} — finishing current step, then saving checkpoint... ===",
            flush=True,
        )
        self.stop_requested = True

    def check_milestone(self):
        pct = self.milestone_steps.get(self.global_step)
        if pct is not None:
            self.save_checkpoint(f"milestone={pct:02d}pct")

    def log_step(self, group_metrics, samples, elapsed, data_wait):
        epoch_samples = min(
            self.train_loader.sampler.num_samples,
            self.batches_per_epoch * self.train_loader.batch_size,
        )
        metrics = {
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/processed_windows": (
                self.current_epoch * epoch_samples
                + min(self.next_micro_step * self.train_loader.batch_size, epoch_samples)
            ),
        }
        if self.last_grad_norm is not None:
            metrics.update(
                {"train/grad_norm": self.last_grad_norm, "train/clip_ratio": self.last_clip_ratio}
            )
        metrics.update({f"train/{key}": value for key, value in group_metrics.items()})
        metrics = to_log_scalars(metrics)
        metrics.update({
            "train/step_time": elapsed,
            "train/data_wait_time": data_wait,
            "train/samples_per_sec": samples / elapsed,
            "train/samples_per_step": samples,
        })
        if self.step_pbar is not None:
            self.step_pbar.set_postfix(
                loss=f"{metrics['train/loss']:.5f}",
                step=f"{self.global_step:,}",
                lr=f"{metrics['train/lr']:.2e}",
                refresh=False,
            )
            self.step_pbar.update(self.global_step - self.step_pbar.n)
        self.workspace.log(metrics, step=self.global_step)

    def train(self, *, resume_state=None, max_updates=None):
        if max_updates is not None:
            positive_int(max_updates, "max_updates")
        torch.set_float32_matmul_precision("high")

        self.global_step, start_epoch, self.next_micro_step = (
            (0, 0, 0) if resume_state is None else resume_state
        )

        stop_step = (
            self.total_train_steps
            if max_updates is None
            else min(self.total_train_steps, self.global_step + max_updates)
        )
        if self.global_step >= self.total_train_steps:
            self.current_epoch = start_epoch
            return
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}, step {self.global_step}")

        if self.ema_model is not None:
            self.ema_model.eval()

        self.stop_requested = False
        prev_sigint = signal.signal(signal.SIGINT, self.signal_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, self.signal_handler)

        epoch = start_epoch
        self.current_epoch = epoch
        self.step_pbar = tqdm(
            initial=self.global_step,
            total=self.total_train_steps,
            desc="Steps",
            position=0,
            mininterval=5.0,
            miniters=1,
        )

        try:
            while self.global_step < stop_step:
                self.train_loader.sampler.set_epoch(epoch, self.next_micro_step)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                num_batches = self.batches_per_epoch
                group_metric_sums = {}
                group_metric_count = 0
                group_samples = 0
                group_data_wait = 0.0
                # Start before iterator creation, including the epoch's first data wait.
                group_start_time = time.perf_counter()
                data_wait_start = group_start_time
                for micro_step, batch in enumerate(self.train_loader, start=self.next_micro_step):
                    group_data_wait += time.perf_counter() - data_wait_start
                    self.current_epoch = epoch
                    group_samples += batch["action"].shape[0]

                    group_start = (
                        micro_step // self.gradient_accumulation_steps
                    ) * self.gradient_accumulation_steps
                    group_size = min(
                        self.gradient_accumulation_steps,
                        num_batches - group_start,
                    )
                    group_pos = micro_step - group_start
                    is_boundary = group_pos + 1 == group_size

                    record_group = (self.global_step + 1) % self.log_interval_steps == 0
                    log_dict = self.train_microbatch(
                        batch,
                        is_accumulation_boundary=is_boundary,
                        loss_divisor=group_size,
                        log_metrics=record_group,
                    )

                    if is_boundary:
                        group_elapsed = time.perf_counter() - group_start_time
                    if record_group:
                        accumulate_metrics(group_metric_sums, log_dict)
                        group_metric_count += 1

                    if is_boundary:
                        group_metrics = {
                            key: value / group_metric_count
                            for key, value in group_metric_sums.items()
                        }
                        group_metric_sums = {}
                        group_metric_count = 0
                        self.global_step += 1
                        self.next_micro_step = micro_step + 1
                        if self.next_micro_step == num_batches:
                            self.current_epoch = epoch + 1
                            self.next_micro_step = 0
                        if record_group:
                            self.log_step(
                                group_metrics, group_samples, group_elapsed, group_data_wait
                            )

                        self.check_milestone()
                        group_samples = 0
                        group_data_wait = 0.0
                        # Exclude this update's logging/checkpoint work from the next one.
                        group_start_time = time.perf_counter()

                    if is_boundary and (self.global_step >= stop_step or self.stop_requested):
                        break
                    data_wait_start = time.perf_counter()

                self.model.eval()
                epoch = self.current_epoch
                if self.stop_requested:
                    break

            if self.stop_requested and self.global_step > 0:
                print(
                    f"Training interrupted at step {self.global_step}/{self.total_train_steps}",
                    flush=True,
                )
                self.save_checkpoint("interrupt")
            elif self.global_step < self.total_train_steps:
                self.save_checkpoint("bounded")
                print(f"Bounded run stopped at step {self.global_step}/{self.total_train_steps}")

        finally:
            callbacks = [
                lambda: signal.signal(signal.SIGINT, prev_sigint),
                lambda: signal.signal(signal.SIGTERM, prev_sigterm),
            ]

            if self.step_pbar is not None:
                if self.step_pbar.n != self.global_step:
                    # Show the actual stop step without metrics from an older step.
                    self.step_pbar.set_postfix({}, refresh=False)
                    self.step_pbar.n = self.global_step
                callbacks.append(self.step_pbar.close)

            if torch.cuda.is_available():
                callbacks.append(torch.cuda.empty_cache)
            close_resources(*callbacks)
