from __future__ import annotations

import contextlib
import copy
import os
import signal
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dexmani_policy.training.common.checkpoint import TrainCheckpoint, unwrap_state_dict
from dexmani_policy.training.common.logging import to_log_scalars
from dexmani_policy.training.workspace import Workspace
from dexmani_policy.utils.config import positive_int
from dexmani_policy.utils.runtime import dict_apply, get_rng_state


@dataclass
class TrainLoopConfig:
    total_train_steps: int
    log_interval_steps: int
    gradient_accumulation_steps: int


class Trainer:
    """Optimizer-step training; callers own model construction and workspace cleanup."""

    def __init__(
        self,
        device,
        model,
        ema_model,
        ema_updater,
        optimizer,
        scheduler,
        train_loader,
        workspace: Workspace | None,
        train_loop_cfg: TrainLoopConfig,
        resume_contract: dict,
        effective_config,
        max_grad_norm: float = 1.0,
        use_bfloat16: bool = False,
        is_main_process: bool = True,
        distributed: bool = False,
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
        self.last_grad_norm: torch.Tensor | None = None
        self.last_clip_ratio: torch.Tensor | None = None

        self.use_bfloat16 = use_bfloat16

        self.gradient_accumulation_steps = train_loop_cfg.gradient_accumulation_steps
        self.batches_per_epoch = resume_contract["batches_per_epoch"]
        self.amp_device_type = torch.device(device).type

        self.is_main_process = is_main_process
        self.distributed = distributed
        self.ddp_backward_initialized = False
        self.resume_contract = resume_contract
        self.effective_config = copy.deepcopy(
            OmegaConf.to_container(effective_config, resolve=True)
            if OmegaConf.is_config(effective_config)
            else effective_config
        )
        self.next_micro_step = 0
        self.current_epoch = 0
        self.global_step = 0

        self.interrupted = False
        self.stop_requested = False
        self.step_pbar = None
        self.last_checkpoint_step = None
        # Integer targets round up; colliding steps retain the largest percentage.
        self.milestone_steps = {
            (pct * self.total_train_steps + 99) // 100: pct for pct in (20, 40, 60, 80, 100)
        }

    @property
    def raw_model(self):
        """Unwrap DDP and compilation without changing the forward path."""
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def find_nonfinite_gradients(self) -> list:
        return [
            name
            for name, param in self.raw_model.named_parameters()
            if param.grad is not None and not torch.isfinite(self.gradient_values(param.grad)).all()
        ]

    @staticmethod
    def gradient_values(grad):
        if grad.layout == torch.sparse_coo:
            return grad._values()
        return grad if grad.layout == torch.strided else grad.values()

    def apply_gradient_step(self):
        grad_norm = None
        grad_nan_params = []

        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )
        else:
            flags = [
                torch.isfinite(self.gradient_values(p.grad)).all()
                for p in self.raw_model.parameters()
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
            self.ema_updater.step(self.raw_model)

    def train_one_step(
        self,
        batch: dict[str, Any],
        *,
        is_accumulation_boundary: bool = True,
        loss_divisor: int = 1,
    ):
        """Backpropagate one micro-batch, updating weights at accumulation boundaries."""
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        loss_kwargs = self.raw_model.get_training_loss_kwargs(self.ema_model)
        with torch.amp.autocast(
            device_type=self.amp_device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16,
        ):
            raw_loss, log_dict = self.model(batch, **loss_kwargs)

        self.validate_loss(raw_loss)
        (raw_loss / loss_divisor).backward()

        if is_accumulation_boundary:
            self.apply_gradient_step()

        return log_dict

    def validate_loss(self, loss):
        """Make every rank reject a nonfinite loss before entering backward."""
        value = loss.detach()
        if self.distributed:
            world_size = dist.get_world_size()
            failed_rank = torch.where(torch.isfinite(value), world_size, dist.get_rank())
            dist.all_reduce(failed_rank, op=dist.ReduceOp.MIN)
            rank = int(failed_rank.item())
            if rank == world_size:
                return
            dist.broadcast(value, src=rank)
            context = f"rank={rank}, "
        else:
            if torch.isfinite(value):
                return
            context = ""
        self.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            f"Non-finite loss at epoch={self.current_epoch}, step={self.global_step} "
            f"({context}raw_loss={value.item()})"
        )

    def save_checkpoint(self, global_step: int, tag_suffix: str):
        if self.last_checkpoint_step == global_step:
            return
        rng_states = [get_rng_state(self.device)]
        if self.distributed:
            local_rng = rng_states[0]
            rng_states = [None] * dist.get_world_size()
            dist.all_gather_object(rng_states, local_rng)
        if self.workspace is None or not self.is_main_process:
            self.last_checkpoint_step = global_step
            return
        checkpoint = TrainCheckpoint(
            epoch=self.current_epoch,
            global_step=global_step,
            next_micro_step=self.next_micro_step,
            model_state=unwrap_state_dict(self.raw_model.state_dict()),
            ema_model_state=(
                unwrap_state_dict(self.ema_model.state_dict())
                if self.ema_model is not None
                else None
            ),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            resume_contract=self.resume_contract,
            effective_config=self.effective_config,
            ema_updater_step=(
                self.ema_updater.optimization_step if self.ema_updater is not None else None
            ),
            rng_states=rng_states,
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

    def check_milestone(self, global_step: int):
        """Save the largest colliding percentage once, at its integer target step."""
        pct = self.milestone_steps.get(global_step)
        if pct is not None:
            self.save_checkpoint(global_step, f"milestone={pct:02d}pct")

    def step_performance_metrics(self, samples: int, elapsed: float, data_wait: float) -> dict:
        """Approximate update wall time, including data waits and excluding logging/saving."""
        if self.distributed:
            local = torch.tensor(
                [samples, elapsed, data_wait], dtype=torch.float64, device=self.device
            )
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            values = torch.stack(gathered).cpu()
            samples = values[:, 0].sum().item()
            elapsed = values[:, 1].max().item()
            data_wait = values[:, 2].max().item()
        return {
            "train/step_time": elapsed,
            "train/data_wait_time": data_wait,
            "train/samples_per_sec": samples / elapsed,
            "train/samples_per_step": samples,
        }

    def stop_requested_on_any_rank(self):
        if not self.distributed:
            return self.stop_requested
        stop = torch.tensor(int(self.stop_requested), device=self.device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        return bool(stop.item())

    def reduce_metrics(self, metrics):
        if not self.distributed:
            return to_log_scalars(metrics)
        keys = sorted(metrics)
        packed = torch.stack(
            [
                torch.as_tensor(metrics[key], dtype=torch.float64, device=self.device).reshape(())
                for key in keys
            ]
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed /= dist.get_world_size()
        return dict(zip(keys, packed.cpu().tolist()))

    def log_step(self, group_metrics, samples, elapsed, data_wait):
        epoch_samples = min(
            self.train_loader.sampler.num_samples,
            self.batches_per_epoch * self.train_loader.batch_size,
        )
        metrics = {
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/processed_windows": self.resume_contract["world_size"]
            * (
                self.current_epoch * epoch_samples
                + min(self.next_micro_step * self.train_loader.batch_size, epoch_samples)
            ),
        }
        if self.last_grad_norm is not None:
            metrics.update(
                {"train/grad_norm": self.last_grad_norm, "train/clip_ratio": self.last_clip_ratio}
            )
        metrics.update({f"train/{key}": value for key, value in group_metrics.items()})
        metrics = self.reduce_metrics(metrics)
        metrics.update(self.step_performance_metrics(samples, elapsed, data_wait))
        if self.is_main_process:
            if self.step_pbar is not None:
                self.step_pbar.set_postfix(
                    loss=f"{metrics['train/loss']:.5f}",
                    step=f"{self.global_step:,}",
                    lr=f"{metrics['train/lr']:.2e}",
                    refresh=False,
                )
                self.step_pbar.update(self.global_step - self.step_pbar.n)
            if self.workspace is not None:
                self.workspace.log(metrics, step=self.global_step)

    def train(self, *, resume_state=None, max_updates=None):
        if max_updates is not None:
            positive_int(max_updates, "max_updates")
        torch.set_float32_matmul_precision("high")

        global_step, start_epoch, self.next_micro_step = (
            (0, 0, 0) if resume_state is None else resume_state
        )

        self.global_step = global_step
        stop_step = (
            self.total_train_steps
            if max_updates is None
            else min(self.total_train_steps, global_step + max_updates)
        )
        if global_step >= self.total_train_steps:
            self.current_epoch = start_epoch
            return
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}, step {global_step}")

        if self.ema_model is not None:
            self.ema_model.eval()

        self.interrupted = False
        self.stop_requested = False
        prev_sigint = signal.signal(signal.SIGINT, self.signal_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, self.signal_handler)

        epoch = start_epoch
        self.current_epoch = epoch
        if self.is_main_process:
            self.step_pbar = tqdm(
                initial=global_step,
                total=self.total_train_steps,
                desc="Steps",
                position=0,
                mininterval=5.0,
                miniters=1,
            )

        try:
            while global_step < stop_step:
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

                    # PyTorch 2.4 static_graph needs one synchronized backward before no_sync.
                    if self.distributed and not is_boundary and self.ddp_backward_initialized:
                        sync_ctx = self.model.no_sync()
                    else:
                        sync_ctx = contextlib.nullcontext()

                    with sync_ctx:
                        log_dict = self.train_one_step(
                            batch,
                            is_accumulation_boundary=is_boundary,
                            loss_divisor=group_size,
                        )

                    if is_boundary:
                        group_elapsed = time.perf_counter() - group_start_time
                    self.ddp_backward_initialized = True
                    record_group = (global_step + 1) % self.log_interval_steps == 0
                    if record_group:
                        for key, value in log_dict.items():
                            if torch.is_tensor(value):
                                if value.numel() != 1:
                                    continue
                                value = value.detach().to(dtype=torch.float64)
                            elif not isinstance(value, (int, float)):
                                continue
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
                        self.interrupted = self.stop_requested_on_any_rank()
                        if record_group:
                            self.log_step(
                                group_metrics, group_samples, group_elapsed, group_data_wait
                            )

                        self.check_milestone(global_step)
                        group_samples = 0
                        group_data_wait = 0.0
                        # Exclude this update's logging/checkpoint work from the next one.
                        group_start_time = time.perf_counter()

                    if is_boundary and (global_step >= stop_step or self.interrupted):
                        break
                    data_wait_start = time.perf_counter()

                self.model.eval()
                epoch = self.current_epoch
                if self.interrupted:
                    break

            if self.interrupted and global_step > 0:
                print(
                    f"Training interrupted at step {global_step}/{self.total_train_steps}",
                    flush=True,
                )
                self.save_checkpoint(global_step, "interrupt")
            elif global_step < self.total_train_steps:
                self.save_checkpoint(global_step, "bounded")
                print(f"Bounded run stopped at step {global_step}/{self.total_train_steps}")

        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

            if self.step_pbar is not None:
                if self.step_pbar.n != global_step:
                    # Show the actual stop step without metrics from an older step.
                    self.step_pbar.set_postfix({}, refresh=False)
                    self.step_pbar.n = global_step
                self.step_pbar.close()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
