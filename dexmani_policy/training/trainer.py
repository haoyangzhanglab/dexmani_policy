import copy
import os
import signal
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

from dexmani_policy.common.checkpoint import (
    Checkpoint, policy_state_dict, save_checkpoint,
)
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import validate_normalizer_state
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.training.common.data import build_train_loader, close_data_loader
from dexmani_policy.training.common.logging import (
    ExperimentLogger, accumulate_metrics, print_param_count, print_training_recipe, to_log_scalars,
)
from dexmani_policy.training.common.optimization import build_optimizer, build_scheduler
from dexmani_policy.training.common import training_signal_handlers


class PolicyTrainer:
    def __init__(self, cfg, model, dataset, *, checkpoint=None):
        device = torch.device(cfg.training.device)
        if model.requires_ema_for_loss and not cfg.training.use_ema:
            raise ValueError(f"{type(model.action_decoder).__name__} requires training.use_ema=true")

        if cfg.training.use_ema:
            active_batchnorm = [
                name or "<root>" for name, module in model.named_modules()
                if isinstance(module, _BatchNorm) and (
                    any(p.requires_grad for p in module.parameters(recurse=False))
                    or (module.training and module.track_running_stats)
                )
            ]
            if active_batchnorm:
                raise ValueError(
                    f"EMA requires frozen BatchNorm or GroupNorm; active modules: {active_batchnorm}"
                )

        # Freeze before EMA selects the trainable parameters.
        obs_lr = cfg.optimizer.get("obs_lr")
        if (cfg.optimizer.lr if obs_lr is None else obs_lr) == 0:
            model.obs_encoder.requires_grad_(False)
        ema_model = None
        ema_updater = None
        if cfg.training.use_ema:
            # Raw weights are ready; restore EMA independently before compile.
            ema_model = copy.deepcopy(model)

        if checkpoint is not None and ema_model is not None:
            if checkpoint.ema_model_state is None:
                raise ValueError("Resume checkpoint is missing EMA state")
            ema_model.load_state_dict(checkpoint.ema_model_state, strict=True, assign=True)
            ema_model.to(device=device, dtype=torch.float32)
            validate_normalizer_state(ema_model.normalizer)
        if ema_model is not None:
            # Bind current Parameters only after assignment and device migration.
            ema_updater = hydra.utils.instantiate(cfg.ema, model=model, ema_model=ema_model)

        self.device = device
        self.model, self.ema_model, self.ema_updater = model, ema_model, ema_updater
        self.train_loader = build_train_loader(cfg, dataset)
        self.optimizer = build_optimizer(model, **cfg.optimizer)
        self.cfg = cfg
        self.total_train_steps = cfg.training.loop.total_train_steps
        self.log_interval_steps = cfg.training.loop.log_interval_steps
        self.gradient_accumulation_steps = cfg.training.loop.gradient_accumulation_steps
        self.max_grad_norm = cfg.training.max_grad_norm
        self.use_bfloat16 = cfg.training.use_bfloat16
        self.batches_per_epoch = len(self.train_loader)
        self.global_step = 0

        if checkpoint is not None:
            groups, saved = self.optimizer.param_groups, checkpoint.optimizer_state["param_groups"]
            if len(groups) != len(saved) or any(
                a["param_names"] != b["param_names"]
                for a, b in zip(groups, saved)
            ):
                raise ValueError("Optimizer parameter groups changed")
            self.optimizer.load_state_dict(checkpoint.optimizer_state)
            self.global_step = checkpoint.global_step
            if self.global_step > self.total_train_steps:
                raise ValueError("Checkpoint global_step exceeds the training budget")

        self.scheduler = build_scheduler(
            self.optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps,
            self.total_train_steps, state=checkpoint.scheduler_state if checkpoint is not None else None,
        )
        self.updates_per_epoch = (
            self.batches_per_epoch + self.gradient_accumulation_steps - 1
        ) // self.gradient_accumulation_steps
        self.trainable_parameters = [
            param for group in self.optimizer.param_groups for param in group["params"]
        ]
        self.last_grad_norm = None
        self.stop_requested = False
        self.milestone_steps = {
            (pct * self.total_train_steps + 99) // 100 for pct in (20, 40, 60, 80, 100)
        }

        print_param_count(model)
        print_training_recipe(self.cfg, batches_per_epoch=self.batches_per_epoch)
        self.logger = ExperimentLogger(**cfg.workspace, resuming=checkpoint is not None)
        self.checkpoint_dir = self.logger.output_dir / "checkpoints"
        try:
            self.logger.save_config(self.cfg)
        except BaseException:
            self.logger.close()
            raise

    def compile_models(self):
        if self.cfg.training.use_compile:
            mode = self.cfg.training.get("compile_mode", "reduce-overhead")
            models = [self.model]
            if self.model.requires_ema_for_loss:
                models.append(self.ema_model)
            for model in models:
                model.compile_backbone(mode=mode)

    def close(self):
        try:
            close_data_loader(self.train_loader)
        finally:
            self.logger.close()

    def optimizer_step(self):
        self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters, self.max_grad_norm or float("inf"), error_if_nonfinite=True
        ).detach()

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema_updater is not None:
            self.ema_updater.step(self.global_step)

    def train_microbatch(
        self,
        batch: dict[str, Any],
        *,
        loss_divisor: int = 1,
        log_metrics: bool = True,
    ):
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        loss_kwargs = self.model.get_training_loss_kwargs(self.ema_model)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16,
        ):
            raw_loss, log_dict = self.model(batch, log_metrics=log_metrics, **loss_kwargs)

        (raw_loss / loss_divisor).backward()
        return raw_loss.detach(), log_dict

    def save_checkpoint(self):
        global_step = self.global_step
        checkpoint = Checkpoint(
            global_step=global_step,
            model_state=policy_state_dict(self.model),
            ema_model_state=(
                policy_state_dict(self.ema_model) if self.ema_model is not None else None
            ),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            effective_config=cast(dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True)),
        )
        checkpoint_path = save_checkpoint(self.checkpoint_dir / f"step_{global_step:08d}.pt", checkpoint)
        with atomic_path(self.checkpoint_dir / "latest.pt") as temporary:
            temporary.unlink()
            temporary.symlink_to(checkpoint_path.name)
        print(f"Saved {checkpoint_path}")

    def signal_handler(self, signum, frame):
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

    def log_step(self, group_metrics, pbar, *, lr, epoch):
        metrics = {
            "train/lr": lr,
            "train/epoch": epoch,
        }
        if self.last_grad_norm is not None:
            metrics["train/grad_norm"] = self.last_grad_norm
            if self.max_grad_norm > 0:
                metrics["train/clip_ratio"] = self.last_grad_norm / self.max_grad_norm
        metrics.update({f"train/{key}": value for key, value in group_metrics.items()})
        metrics = to_log_scalars(metrics)
        pbar.set_postfix(
            loss=f"{metrics['train/loss']:.5f}",
            step=f"{self.global_step:,}",
            lr=f"{metrics['train/lr']:.2e}",
            refresh=False,
        )
        self.logger.log(metrics, step=self.global_step)

    def train(self, *, max_updates=None):
        if max_updates is not None:
            positive_int(max_updates, "max_updates")
        torch.set_float32_matmul_precision("highest")

        stop_step = (
            self.total_train_steps
            if max_updates is None
            else min(self.total_train_steps, self.global_step + max_updates)
        )
        if self.global_step >= self.total_train_steps:
            return
        if self.global_step:
            epoch = self.global_step // self.updates_per_epoch
            print(f"Resuming at step {self.global_step}, logical epoch {epoch}")

        if self.ema_model is not None:
            self.ema_model.eval()

        initial_step = self.global_step
        self.stop_requested = False
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with training_signal_handlers(self.signal_handler), tqdm(
            initial=self.global_step,
            total=self.total_train_steps,
            desc="Steps",
            position=0,
            mininterval=5.0,
            miniters=1,
        ) as pbar:

            iterator = None
            try:
                while self.global_step < stop_step:
                    num_batches = self.batches_per_epoch
                    iterator = iter(self.train_loader)
                    for group_start in range(
                        0, num_batches, self.gradient_accumulation_steps
                    ):
                        epoch = self.global_step // self.updates_per_epoch
                        group_size = min(self.gradient_accumulation_steps, num_batches - group_start)
                        record_group = (
                            self.global_step == initial_step
                            or (self.global_step + 1) % self.log_interval_steps == 0
                            or self.global_step + 1 == stop_step
                        )
                        finite_loss = torch.ones((), dtype=torch.bool, device=self.device)
                        group_metric_sums: dict[str, torch.Tensor | float] = {}
                        for _ in range(group_size):
                            batch = next(iterator)
                            loss, log_dict = self.train_microbatch(
                                batch, loss_divisor=group_size, log_metrics=record_group,
                            )
                            finite_loss.logical_and_(torch.isfinite(loss))
                            if record_group:
                                accumulate_metrics(group_metric_sums, log_dict)

                        if not finite_loss:
                            raise RuntimeError(f"Non-finite loss at step {self.global_step}")
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.optimizer_step()
                        self.global_step += 1
                        pbar.update(1)
                        if record_group:
                            group_metrics = {
                                key: value / group_size
                                for key, value in group_metric_sums.items()
                            }
                            self.log_step(group_metrics, pbar, lr=lr, epoch=epoch)

                        if self.global_step in self.milestone_steps or self.global_step >= stop_step or self.stop_requested:
                            self.save_checkpoint()
                        if self.global_step >= stop_step or self.stop_requested:
                            break
                    if self.stop_requested:
                        break

            finally:
                try:
                    self.model.eval()
                finally:
                    if iterator is not None:
                        close_data_loader(self.train_loader, iterator)
