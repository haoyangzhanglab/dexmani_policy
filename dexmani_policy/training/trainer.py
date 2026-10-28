import copy
import os
import signal
import time
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

from dexmani_policy.common.checkpoint import (
    Checkpoint, load_policy_state, policy_state_dict,
    prepare_restore_config, save_checkpoint,
)
from dexmani_policy.common.config import (
    positive_int, validate_observation_fields,
)
from dexmani_policy.common.experiments import milestone_steps
from dexmani_policy.common.io_util import atomic_path, close_resources
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state
from dexmani_policy.training.common.config import assemble_effective_config
from dexmani_policy.training.common.data import build_train_loader, close_data_loader, epoch_position
from dexmani_policy.training.common.logging import (
    ExperimentLogger, accumulate_metrics, print_param_count, print_training_recipe, to_log_scalars,
)
from dexmani_policy.training.common.optimization import build_optimizer, get_scheduler
from dexmani_policy.training.common.resume import archive_resume_milestones, validate_source_recipe


class PolicyTrainer:
    def __init__(self, cfg, dataset, normalizer, data_description, *, checkpoint=None):
        device = torch.device(cfg.training.device)
        model_cfg = cfg.agent if checkpoint is None else prepare_restore_config(cfg.agent)
        model = hydra.utils.instantiate(model_cfg)
        validate_observation_fields(model, cfg.dataset.sensor_modalities)
        if data_description["action_start_index"] != model.action_start_index:
            raise ValueError("Dataset/policy action_start_index mismatch")
        if data_description["control_action_dim"] != model.control_action_dim:
            raise ValueError("Dataset primary action width must match policy control output")
        model.set_normalization_spec(
            cfg.normalization, data_description["normalization_dims"]
        )
        if checkpoint is None:
            model.initialize_training(data_description)
            model.load_normalizer(normalizer)
        if checkpoint is not None and dict(cfg.policy_metadata) != model.inference_metadata:
            raise ValueError("Restored policy metadata differs from saved construction")

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

        # Freeze before EMA chooses which parameters need FP32 accumulation.
        obs_lr = cfg.optimizer.get("obs_lr")
        if (cfg.optimizer.lr if obs_lr is None else obs_lr) == 0:
            model.obs_encoder.requires_grad_(False)
        ema_model = None
        ema_updater = None
        if cfg.training.use_ema:
            # Copy before compile; resume restores raw and EMA independently below.
            ema_model = copy.deepcopy(model)
            ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

        if checkpoint is not None:
            load_policy_state(model, checkpoint.model_state)
            if ema_model is not None:
                if checkpoint.ema_model_state is None or checkpoint.ema_updater_step is None:
                    raise ValueError("Resume checkpoint is missing EMA state")
                load_policy_state(ema_model, checkpoint.ema_model_state)
                averaged = dict(ema_model.named_parameters())
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad and (
                        parameter.dtype != averaged[name].dtype
                        or not torch.equal(parameter, averaged[name])
                    ):
                        raise ValueError(f"Raw/EMA frozen parameter mismatch: {name}")
                for name, value in model.normalizer.named_buffers():
                    if not torch.equal(value, ema_model.normalizer.get_buffer(name)):
                        raise ValueError(f"Raw/EMA normalizer mismatch: {name}")
        model.to(device)
        if ema_model is not None:
            ema_model.to(device)

        self.device = device
        self.model, self.ema_model, self.ema_updater = model, ema_model, ema_updater
        self.train_loader = build_train_loader(cfg, dataset)
        self.optimizer = build_optimizer(model, **cfg.optimizer)
        self.scheduler = get_scheduler(
            self.optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps,
            cfg.training.loop.total_train_steps, cfg.training.get("lr_min_ratio", 0.1),
        )
        self.cfg = assemble_effective_config(cfg, data_description, model, checkpoint=checkpoint)
        self.total_train_steps = cfg.training.loop.total_train_steps
        self.log_interval_steps = cfg.training.loop.log_interval_steps
        self.gradient_accumulation_steps = cfg.training.loop.gradient_accumulation_steps
        self.max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
        self.use_bfloat16 = cfg.training.get("use_bfloat16", False)
        self.amp_device_type = device.type
        self.batches_per_epoch = self.train_loader.batch_sampler.batches_per_epoch
        self.current_epoch = self.global_step = 0

        if checkpoint is not None:
            validate_source_recipe(checkpoint.effective_config, self.cfg)
            groups, saved = self.optimizer.param_groups, checkpoint.optimizer_state["param_groups"]
            if len(groups) != len(saved) or any(
                a["param_names"] != b["param_names"] or a.get("clip_group") != b.get("clip_group")
                for a, b in zip(groups, saved)
            ):
                raise ValueError("Optimizer parameter or clipping groups changed")
            self.optimizer.load_state_dict(checkpoint.optimizer_state)
            self.scheduler.load_state_dict(checkpoint.scheduler_state)
            if ema_updater is not None:
                ema_updater.optimization_step = checkpoint.ema_updater_step
            self.global_step = checkpoint.global_step
            self.current_epoch, _ = epoch_position(
                self.global_step, self.batches_per_epoch, self.gradient_accumulation_steps
            )
            if self.global_step > self.total_train_steps:
                raise ValueError("Checkpoint global_step exceeds the training budget")

        self.clip_groups: dict[str | None, list[torch.nn.Parameter]] = {}
        for group in self.optimizer.param_groups:
            self.clip_groups.setdefault(group.get("clip_group"), []).extend(group["params"])
        self.last_grad_norm = self.last_clip_ratio = None
        self.stop_requested = False
        self.step_pbar = None
        self.last_checkpoint_step = None
        self.milestone_steps = milestone_steps(self.total_train_steps)

        print_param_count(model)
        print_training_recipe(
            self.cfg, batches_per_epoch=self.batches_per_epoch, samples_per_epoch=len(dataset)
        )
        self.logger = ExperimentLogger(**cfg.workspace)
        self.checkpoint_dir = self.logger.output_dir / "checkpoints"
        try:
            if checkpoint is not None:
                archive_resume_milestones(cfg.resume_from, self.logger.output_dir, checkpoint)
            self.logger.save_config(self.cfg)
            if checkpoint is not None:
                set_rng_state(checkpoint.rng_state, device=device)
        except BaseException:
            self.logger.close()
            raise

    def compile_models(self):
        if self.cfg.training.get("use_compile", False):
            mode = self.cfg.training.get("compile_mode", "reduce-overhead")
            for model in (self.model, self.ema_model):
                if model is not None:
                    model.compile_backbone(mode=mode)

    def close(self):
        close_resources(lambda: close_data_loader(self.train_loader), self.logger.close)

    def optimizer_step(self):
        if self.max_grad_norm > 0:
            norms = [
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm, error_if_nonfinite=True)
                for params in self.clip_groups.values()
            ]
            self.last_grad_norm = torch.stack(norms).max().detach()
            self.last_clip_ratio = self.last_grad_norm / self.max_grad_norm
        else:
            finite = [
                torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None
            ]
            if finite and not torch.stack(finite).all():
                raise RuntimeError(f"Non-finite gradient at step {self.global_step}")

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema_updater is not None:
            self.ema_updater.step(self.model)

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

        return log_dict

    def save_checkpoint(self, tag_suffix: str):
        global_step = self.global_step
        if self.last_checkpoint_step == global_step:
            return
        checkpoint = Checkpoint(
            epoch=self.current_epoch,
            global_step=global_step,
            model_state=policy_state_dict(self.model),
            ema_model_state=(
                policy_state_dict(self.ema_model) if self.ema_model is not None else None
            ),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            effective_config=cast(dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True)),
            ema_updater_step=(
                self.ema_updater.optimization_step if self.ema_updater is not None else None
            ),
            rng_state=get_rng_state(self.device),
        )
        tag = f"epoch={self.current_epoch:04d}-step={global_step:08d}-{tag_suffix}"
        checkpoint_path = save_checkpoint(self.checkpoint_dir / f"{tag}.pt", checkpoint)
        with atomic_path(self.checkpoint_dir / "latest.pt") as temporary:
            temporary.unlink()
            temporary.symlink_to(checkpoint_path.name)
        self.last_checkpoint_step = global_step

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

    def check_milestone(self):
        pct = self.milestone_steps.get(self.global_step)
        if pct is not None:
            self.save_checkpoint(f"milestone={pct:02d}pct")

    def log_step(self, group_metrics, samples, elapsed, data_wait):
        metrics = {
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/epoch": self.current_epoch,
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
        self.logger.log(metrics, step=self.global_step)

    def train(self, *, max_updates=None):
        if max_updates is not None:
            positive_int(max_updates, "max_updates")
        torch.set_float32_matmul_precision("high")

        stop_step = (
            self.total_train_steps
            if max_updates is None
            else min(self.total_train_steps, self.global_step + max_updates)
        )
        if self.global_step >= self.total_train_steps:
            return
        if self.global_step:
            print(f"Resuming at step {self.global_step}, logical epoch {self.current_epoch}")

        if self.ema_model is not None:
            self.ema_model.eval()

        self.stop_requested = False
        prev_sigint = signal.signal(signal.SIGINT, self.signal_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, self.signal_handler)

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
                epoch, start_batch = epoch_position(
                    self.global_step, self.batches_per_epoch, self.gradient_accumulation_steps
                )
                self.train_loader.batch_sampler.set_epoch(epoch, start_batch)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                num_batches = self.batches_per_epoch
                group_metric_sums: dict[str, torch.Tensor | float] = {}
                group_samples = 0
                group_data_wait = 0.0
                # Start before iterator creation, including the epoch's first data wait.
                group_start_time = time.perf_counter()
                data_wait_start = group_start_time
                for micro_step, batch in enumerate(self.train_loader, start=start_batch):
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
                        loss_divisor=group_size,
                        log_metrics=record_group,
                    )

                    if is_boundary:
                        self.optimizer_step()
                        group_elapsed = time.perf_counter() - group_start_time
                    if record_group:
                        accumulate_metrics(group_metric_sums, log_dict)

                    if is_boundary:
                        group_metrics = {
                            key: value / group_size
                            for key, value in group_metric_sums.items()
                        }
                        group_metric_sums = {}
                        self.global_step += 1
                        if micro_step + 1 == num_batches:
                            self.current_epoch = epoch + 1
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

            close_resources(*callbacks)
