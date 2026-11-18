import copy
import os
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, policy_state_dict, save_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.pytorch_util import dict_apply, set_seed
from dexmani_policy.training import initialization
from dexmani_policy.training.config import check_fresh_output, normalize_output_dir, validate_config
from dexmani_policy.training.data import (
    build_dataset, build_train_loader, close_data_loader, record_data_processing,
    record_prior_data, record_training_data, restore_data_records,
)
from dexmani_policy.training.logging import (
    ExperimentLogger, accumulate_metrics, print_param_count, print_training_recipe, to_log_scalars,
)
from dexmani_policy.training.optimization import (
    build_optimizer, build_scheduler, clip_gradients, normalize_gradients, restore_optimizer,
)
from dexmani_policy.training.runtime import training_signals


def build_trainer(cfg, *, checkpoint: Checkpoint | None = None) -> "PolicyTrainer":
    cfg = copy.deepcopy(cfg)
    normalize_output_dir(cfg)
    with open_dict(cfg):
        remote_source = os.environ.get("DEXMANI_POLICY_REMOTE_SOURCE")
        cfg.pop("remote_source", None)
        if remote_source is not None:
            cfg.remote_source = remote_source
    source = cfg.initialization.policy if checkpoint is None else None
    validate_config(cfg)
    model_checkpoint = checkpoint
    if source is not None:
        source = Path(source).expanduser().resolve()
        model_checkpoint = load_checkpoint(source)
        saved = model_checkpoint.effective_config
        if cfg.agent._target_ != saved["agent"]["_target_"] or cfg.action_key != saved["action_key"]:
            raise ValueError("Policy initialization requires the same policy and action semantics")
        with open_dict(cfg):
            cfg.normalization = saved["normalization"]
            cfg.normalization_dims = saved["normalization_dims"]
            cfg.initialization.policy = None
        initialization.prepare_warm_start(cfg.policy_name, cfg.agent, saved["agent"])
    if checkpoint is None:
        check_fresh_output(cfg.workspace.output_dir)
    if model_checkpoint is None and cfg.policy_name in {"dp", "sat", "dqrise", "dexlatent"}:
        initialization.resolve_rgb_architecture(cfg.agent, cfg.initialization.rgb)
    set_seed(cfg.training.seed + (checkpoint.global_step if checkpoint is not None else 0))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    dataset, data_description = build_dataset(cfg)
    store = dataset.episode_store
    data_dt = data_description["control_dt_s"]
    record_data_processing(cfg, dataset)
    if checkpoint is None:
        record_training_data(cfg, store, dataset.train_mask)
        policy_dt = data_dt
        if source is not None:
            record_prior_data(cfg, dataset, source, saved["training_data"], from_policy=True)
    else:
        saved = checkpoint.effective_config
        restore_data_records(cfg, store, dataset.train_mask, saved=saved)
        if cfg.policy_name == "ode":
            current = (cfg.data_processing or {}).get("action_ode")
            previous = (saved["data_processing"] or {}).get("action_ode")
            if (current or {}).get("semantics") != (previous or {}).get("semantics"):
                raise ValueError("Resume ODE action target semantics mismatch; use a fresh or warm-start run")
        policy_dt = saved["control_dt_s"]
        if not np.isclose(data_dt, policy_dt, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Resume control_dt_s mismatch: dataset dt={data_dt} s, "
                f"checkpoint control_dt_s={policy_dt} s"
            )
    with open_dict(cfg):
        cfg.control_dt_s = policy_dt
    model = initialization.build_policy(
        cfg, dataset, data_description["field_dims"], checkpoint=model_checkpoint, resuming=checkpoint is not None,
    )
    if checkpoint is None:
        with open_dict(cfg):
            cfg.pop("max_updates", None)
            cfg.normalization_dims = model.normalization_dims
            cfg.agent.update(model.effective_config)
        OmegaConf.resolve(cfg)
    return PolicyTrainer(cfg, model, dataset, checkpoint=checkpoint)


def run_trainer(trainer, *, max_updates=None):
    try:
        trainer.compile_models()
        trainer.train(max_updates=max_updates)
        return trainer.global_step
    finally:
        trainer.close()


def run_training(cfg, *, max_updates=None):
    return run_trainer(build_trainer(cfg), max_updates=max_updates)


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
            ema_model.load_weights(checkpoint.ema_model_state, device)
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
            restore_optimizer(self.optimizer, checkpoint.optimizer_state)
            self.global_step = checkpoint.global_step

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
        self.milestone_steps = {
            (pct * self.total_train_steps + 99) // 100 for pct in (20, 40, 60, 80, 100)
        }

        print_param_count(model)
        print_training_recipe(self.cfg, batches_per_epoch=self.batches_per_epoch)
        self.logger = ExperimentLogger(cfg, resuming=checkpoint is not None)
        self.checkpoint_dir = self.logger.output_dir / "checkpoints"

    def compile_models(self):
        if self.cfg.training.use_compile:
            models = [self.model]
            if self.model.requires_ema_for_loss:
                models.append(self.ema_model)
            for model in models:
                model.compile_backbone(mode=self.cfg.training.compile_mode)

    def close(self):
        self.logger.close()

    def optimizer_step(self, finite_loss: torch.Tensor) -> None:
        self.last_grad_norm = clip_gradients(
            self.trainable_parameters, finite_loss, self.max_grad_norm, step=self.global_step,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema_updater is not None:
            self.ema_updater.step(self.global_step)

    def train_microbatch(
        self,
        batch: dict[str, Any],
        *,
        loss_weight: int = 1,
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

        (raw_loss if loss_weight == 1 else raw_loss * loss_weight).backward()
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
            effective_config=cast(dict, OmegaConf.to_container(self.cfg, resolve=True)),
        )
        checkpoint_path = save_checkpoint(self.checkpoint_dir / f"step_{global_step:08d}.pt", checkpoint)
        with atomic_path(self.checkpoint_dir / "latest.pt") as temporary:
            temporary.unlink()
            temporary.symlink_to(checkpoint_path.name)
        print(f"Saved {checkpoint_path}")

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
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with training_signals() as stop, tqdm(
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
                        sample_count = 0
                        for _ in range(group_size):
                            batch = next(iterator)
                            count = batch["action"].shape[0]
                            sample_count += count
                            loss, log_dict = self.train_microbatch(
                                batch, loss_weight=count if group_size > 1 else 1,
                                log_metrics=record_group,
                            )
                            finite_loss.logical_and_(torch.isfinite(loss))
                            if record_group:
                                accumulate_metrics(group_metric_sums, log_dict, weight=count)

                        if group_size > 1:
                            normalize_gradients(self.trainable_parameters, sample_count)
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.optimizer_step(finite_loss)
                        self.global_step += 1
                        pbar.update(1)
                        if record_group:
                            group_metrics = {
                                key: value / sample_count
                                for key, value in group_metric_sums.items()
                            }
                            self.log_step(group_metrics, pbar, lr=lr, epoch=epoch)

                        should_stop = stop.requested or self.global_step >= stop_step
                        if self.global_step in self.milestone_steps or should_stop:
                            self.save_checkpoint()
                        if should_stop:
                            break
                    if should_stop:
                        break

            finally:
                close_data_loader(self.train_loader, iterator)
                self.model.eval()
