import copy
import os
from functools import partial
from itertools import islice
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, policy_state_dict, save_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path, cleanup_on_exit
from dexmani_policy.common.pytorch_util import dict_apply, set_seed
from dexmani_policy.env_runner.dexjoco_runner import DexJoCoRunner
from dexmani_policy.training import initialization
from dexmani_policy.training.config import check_fresh_output, normalize_output_dir, validate_config
from dexmani_policy.training.data import (
    build_dataset, build_train_loader, close_data_loader, record_data_processing,
    record_training_data, validate_resume_data,
)
from dexmani_policy.training.logging import (
    ExperimentLogger, print_param_count, print_training_recipe, to_log_scalars,
)
from dexmani_policy.training.optimization import (
    backward_microbatches, build_optimizer, build_scheduler, clip_gradients, restore_optimizer,
)
from dexmani_policy.training.runtime import training_signals


def build_trainer(cfg, *, checkpoint: Checkpoint | None = None) -> "PolicyTrainer":
    cfg = copy.deepcopy(cfg)
    output_dir = normalize_output_dir(cfg)
    with open_dict(cfg):
        remote_source = os.environ.get("DEXMANI_POLICY_REMOTE_SOURCE")
        cfg.pop("remote_source", None)
        if remote_source is not None:
            cfg.remote_source = remote_source
    warm_start = cfg.initialization.policy if checkpoint is None else None
    validate_config(cfg)
    model_checkpoint = checkpoint
    if warm_start is not None:
        warm_start = Path(warm_start).expanduser().resolve()
        model_checkpoint = load_checkpoint(warm_start)
        warm_config = model_checkpoint.effective_config
        if cfg.agent._target_ != warm_config["agent"]["_target_"] or cfg.action_key != warm_config["action_key"]:
            raise ValueError("Policy initialization requires the same policy and action semantics")
        if cfg.agent.get("center_rotvec", False) != warm_config["agent"].get("center_rotvec", False):
            raise ValueError("Policy initialization requires the same agent.center_rotvec setting")
        with open_dict(cfg):
            cfg.normalization = warm_config["normalization"]
            cfg.normalization_dims = warm_config["normalization_dims"]
            cfg.initialization.policy = str(warm_start)
        initialization.inherit_rgb_architecture(cfg.agent, warm_config["agent"])
    if checkpoint is None:
        check_fresh_output(output_dir)
    if model_checkpoint is None:
        initialization.resolve_rgb_architecture(cfg.agent, cfg.initialization.get("rgb"))
    set_seed(cfg.training.seed + (checkpoint.global_step if checkpoint is not None else 0))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    dataset, data_description = build_dataset(cfg)
    store = dataset.episode_store
    data_dt = data_description["control_dt_s"]
    runner_class = hydra.utils.get_class(cfg.env_runner._target_)
    dexjoco = issubclass(runner_class, DexJoCoRunner)
    record_data_processing(cfg, dataset, dexjoco=dexjoco)
    if checkpoint is None:
        record_training_data(cfg, store, dataset.train_mask, dataset.val_mask)
        policy_dt = data_dt
    else:
        saved = checkpoint.effective_config
        validate_resume_data(store, dataset.train_mask, dataset.val_mask, saved=saved)
        if dexjoco and cfg.data_processing != saved["data_processing"]:
            raise ValueError(
                f"Resume DexJoCo observation processing mismatch: "
                f"checkpoint={saved['data_processing']}, dataset={cfg.data_processing}"
            )
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
            cfg.normalization_dims = model.normalization_dims
            cfg.agent.update(model.effective_config)
        OmegaConf.resolve(cfg)
    return PolicyTrainer(cfg, model, dataset, checkpoint=checkpoint)


def run_trainer(trainer: "PolicyTrainer", *, max_updates: int | None = None) -> None:
    with cleanup_on_exit(trainer.close):
        trainer.compile_models()
        trainer.train(max_updates=max_updates)


class PolicyTrainer:
    def __init__(self, cfg, model, dataset, *, checkpoint=None):
        device = torch.device(cfg.training.device)
        if model.requires_ema_for_loss and not cfg.training.use_ema:
            raise ValueError(f"{type(model).__name__} requires training.use_ema=true")

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
            # Bind tensors after checkpoint assignment and device migration.
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
        self.milestone_steps = {
            (pct * self.total_train_steps + 99) // 100 for pct in (20, 40, 60, 80, 100)
        }

        print_param_count(model)
        print_training_recipe(self.cfg, updates_per_epoch=self.updates_per_epoch)
        self.logger = ExperimentLogger(cfg, resuming=checkpoint is not None)
        self.checkpoint_dir = self.logger.output_dir / "checkpoints"

    def compile_models(self):
        if self.cfg.training.use_compile:
            mode = self.cfg.training.compile_mode
            self.model.compile_backbone(mode=mode)
            if self.model.requires_ema_for_loss:
                self.ema_model.compile_backbone(mode=mode)

    def close(self):
        self.logger.close()

    def optimizer_step(self, finite_loss: torch.Tensor) -> torch.Tensor:
        grad_norm = clip_gradients(
            self.trainable_parameters, finite_loss, self.max_grad_norm, step=self.global_step,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema_updater is not None:
            self.ema_updater.step(self.global_step)
        return grad_norm

    def _forward_microbatch(self, microbatch, *, log_metrics: bool):
        batch, loss_options = microbatch
        count = batch["action"].shape[0]
        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16,
        ):
            loss, metrics = self.model(batch, log_metrics=log_metrics, **loss_options)
        return loss, metrics, count

    def _backward_group(self, iterator, group_size, *, log_metrics):
        batches = self.model.training_microbatches(islice(iterator, group_size), self.ema_model)
        return backward_microbatches(
            batches, group_size, self.trainable_parameters, partial(self._forward_microbatch, log_metrics=log_metrics),
            device=self.device, log_metrics=log_metrics,
        )

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

    def log_step(self, group_metrics, pbar, *, lr, epoch, grad_norm):
        metrics = {
            "train/lr": lr,
            "train/epoch": epoch,
            "train/grad_norm": grad_norm,
        }
        if self.max_grad_norm > 0:
            metrics["train/clip_ratio"] = grad_norm / self.max_grad_norm
        metrics.update({f"train/{key}": value for key, value in group_metrics.items()})
        metrics = to_log_scalars(metrics)
        pbar.set_postfix(
            loss=f"{metrics['train/loss']:.5f}",
            step=f"{self.global_step:,}",
            lr=f"{metrics['train/lr']:.2e}",
            refresh=False,
        )
        self.logger.log(metrics, step=self.global_step)

    def train(self, *, max_updates: int | None = None) -> None:
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
            with (
                cleanup_on_exit(self.model.eval),
                cleanup_on_exit(lambda: close_data_loader(self.train_loader, iterator)),
            ):
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
                        finite_loss, group_metrics = self._backward_group(iterator, group_size, log_metrics=record_group)
                        lr = self.optimizer.param_groups[0]["lr"]
                        grad_norm = self.optimizer_step(finite_loss)
                        self.global_step += 1
                        pbar.update(1)
                        if record_group:
                            self.log_step(group_metrics, pbar, lr=lr, epoch=epoch, grad_norm=grad_norm)

                        should_stop = stop.requested or self.global_step >= stop_step
                        if self.global_step in self.milestone_steps or should_stop:
                            self.save_checkpoint()
                        if should_stop:
                            break
                    if should_stop:
                        break
