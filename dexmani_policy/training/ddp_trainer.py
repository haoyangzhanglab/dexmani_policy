import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Optional, Dict, Any

from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.checkpoint_io import TrainCheckpoint
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.training.common.ddp_util import unwrap_model, get_model_state_dict


class DDPTrainer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        model,
        ema_model,
        ema_updater,
        optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        env_runner,
        workspace: TrainWorkspace,
        train_loop_cfg: TrainLoopConfig,
        use_ema_teacher_for_consistency: bool,
        actual_gpu_id: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        # 只对主 model 做 DDP 包装；EMA model 是 eval/no_grad 的纯推理副本，
        # 不参与梯度同步，不应被 DDP 包装（否则会注册不必要的 all-reduce hook，
        # 且通过 .module 才能访问真实属性，导致 compute_loss / validate 内部访问出错）。
        model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id)

        self.trainer = Trainer(
            device=device,
            model=model,
            ema_model=ema_model,
            ema_updater=ema_updater,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            env_runner=env_runner,
            workspace=workspace,
            train_loop_cfg=train_loop_cfg,
            use_ema_teacher_for_consistency=use_ema_teacher_for_consistency,
        )

        self.train_sampler = train_loader.sampler
        if not isinstance(self.train_sampler, DistributedSampler):
            raise ValueError("train_loader must use DistributedSampler for DDP training")


    def synchronize_states(self):
        """
        将 rank-0 的完整训练状态广播到所有进程。

        必须在 load_for_resume 之后调用：rank-0 已从 checkpoint 加载权重，
        其他 rank 只有随机初始化或空权重，需要由此函数对齐。
        DDP 初始化时只在第一次构造时广播参数，resume 后不会再次自动广播，
        因此需要手动同步以下三块状态：
          1. 主 model 参数（DDP-wrapped 的底层 module）
          2. EMA model 参数（未被 DDP 包装，rank-0 从 checkpoint 加载）
          3. Normalizer 参数（各 rank 从各自 dataset 实例化，数值应相同，但显式同步更安全）
        """
        # 1. 同步主 model 参数（unwrap DDP，广播 rank-0 的权重）
        main_model = unwrap_model(self.trainer.model)
        main_state = main_model.state_dict()
        for key in main_state:
            if isinstance(main_state[key], torch.Tensor):
                dist.broadcast(main_state[key], src=0)
        if self.rank != 0:
            main_model.load_state_dict(main_state)

        # 2. 同步 EMA 模型（未被 DDP 包装，需手动广播）
        if self.trainer.use_ema:
            ema_model = self.trainer.ema_model   # 直接是裸 Module，无需 unwrap
            ema_state = ema_model.state_dict()
            for key in ema_state:
                if isinstance(ema_state[key], torch.Tensor):
                    dist.broadcast(ema_state[key], src=0)
            if self.rank != 0:
                ema_model.load_state_dict(ema_state)

        # 3. 同步 Normalizer
        if hasattr(main_model, 'normalizer'):
            normalizer = main_model.normalizer
            if isinstance(normalizer, nn.Module):
                norm_state = normalizer.state_dict()
                for key in norm_state:
                    if isinstance(norm_state[key], torch.Tensor):
                        dist.broadcast(norm_state[key], src=0)
                if self.rank != 0:
                    normalizer.load_state_dict(norm_state)
            else:
                import warnings
                warnings.warn(
                    f"Normalizer type {type(normalizer).__name__} is not an nn.Module, "
                    "skipping synchronization. This may cause inconsistent normalization across ranks."
                )


    def train(self, resume_tag: str = "latest"):
        global_step, start_epoch = self.trainer.workspace.load_for_resume(
            model=unwrap_model(self.trainer.model),
            ema_model=unwrap_model(self.trainer.ema_model) if self.trainer.use_ema else None,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            tag_or_path=resume_tag,
        )

        self.synchronize_states()
        dist.barrier()

        epoch_pbar = None
        if self.is_main:
            epoch_pbar = tqdm(
                range(start_epoch, self.trainer.num_epochs),
                desc="Epoch",
                position=0,
                mininterval=1.0,
            )
        else:
            epoch_pbar = range(start_epoch, self.trainer.num_epochs)

        for epoch in epoch_pbar:
            self.train_sampler.set_epoch(epoch)

            self.trainer.model.train()

            for batch in self.trainer.train_loader:
                _, log_dict = self.trainer.train_one_step(batch)
                global_step += 1

                if self.is_main and (global_step % self.trainer.log_interval_steps) == 0:
                    step_metrics = {"train/lr": self.trainer.scheduler.get_last_lr()[0]}
                    from dexmani_policy.training.common.logging import to_log_scalars
                    for key, value in to_log_scalars(log_dict).items():
                        step_metrics[f"train/{key}"] = value

                    if hasattr(epoch_pbar, 'set_postfix'):
                        epoch_pbar.set_postfix(
                            global_step=global_step,
                            loss=step_metrics.get("train/loss", None),
                        )
                    self.trainer.workspace.log(step_metrics, step=global_step)

            self.trainer.model.eval()

            if self.is_main:
                epoch_metrics = {}
                epoch_end_tasks = self.trainer.plan_epoch_end_tasks(epoch)

                if epoch_end_tasks["sample"]:
                    sample_batch = dict_apply(
                        next(iter(self.trainer.train_loader)),
                        lambda x: x.to(self.trainer.device, non_blocking=True)
                    )
                    epoch_metrics["train/action_mse_error"] = self.trainer.compute_action_mse_for_one_batch(
                        unwrap_model(self.trainer.model), sample_batch
                    )

                if epoch_end_tasks["validate"]:
                    val_agent = unwrap_model(self.trainer.ema_model) if self.trainer.use_ema else unwrap_model(self.trainer.model)
                    val_loss = self.trainer.validate(val_agent)
                    if val_loss is not None:
                        epoch_metrics["val/loss"] = val_loss

                if epoch_end_tasks["evaluate"]:
                    eval_model = unwrap_model(self.trainer.ema_model) if self.trainer.use_ema else unwrap_model(self.trainer.model)
                    epoch_metrics.update(self.trainer.evaluate(eval_model))

                self.trainer.workspace.log(epoch_metrics, step=global_step)

                if epoch_end_tasks["save_checkpoint"]:
                    if self.trainer.enable_env_eval:
                        test_mean_score = epoch_metrics.get("eval/success_rate")
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
                        model_state=get_model_state_dict(self.trainer.model),
                        ema_model_state=get_model_state_dict(self.trainer.ema_model) if self.trainer.use_ema else None,
                        optimizer_state=self.trainer.optimizer.state_dict(),
                        scheduler_state=self.trainer.scheduler.state_dict(),
                        monitor=monitor,
                    )

                    if test_mean_score is None:
                        tag = f"epoch={epoch:04d}-step={global_step:08d}"
                    else:
                        tag = f"epoch={epoch:04d}-step={global_step:08d}-score={test_mean_score:.4f}"

                    checkpoint_path = self.trainer.workspace.save_checkpoint(tag, checkpoint)
                    self.trainer.workspace.save_latest(checkpoint_path)

                    if test_mean_score is not None:
                        self.trainer.workspace.save_topk(checkpoint_path, checkpoint)

            dist.barrier()
