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
from dexmani_policy.common.pytorch_util import dict_apply, optimizer_to, fix_state_dict


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

        # 只包装训练模型，EMA 模型不需要梯度同步
        self.raw_model = model
        ddp_model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id)

        self.trainer = Trainer(
            device=device,
            model=ddp_model,
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
        self.trainer.is_main_process = (rank == 0)

        self.train_sampler = train_loader.sampler
        if not isinstance(self.train_sampler, DistributedSampler):
            raise ValueError("train_loader must use DistributedSampler for DDP training")


    def synchronize_states(self):
        """同步 normalizer：rank 0 broadcast state_dict 到所有进程

        注意：
            - 只同步 normalizer，不同步 optimizer/scheduler
            - optimizer/scheduler 在各 rank 独立创建，状态由 DDP 梯度同步保证一致
            - EMA 模型不需要同步（不参与梯度计算，由训练模型更新驱动）
        """
        norm_state = self.raw_model.normalizer.state_dict()
        for key in norm_state:
            if isinstance(norm_state[key], torch.Tensor):
                dist.broadcast(norm_state[key], src=0)
        if self.rank != 0:
            self.raw_model.normalizer.load_state_dict(norm_state)


    def train(self, resume_tag: str = "latest"):
        torch.set_float32_matmul_precision('high')
        try:
            checkpoint = self.trainer.workspace.load_checkpoint(resume_tag)

            model_state = fix_state_dict(checkpoint.model_state, is_current_ddp=False)
            self.raw_model.load_state_dict(model_state, strict=True)

            if self.trainer.use_ema and checkpoint.ema_model_state is not None:
                ema_state = fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False)
                self.trainer.ema_model.load_state_dict(ema_state, strict=True)

            self.trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
            self.trainer.scheduler.load_state_dict(checkpoint.scheduler_state)

            if self.trainer.ema_updater is not None and checkpoint.ema_updater_state is not None:
                self.trainer.ema_updater.load_state_dict(checkpoint.ema_updater_state)

            global_step = checkpoint.global_step
            start_epoch = checkpoint.epoch + 1

            if self.is_main:
                print(f"Resuming training from epoch {start_epoch}, step {global_step}")
        except FileNotFoundError:
            global_step = 0
            start_epoch = 0

        optimizer_to(self.trainer.optimizer, self.trainer.device)
        if self.trainer.use_ema and self.trainer.ema_model is not None:
            self.trainer.ema_model.to(self.trainer.device)
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

            self.trainer.on_epoch_start(epoch)

            self.trainer.model.train()

            # 避免跨 epoch 边界的梯度累积错位
            self.trainer.accum_step = 0
            self.trainer.optimizer.zero_grad(set_to_none=True)

            last_batch = None
            for batch in self.trainer.train_loader:
                last_batch = batch
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

            # flush 未完成的梯度累积，避免浪费计算
            self.trainer.flush_gradient_accumulation()

            self.trainer.model.eval()

            if self.is_main:
                self.trainer.finish_epoch(
                    epoch, global_step,
                    last_batch=last_batch,
                    eval_model=(self.trainer.ema_model if self.trainer.use_ema else self.raw_model),
                    checkpoint_model=self.raw_model,
                )

            dist.barrier()
