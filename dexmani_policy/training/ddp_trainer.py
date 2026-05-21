import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Optional, Dict, Any

from dexmani_policy.training.common.logging import to_log_scalars
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
        workspace,  # Optional[TrainWorkspace] — rank 0 only
        checkpoint_store,  # CheckpointStore — all ranks
        train_loop_cfg: TrainLoopConfig,
        use_ema_teacher_for_consistency: bool,
        actual_gpu_id: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        self.checkpoint_store = checkpoint_store

        # 只包装训练模型，EMA 模型不需要梯度同步
        self.raw_model = model
        ddp_model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id,
                        find_unused_parameters=False)

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
            is_main_process=(rank == 0),
        )

        self.train_sampler = train_loader.sampler
        if not isinstance(self.train_sampler, DistributedSampler):
            raise ValueError("train_loader must use DistributedSampler for DDP training")


    def synchronize_states(self):
        norm_state = self.raw_model.normalizer.state_dict()
        for key in sorted(norm_state):
            if isinstance(norm_state[key], torch.Tensor):
                dist.broadcast(norm_state[key], src=0)
        if self.rank != 0:
            self.raw_model.normalizer.load_state_dict(norm_state)


    def train(self, resume_tag: str = "latest"):
        torch.set_float32_matmul_precision('high')
        try:
            checkpoint = self.checkpoint_store.load(self.checkpoint_store.resolve_path(resume_tag))

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
                from termcolor import cprint
                cprint(f"Resuming training from epoch {start_epoch}, step {global_step}", "cyan")
        except FileNotFoundError:
            global_step = 0
            start_epoch = 0

        optimizer_to(self.trainer.optimizer, self.trainer.device)
        if self.trainer.use_ema and self.trainer.ema_model is not None:
            self.trainer.ema_model.to(self.trainer.device)
        self.synchronize_states()
        dist.barrier()

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
                if self.trainer.train_sampling_batch is None:
                    self.trainer.train_sampling_batch = dict_apply(
                        batch, lambda x: x.to(self.trainer.device, non_blocking=True))
                last_batch = batch
                self.trainer.current_epoch = epoch
                self.trainer.global_step = global_step

                # NaN check BEFORE backward as a collective so no rank enters
                # the DDP gradient all-reduce while another has already bailed out.
                batch = dict_apply(batch, lambda x: x.to(self.trainer.device, non_blocking=True))
                loss_kwargs = {'ema_backbone': self.trainer.ema_model.action_decoder.model} if self.trainer.use_ema_teacher_for_consistency else {}
                raw_loss, log_dict = self.trainer.model.compute_loss(batch, **loss_kwargs)

                nan_flag = torch.tensor(
                    [0 if torch.isfinite(raw_loss) else 1],
                    dtype=torch.int, device=self.trainer.device,
                )
                dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
                if nan_flag.item():
                    self.trainer._save_nan_debug(raw_loss.item() if torch.isfinite(raw_loss) else float('nan'))
                    self.trainer.optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch={epoch}, step={global_step}. "
                        f"Training aborted on all ranks."
                    )

                scaled_loss = raw_loss / self.trainer.grad_accum_steps
                scaled_loss.backward()

                self.trainer.accum_step += 1
                if self.trainer.accum_step % self.trainer.grad_accum_steps == 0:
                    self.trainer.apply_gradient_step()

                global_step += 1

                if self.is_main and (global_step % self.trainer.log_interval_steps) == 0:
                    step_metrics = {"train/lr": self.trainer.scheduler.get_last_lr()[0]}
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

            # rank 0 独享 logging/checkpoint/eval；
            # validate() 的 @torch.no_grad() 确保 DDP 不会在此处同步参数。
            finish_error = None
            if self.is_main:
                try:
                    self.trainer.finish_epoch(
                        epoch, global_step,
                        last_batch=last_batch,
                        eval_model=(self.trainer.ema_model if self.trainer.use_ema else self.raw_model),
                        checkpoint_model=self.raw_model,
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("[Rank 0] finish_epoch failed, terminating.")
                    finish_error = e

            error_flag = torch.tensor(
                [1 if finish_error is not None else 0],
                dtype=torch.int,
                device=self.trainer.device,
            )
            dist.broadcast(error_flag, src=0)
            dist.barrier()

            if error_flag.item():
                raise RuntimeError(
                    "Training aborted due to error in finish_epoch on rank 0"
                ) from (finish_error if finish_error is not None else None)
