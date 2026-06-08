import os
import pathlib
import socket

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dexmani_policy.common.pytorch_util import set_seed, worker_init_fn, optimizer_to, fix_state_dict
from dexmani_policy.common.resolver import register_resolvers
from dexmani_policy.train import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    validate_config,
)
from dexmani_policy.training.common.checkpoint_io import CheckpointStore
from dexmani_policy.training.common.lr_scheduler import get_scheduler
from dexmani_policy.training.trainer import Trainer

register_resolvers()


def setup_ddp(rank: int, world_size: int):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


def cleanup_ddp():
    dist.destroy_process_group()


def ddp_worker(rank: int, world_size: int, cfg, gpu_ids):
    setup_ddp(rank, world_size)

    actual_gpu_id = gpu_ids[rank] if gpu_ids else rank
    device = torch.device(f'cuda:{actual_gpu_id}')
    torch.cuda.set_device(device)

    # DDP 要求所有 rank 的模型初始参数一致，此处用相同 seed
    set_seed(cfg.training.seed)

    dataset, normalizer = build_dataset_and_normalizer(cfg, rank=rank, world_size=world_size)

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=cfg.dataloader.shuffle,
        seed=cfg.training.seed,
    )

    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
        drop_last=cfg.dataloader.get('drop_last', False),
        worker_init_fn=worker_init_fn,
    )

    val_loader = None
    if rank == 0:
        val_dataset = dataset.get_validation_dataset()
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                worker_init_fn=worker_init_fn,
                **cfg.val_dataloader,
            )

    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)

    # 模型初始化完成后，各 rank 使用不同 seed 以增加数据增强多样性
    set_seed(cfg.training.seed + rank)

    batches_per_epoch = len(train_loader)
    optimizer = model.configure_optimizer(**cfg.optimizer)

    if rank == 0:
        workspace = hydra.utils.instantiate(cfg.workspace)
        workspace.save_hydra_config(cfg)
        checkpoint_store = workspace.checkpoint_store
    else:
        checkpoint_dir = pathlib.Path(cfg.workspace.output_dir) / "checkpoints"
        workspace = None
        checkpoint_store = CheckpointStore(checkpoint_dir)

    env_runner = None
    if rank == 0 and cfg.training.loop.eval_interval_epochs > 0 and cfg.get("env_runner") is not None:
        env_runner = hydra.utils.instantiate(cfg.env_runner)

    ddp_model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id,
                    find_unused_parameters=False)

    # 加载 checkpoint（在 scheduler 前获取 global_step 用于 last_epoch）
    try:
        ckpt_path = checkpoint_store.resolve_path("latest")
        checkpoint = checkpoint_store.load(ckpt_path)
        model.load_state_dict(fix_state_dict(checkpoint.model_state, is_current_ddp=False), strict=True)
        if ema_model is not None and checkpoint.ema_model_state is not None:
            ema_model.load_state_dict(fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False), strict=True)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        resume_state = (checkpoint.global_step, checkpoint.epoch + 1)
    except FileNotFoundError:
        resume_state = (0, 0)

    total_steps = batches_per_epoch * cfg.training.loop.num_epochs
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
        last_epoch=resume_state[0] - 1,
    )

    trainer = Trainer(
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
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
        max_grad_norm=cfg.training.get('max_grad_norm', 1.0),
        use_bfloat16=cfg.training.get('use_bfloat16', False),
        is_main_process=(rank == 0),
        distributed=True,
        train_sampler=train_sampler,
    )

    optimizer_to(optimizer, device)
    if trainer.use_ema:
        ema_model.to(device)

    norm_state = model.normalizer.state_dict()
    for key in sorted(norm_state):
        if isinstance(norm_state[key], torch.Tensor):
            dist.broadcast(norm_state[key], src=0)
    if rank != 0:
        model.normalizer.load_state_dict(norm_state)
    dist.barrier()

    try:
        trainer.train(resume_state=resume_state)
    finally:
        cleanup_ddp()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    validate_config(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP training requires GPU.")

    num_gpus = cfg.training.get("num_gpus")

    if num_gpus is None:
        raise ValueError(
            "train_ddp.py requires 'training.num_gpus' to be set in config. "
            "Please use a DDP config file (e.g., maniflow_ddp.yaml) or add 'training.num_gpus' to your config."
        )

    if num_gpus <= 1:
        raise ValueError(
            f"train_ddp.py requires num_gpus > 1, got {num_gpus}. "
            f"Please use train.py for single-GPU training."
        )

    available_gpus = torch.cuda.device_count()
    gpu_ids = cfg.training.get("gpu_ids", None)
    if gpu_ids is not None:
        if len(gpu_ids) != num_gpus:
            raise ValueError(
                f"Length of gpu_ids ({len(gpu_ids)}) must match num_gpus ({num_gpus}). "
                f"Got gpu_ids={gpu_ids}"
            )
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(
                    f"GPU {gpu_id} not available. Only {available_gpus} GPUs detected."
                )
        print(f"Using GPUs: {gpu_ids}")
    else:
        if num_gpus > available_gpus:
            raise ValueError(
                f"Requested {num_gpus} GPUs but only {available_gpus} available."
            )
        gpu_ids = list(range(num_gpus))
        print(f"Using default GPUs: {gpu_ids}")

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'

    # 子进程无法访问 Hydra runtime resolver，必须在 spawn 前解析所有插值
    OmegaConf.resolve(cfg)

    if 'MASTER_PORT' not in os.environ:
        sock = socket.socket()
        sock.bind(('', 0))
        os.environ['MASTER_PORT'] = str(sock.getsockname()[1])
        sock.close()

    mp.spawn(
        ddp_worker,
        args=(num_gpus, cfg, gpu_ids),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
