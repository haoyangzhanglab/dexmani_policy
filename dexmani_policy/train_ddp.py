import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)


import hydra
import torch
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.common.pytorch_util import worker_init_fn
from dexmani_policy.training.ddp_trainer import DDPTrainer
from dexmani_policy.train import (
    validate_config,
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
)

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


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
    set_seed(cfg.training.seed + rank)

    # 构建 dataset 和 normalizer（DDP 模式：每个 rank 不同的 seed）
    dataset, normalizer = build_dataset_and_normalizer(cfg, rank=rank, world_size=world_size)

    # 构建 DDP dataloader（使用 DistributedSampler）
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
        worker_init_fn=worker_init_fn,
    )

    # 验证集只在 rank 0 创建
    val_loader = None
    if rank == 0:
        val_dataset = dataset.get_validation_dataset()
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.val_dataloader.batch_size,
                num_workers=cfg.val_dataloader.num_workers,
                pin_memory=cfg.val_dataloader.pin_memory,
                persistent_workers=cfg.val_dataloader.persistent_workers,
                shuffle=False,
                worker_init_fn=worker_init_fn,
            )

    # 构建 model 和 EMA（复用单卡逻辑）
    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)

    # 构建 optimizer 和 scheduler（复用单卡逻辑）
    batches_per_epoch = len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, batches_per_epoch)

    # workspace 只在 rank 0 创建完整版本，其他 rank 使用 ReadOnly
    workspace = None
    if rank == 0:
        workspace = hydra.utils.instantiate(cfg.workspace)
        workspace.save_hydra_config(cfg)
    else:
        from dexmani_policy.training.common.workspace import ReadOnlyWorkspace
        workspace = ReadOnlyWorkspace(output_dir=cfg.workspace.output_dir)

    # env_runner 只在 rank 0 创建
    env_runner = None
    if rank == 0 and cfg.training.loop.eval_interval_epochs > 0 and cfg.get("env_runner") is not None:
        env_runner = hydra.utils.instantiate(cfg.env_runner)

    ddp_trainer = DDPTrainer(
        rank=rank,
        world_size=world_size,
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
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
        actual_gpu_id=actual_gpu_id,
    )

    ddp_trainer.train(resume_tag="latest")

    cleanup_ddp()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    # 验证配置
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
    if 'MASTER_PORT' not in os.environ:
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        os.environ['MASTER_PORT'] = str(sock.getsockname()[1])
        sock.close()

    # 子进程无法访问 Hydra runtime resolver，必须在 spawn 前解析所有插值
    OmegaConf.resolve(cfg)

    mp.spawn(
        ddp_worker,
        args=(num_gpus, cfg, gpu_ids),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
