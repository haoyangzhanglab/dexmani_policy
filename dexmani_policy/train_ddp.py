import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


import copy
import hydra
import torch
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dexmani_policy.common.pytorch_util import set_seed, optimizer_to
from dexmani_policy.datasets.augmentation import worker_init_fn
from dexmani_policy.training.ddp_trainer import DDPTrainer
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.lr_scheduler import get_scheduler

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    if 'MASTER_PORT' not in os.environ:
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ['MASTER_PORT'] = str(port)

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

    # 为多任务数据集设置不同的种子（如果有 seed 参数）
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if 'seed' in dataset_cfg:
        dataset_cfg['seed'] = dataset_cfg['seed'] + rank
    dataset = hydra.utils.instantiate(dataset_cfg)
    normalizer = dataset.get_normalizer()

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

    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            dataset.get_validation_dataset(),
            batch_size=cfg.val_dataloader.batch_size,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.pin_memory,
            persistent_workers=cfg.val_dataloader.persistent_workers,
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )

    model = hydra.utils.instantiate(cfg.agent)

    # 对于多任务 Agent，验证配置一致性
    if hasattr(model, 'num_tasks') and hasattr(dataset, 'task_names'):
        if model.num_tasks != len(dataset.task_names):
            raise ValueError(
                f"Configuration mismatch: agent.num_tasks={model.num_tasks} but "
                f"dataset has {len(dataset.task_names)} tasks {dataset.task_names}. "
                f"Please update agent.num_tasks in the config file."
            )

    model.load_normalizer_from_dataset(normalizer)
    model.to(device)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        try:
            ema_model = copy.deepcopy(model)
        except Exception as e:
            warnings.warn(f"copy.deepcopy(model) failed ({e}), falling back to fresh instantiation.")
            ema_model = hydra.utils.instantiate(cfg.agent)
            ema_model.load_normalizer_from_dataset(normalizer)
            ema_model.load_state_dict(model.state_dict())

        ema_model.to(device)
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    optimizer = model.configure_optimizer(**cfg.optimizer)

    gradient_accumulate_every = cfg.training.loop.gradient_accumulate_every
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = (batches_per_epoch + gradient_accumulate_every - 1) // gradient_accumulate_every
    total_steps = optimizer_steps_per_epoch * cfg.training.loop.num_epochs

    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    workspace = None
    if rank == 0:
        workspace = hydra.utils.instantiate(cfg.workspace)
        workspace.save_hydra_config(cfg)
    else:
        from dexmani_policy.training.common.workspace import ReadOnlyWorkspace
        workspace = ReadOnlyWorkspace(output_dir=cfg.workspace.output_dir)

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

    mp.spawn(
        ddp_worker,
        args=(num_gpus, cfg, gpu_ids),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
