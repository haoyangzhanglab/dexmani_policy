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

from dexmani_policy.common.pytorch_util import set_seed, worker_init_fn, fix_state_dict, compile_models
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.training.lr_scheduler import compute_num_training_steps
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    validate_config,
    build_scheduler,
)
from dexmani_policy.training.trainer import Trainer

register_resolvers()


def setup_ddp(rank: int, world_size: int):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


def ddp_worker(rank: int, world_size: int, cfg, gpu_ids):
    setup_ddp(rank, world_size)

    actual_gpu_id = gpu_ids[rank] if gpu_ids else rank
    device = torch.device(f'cuda:{actual_gpu_id}')
    torch.cuda.set_device(device)

    # DDP requires identical initial parameters across all ranks — use same seed
    set_seed(cfg.training.seed)

    dataset, normalizer = build_dataset_and_normalizer(cfg)

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
        prefetch_factor=cfg.dataloader.get('prefetch_factor', 2),
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

    # After model init, use different seeds per rank for augmentation diversity
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

    # Load checkpoint — must happen before torch.compile to avoid
    # recompilation on load_state_dict-triggered guard failures.
    resume_global_step = 0
    resume_start_epoch = 0
    try:
        ckpt_path = checkpoint_store.resolve_path("latest")
        checkpoint = checkpoint_store.load(ckpt_path)
        model.load_state_dict(fix_state_dict(checkpoint.model_state, is_current_ddp=False), strict=True)
        if ema_model is not None and checkpoint.ema_model_state is not None:
            ema_model.load_state_dict(fix_state_dict(checkpoint.ema_model_state, is_current_ddp=False), strict=True)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        resume_global_step = checkpoint.global_step
        resume_start_epoch = checkpoint.epoch + 1
    except FileNotFoundError:
        checkpoint = None

    # torch.compile must happen before DDP wrapping and after checkpoint load.
    # Use compile_models() for unified single-GPU/DDP behavior: backbone only, mode='reduce-overhead'.
    if cfg.training.get('use_compile', False):
        compile_models(model, ema_model)

    ddp_model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id,
                    find_unused_parameters=False, gradient_as_bucket_view=True,
                    static_graph=True)

    total_steps = compute_num_training_steps(cfg, batches_per_epoch)

    # Warn if num_training_steps changed since the checkpoint was saved.
    # Duplicates trainer.py:load_for_resume logic — necessary because DDP
    # bypasses load_for_resume (passes resume_state directly to train()).
    if checkpoint is not None:
        saved_steps = checkpoint.train_params.get('num_training_steps') if checkpoint.train_params else None
        if saved_steps is not None and saved_steps != total_steps:
            import warnings
            warnings.warn(
                f"DDP Resume: num_training_steps mismatch — saved={saved_steps}, current={total_steps}. "
                f"The LR schedule was originally configured for {saved_steps} total steps; "
                f"the current config would produce {total_steps}. "
                f"The scheduler state_dict will be restored from the checkpoint, but the "
                f"underlying schedule curve may be distorted. "
                f"Consider matching the original dataloader configuration to avoid LR drift.",
                UserWarning,
            )

    scheduler = build_scheduler(
        cfg, optimizer, batches_per_epoch,
        last_epoch=resume_global_step - 1 if checkpoint is None else -1,
    )
    # Restore full scheduler state (aligned with single-GPU load_for_resume).
    # When resuming, load_state_dict overrides the initial state set by
    # last_epoch, so last_epoch=-1 above is intentional — it avoids a
    # misleading intermediate state before the checkpoint state is applied.
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint.scheduler_state)

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
        use_compile=False,  # already applied before DDP wrapping above
        is_main_process=(rank == 0),
        distributed=True,
        train_sampler=train_sampler,
        num_training_steps=total_steps,
    )

    # Broadcast normalizer state from rank 0 to all ranks
    norm_state = model.normalizer.state_dict()
    for key in sorted(norm_state):
        if isinstance(norm_state[key], torch.Tensor):
            dist.broadcast(norm_state[key], src=0)
    if rank != 0:
        model.normalizer.load_state_dict(norm_state)
    dist.barrier()

    resume_state = (resume_global_step, resume_start_epoch)
    try:
        trainer.train(resume_state=resume_state)
    finally:
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    validate_config(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP training requires GPU.")

    num_gpus = cfg.training.get("num_gpus")

    if num_gpus is None:
        raise ValueError(
            "train_ddp.py requires 'training.num_gpus' to be set in config. "
            "Please use a DDP config (e.g., ddp/maniflow) or add 'training.num_gpus' to your config."
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

    # Child processes cannot access Hydra runtime resolvers — resolve all
    # interpolations before mp.spawn.
    OmegaConf.resolve(cfg)

    if 'MASTER_PORT' not in os.environ:
        # Auto-assign a free port.  There is a theoretical TOCTOU race
        # between close() and mp.spawn() — another process could bind the
        # same port.  In practice this window is microseconds and shared
        # training machines are the only scenario where it matters.
        # If you hit an address-in-use error, set MASTER_PORT explicitly.
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
