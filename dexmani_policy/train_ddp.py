import datetime
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pathlib
import socket

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
)
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import (
    compile_models,
    set_project_root,
    set_seed,
)
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_scheduler,
    validate_config,
    validate_gradient_accumulation,
)
from dexmani_policy.training.lr_scheduler import compute_num_training_steps
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.resume import (
    build_train_loader, build_resume_contract, restore_training_state, validate_gpu_ids,
)

register_resolvers()
set_project_root()


def setup_ddp(rank: int, world_size: int):
    """Initialise the NCCL process group with a 30-minute timeout.

    Without an explicit timeout, the default is effectively unbounded — a
    hung or crashed rank causes all other ranks to hang indefinitely on
    the next collective (all_reduce, broadcast, barrier).  Thirty minutes
    is long enough to survive transient NCCL stalls under heavy I/O but
    short enough to avoid wasting cluster time on a truly dead rank.

    The timeout covers **every** collective in this process group:
    ``dist.all_gather`` in the NaN sentinel,
    ``dist.broadcast`` in normalizer sync, and the implicit
    barrier inside DDP backward.
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
    )


def ddp_worker(rank: int, world_size: int, cfg, gpu_ids, resume_from=None):
    gpu_ids = validate_gpu_ids(world_size, gpu_ids, torch.cuda.device_count())
    actual_gpu_id = gpu_ids[rank] if gpu_ids else rank
    device = torch.device(f"cuda:{actual_gpu_id}")
    torch.cuda.set_device(device)
    setup_ddp(rank, world_size)

    # DDP requires identical initial parameters across all ranks — use same seed
    set_seed(cfg.training.seed)

    dataset, normalizer = build_dataset_and_normalizer(cfg)

    train_loader = build_train_loader(cfg, dataset, rank=rank, world_size=world_size)
    train_sampler = train_loader.sampler

    model, ema_model, ema_updater = build_model_and_ema(
        cfg, device, normalizer, rank=rank
    )

    # After model init, use different seeds per rank for augmentation diversity
    set_seed(cfg.training.seed + rank)

    batches_per_epoch = len(train_loader)
    validate_gradient_accumulation(
        batches_per_epoch,
        cfg.training.get("loop", {}).get("gradient_accumulation_steps", 1),
    )
    optimizer = model.configure_optimizer(**cfg.optimizer)

    if rank == 0:
        workspace = hydra.utils.instantiate(cfg.workspace)
        workspace.save_hydra_config(cfg)
        checkpoint_store = workspace.checkpoint_store
    else:
        checkpoint_dir = pathlib.Path(cfg.workspace.output_dir) / "checkpoints"
        workspace = None
        checkpoint_store = CheckpointStore(checkpoint_dir)

    total_steps = compute_num_training_steps(cfg)
    scheduler = build_scheduler(cfg, optimizer)
    resume_contract = build_resume_contract(cfg, model, train_loader, world_size=world_size)
    resume_state = (0, 0, 0)
    if resume_from is not None:
        checkpoint = checkpoint_store.load(checkpoint_store.resolve_path(resume_from))
        resume_state = restore_training_state(
            checkpoint, resume_contract=resume_contract, model=model,
            ema_model=ema_model, ema_updater=ema_updater, optimizer=optimizer,
            scheduler=scheduler, device=device, rank=rank,
        )

    # torch.compile must happen before DDP wrapping and after checkpoint load.
    # Use compile_models() for unified single-GPU/DDP behavior: backbone only.
    if cfg.training.get("use_compile", False):
        compile_models(
            model, ema_model, mode=cfg.training.get("compile_mode", "reduce-overhead")
        )

    ddp_model = DDP(
        model,
        device_ids=[actual_gpu_id],
        output_device=actual_gpu_id,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )

    trainer = Trainer(
        device=device,
        model=ddp_model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        workspace=workspace,
        train_loop_cfg=TrainLoopConfig(
            **OmegaConf.to_container(cfg.training.loop, resolve=True)
        ),
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        fast_grad_finite_check=cfg.training.get("fast_grad_finite_check", False),
        use_bfloat16=cfg.training.get("use_bfloat16", False),
        use_compile=False,  # already applied before DDP wrapping above
        is_main_process=(rank == 0),
        distributed=True,
        train_sampler=train_sampler,
        num_training_steps=total_steps,
        resume_contract=resume_contract,
    )

    # Broadcast normalizer state from rank 0 to all ranks
    norm_state = model.normalizer.state_dict()
    for key in norm_state:
        if isinstance(norm_state[key], torch.Tensor):
            dist.broadcast(norm_state[key], src=0)
    if rank != 0:
        model.normalizer.load_state_dict(norm_state)

    try:
        trainer.train(resume_state=resume_state)
    finally:
        # Safe even after a collective timeout — destroy_process_group()
        # performs only local cleanup (no communication).
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

    gpu_ids = validate_gpu_ids(
        num_gpus, cfg.training.get("gpu_ids", None), torch.cuda.device_count()
    )
    print(f"Using GPUs: {gpu_ids}")

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    # Child processes cannot access Hydra runtime resolvers — resolve all
    # interpolations before mp.spawn.
    OmegaConf.resolve(cfg)

    if "MASTER_PORT" not in os.environ:
        # Auto-assign a free port.  There is a theoretical TOCTOU race
        # between close() and mp.spawn() — another process could bind the
        # same port.  In practice this window is microseconds and shared
        # training machines are the only scenario where it matters.
        # If you hit an address-in-use error, set MASTER_PORT explicitly.
        sock = socket.socket()
        sock.bind(("", 0))
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        sock.close()

    resume_from = cfg.get("resume_from", None)
    mp.spawn(
        ddp_worker,
        args=(num_gpus, cfg, gpu_ids, resume_from),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
