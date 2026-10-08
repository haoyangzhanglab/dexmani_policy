import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pathlib
import socket

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from dexmani_policy.training.checkpoint import CheckpointStore
from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.utils.path import set_project_root
from dexmani_policy.utils.random import set_seed
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
    compile_models,
    print_training_recipe,
    validate_config,
)
from dexmani_policy.training.resume import (
    build_resume_contract,
    build_train_loader,
    restore_training_state,
    validate_gpu_ids,
)
from dexmani_policy.training.run_identity import claim_run, resolve_resume_source
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig

register_resolvers()
set_project_root()


def setup_ddp(rank: int, world_size: int):
    """Initialise NCCL with the installed PyTorch default timeout."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )


def ddp_worker(rank: int, world_size: int, cfg, gpu_ids, resume_from=None):
    gpu_ids = validate_gpu_ids(world_size, gpu_ids, torch.cuda.device_count())
    actual_gpu_id = gpu_ids[rank] if gpu_ids else rank
    device = torch.device(f"cuda:{actual_gpu_id}")
    torch.cuda.set_device(device)
    setup_ddp(rank, world_size)

    # DDP requires identical initial parameters across all ranks — use same seed
    set_seed(cfg.training.seed)

    checkpoint = CheckpointStore(pathlib.Path(resume_from).parent).load(resume_from) if resume_from else None
    dataset, normalizer = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)

    train_loader = build_train_loader(cfg, dataset, rank=rank, world_size=world_size)

    model, ema_model, ema_updater = build_model_and_ema(
        cfg, device, normalizer, rank=rank, checkpoint=checkpoint
    )

    # After model init, use different seeds per rank for augmentation diversity
    set_seed(cfg.training.seed + rank)

    batches_per_epoch = len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        cfg, model, batches_per_epoch, verbose=rank == 0
    )

    if rank == 0:
        print_training_recipe(
            cfg, world_size=world_size, batches_per_epoch=batches_per_epoch
        )
        workspace = hydra.utils.instantiate(cfg.workspace)
    else:
        workspace = None

    resume_contract = build_resume_contract(
        cfg, model, train_loader, world_size=world_size
    )
    resume_state = (0, 0, 0)
    if resume_from is not None:
        resume_state = restore_training_state(
            checkpoint,
            resume_contract=resume_contract,
            model=model,
            ema_model=ema_model,
            ema_updater=ema_updater,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            rank=rank,
        )

    del checkpoint
    if rank == 0:
        workspace.save_hydra_config(cfg)

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

    if resume_from is None and ema_model is not None:
        # DDP synchronizes raw state only. Fresh EMA starts from that state;
        # never overwrite the separately restored historical EMA on resume.
        ema_model.load_state_dict(model.state_dict(), strict=True)

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
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        use_bfloat16=cfg.training.get("use_bfloat16", False),
        is_main_process=(rank == 0),
        distributed=True,
        resume_contract=resume_contract,
        batches_per_epoch=batches_per_epoch,
    )

    try:
        trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))
    finally:
        # Safe even after a collective timeout — destroy_process_group()
        # performs only local cleanup (no communication).
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    from dexmani_policy.training.resume import resolve_training_config
    cfg = resolve_training_config(cfg)
    validate_config(cfg)
    with open_dict(cfg):
        cfg.resume_from = resolve_resume_source(cfg.get("resume_from"))
        cfg.workspace.claim_token = claim_run(
            cfg.workspace.output_dir, resume_from=cfg.resume_from
        )

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
