import os
import socket
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from dexmani_policy.agents.core.normalizer import AffineNormalizer
from dexmani_policy.training.builder import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
    build_train_loader,
    compile_models,
)
from dexmani_policy.training.common.checkpoint import CheckpointStore
from dexmani_policy.training.common.logging import print_training_recipe
from dexmani_policy.training.resume import (
    build_resume_contract,
    resolve_training_config,
    restore_training_state,
)
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.validation import validate_config
from dexmani_policy.training.workspace import prepare_output_dir
from dexmani_policy.utils.config import register_resolvers, resolve_resume_source, set_project_root
from dexmani_policy.utils.runtime import set_seed

register_resolvers()
set_project_root()


def validate_gpu_ids(num_gpus, gpu_ids, available_gpus):
    if type(num_gpus) is not int or num_gpus <= 1:
        raise ValueError("DDP num_gpus must be an integer > 1")
    ids = list(range(num_gpus)) if gpu_ids is None else list(gpu_ids)
    if len(ids) != num_gpus:
        raise ValueError("Length of gpu_ids must match num_gpus")
    if any(type(i) is not int or i < 0 or i >= available_gpus for i in ids):
        raise ValueError(f"gpu_ids must contain integers in [0, {available_gpus})")
    if len(set(ids)) != len(ids):
        raise ValueError("gpu_ids must not contain duplicates")
    return ids


def ddp_worker(rank: int, world_size: int, cfg, gpu_ids, dataset, normalizer_state):
    actual_gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{actual_gpu_id}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    workspace = None
    try:
        # Use the same seed for model initialization on every rank.
        set_seed(cfg.training.seed)

        resume_from = cfg.get("resume_from")
        checkpoint = (
            CheckpointStore(Path(resume_from).parent).load(resume_from) if resume_from else None
        )
        normalizer = None
        if normalizer_state is not None:
            normalizer = AffineNormalizer()
            normalizer.load_state_dict(normalizer_state)

        train_loader = build_train_loader(cfg, dataset, rank=rank, world_size=world_size)

        model, ema_model, ema_updater = build_model_and_ema(
            cfg, device, normalizer, rank=rank, checkpoint=checkpoint, dataset=dataset
        )

        # Use different seeds per rank for observation augmentation.
        set_seed(cfg.training.seed + rank)

        optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=rank == 0)

        if rank == 0:
            print_training_recipe(
                cfg,
                world_size=world_size,
                batches_per_epoch=len(train_loader),
                samples_per_rank=train_loader.sampler.num_samples,
            )
            workspace = hydra.utils.instantiate(cfg.workspace)

        resume_contract = build_resume_contract(train_loader)
        resume_state = (0, 0, 0)
        if resume_from is not None:
            resume_state = restore_training_state(
                checkpoint,
                resume_contract=resume_contract,
                effective_config=cfg,
                model=model,
                ema_updater=ema_updater,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                rank=rank,
            )

        del checkpoint
        if rank == 0:
            workspace.save_hydra_config(cfg)

        # Compile restored backbones before DDP wrapping.
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
            # Only fresh EMA follows the raw state synchronized by DDP.
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
            effective_config=cfg,
        )

        trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))
    finally:
        try:
            if workspace is not None:
                workspace.close()
        finally:
            dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    source = resolve_resume_source(cfg.get("resume_from"))
    checkpoint = CheckpointStore(Path(source).parent).load(source) if source else None
    cfg = resolve_training_config(cfg, resume_checkpoint=checkpoint)
    validate_config(cfg)
    with open_dict(cfg):
        cfg.resume_from = source
    prepare_output_dir(cfg.workspace.output_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP training requires GPU.")

    num_gpus = cfg.training.get("num_gpus")

    gpu_ids = validate_gpu_ids(
        num_gpus, cfg.training.get("gpu_ids", None), torch.cuda.device_count()
    )
    print(f"Using GPUs: {gpu_ids}")

    os.environ.setdefault("MASTER_ADDR", "localhost")

    set_seed(cfg.training.seed)
    dataset, normalizer = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
    normalizer_state = normalizer.state_dict() if normalizer is not None else None
    del normalizer, checkpoint
    # Child processes cannot access Hydra runtime resolvers.
    OmegaConf.resolve(cfg)
    print(f"[Dataset] Prepared once in parent pid={os.getpid()} for {num_gpus} ranks")

    if "MASTER_PORT" not in os.environ:
        # The probe does not reserve the port; set MASTER_PORT if launches collide.
        with socket.socket() as sock:
            sock.bind(("", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])

    mp.spawn(
        ddp_worker,
        args=(num_gpus, cfg, gpu_ids, dataset, normalizer_state),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
