"""Train a policy with one spawned process per configured GPU."""

import os
import socket

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from dexmani_policy.common.checkpoint import load_training_checkpoint
from dexmani_policy.common.config import register_resolvers, set_project_root
from dexmani_policy.common.normalization import AffineNormalizer
from dexmani_policy.common.runtime import set_seed
from dexmani_policy.training.builder import (
    build_dataset_and_normalizer,
    build_training_components,
    build_trainer,
    compile_models,
    prepare_training,
)
from dexmani_policy.training.resume import restore_training_state

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
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    components = None
    try:
        # Use the same seed for model initialization on every rank.
        set_seed(cfg.training.seed)

        resume_from = cfg.get("resume_from")
        checkpoint = load_training_checkpoint(resume_from) if resume_from else None
        normalizer = None
        if normalizer_state is not None:
            normalizer = AffineNormalizer()
            normalizer.load_state_dict(normalizer_state)

        components = build_training_components(
            cfg,
            device,
            dataset,
            normalizer,
            rank=rank,
            world_size=world_size,
            checkpoint=checkpoint,
        )

        resume_state = None
        if resume_from is not None:
            resume_state = restore_training_state(checkpoint, components, effective_config=cfg)

        del checkpoint
        if rank == 0:
            components.workspace.save_hydra_config(cfg)

        # Compile restored backbones before DDP wrapping.
        if cfg.training.get("use_compile", False):
            compile_models(
                components.model,
                components.ema_model,
                mode=cfg.training.get("compile_mode", "reduce-overhead"),
            )

        components.model = DDP(
            components.model,
            device_ids=[gpu_id],
            output_device=gpu_id,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

        if resume_from is None and components.ema_model is not None:
            # Only fresh EMA follows the raw state synchronized by DDP.
            components.ema_model.load_state_dict(components.model.module.state_dict(), strict=True)

        trainer = build_trainer(cfg, components)
        trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))
    finally:
        try:
            if components is not None and components.workspace is not None:
                components.workspace.close()
        finally:
            dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    cfg, checkpoint = prepare_training(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP training requires GPU.")

    num_gpus = cfg.training.get("num_gpus")
    gpu_ids = validate_gpu_ids(num_gpus, cfg.training.get("gpu_ids"), torch.cuda.device_count())
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
