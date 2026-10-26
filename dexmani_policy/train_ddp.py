"""Train a policy with one spawned process per configured GPU."""

import os
import socket
from contextlib import closing

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
from dexmani_policy.common.runtime import close_resources, set_seed
from dexmani_policy.training.builder import (
    build_training_components,
    build_trainer,
    compile_training_models,
    prepare_training,
)
from dexmani_policy.training.data import build_dataset_and_normalizer
from dexmani_policy.training.resume import restore_training_state

register_resolvers()
set_project_root()


def resolve_gpu_ids(num_gpus, gpu_ids, available_gpus):
    if type(num_gpus) is not int or num_gpus <= 1:
        raise ValueError("DDP num_gpus must be an integer > 1")
    ids = list(range(num_gpus)) if gpu_ids is None else list(gpu_ids)
    if any(type(i) is not int or i < 0 or i >= available_gpus for i in ids):
        raise ValueError(f"gpu_ids must contain integers in [0, {available_gpus})")
    if len(ids) != num_gpus or len(set(ids)) != num_gpus:
        raise ValueError("gpu_ids must specify num_gpus distinct GPUs")
    return ids


def train_distributed(rank: int, cfg, gpu_ids, dataset, normalizer_state, data_description):
    world_size = len(gpu_ids)
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    try:
        # Use the same seed for model initialization on every rank.
        set_seed(cfg.training.seed)

        resume_from = cfg.get("resume_from")
        resuming = resume_from is not None
        checkpoint = load_training_checkpoint(resume_from) if resuming else None
        normalizer = None
        if normalizer_state is not None:
            normalizer = AffineNormalizer()
            normalizer.load_state_dict(normalizer_state)

        components = build_training_components(
            cfg,
            device,
            dataset,
            normalizer,
            data_description=data_description,
            rank=rank,
            world_size=world_size,
            checkpoint=checkpoint,
        )
        cfg = components.effective_config
        with closing(components):
            resume_state = None
            if resuming:
                resume_state = restore_training_state(checkpoint, components, effective_config=cfg)

            del checkpoint
            if rank == 0:
                components.workspace.save_hydra_config(cfg)

            # Compile restored backbones before DDP wrapping.
            compile_training_models(cfg, components)

            components.model = DDP(
                components.model,
                device_ids=[gpu_id],
                output_device=gpu_id,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

            if not resuming and components.ema_model is not None:
                # Only fresh EMA follows the raw state synchronized by DDP.
                components.ema_model.load_state_dict(components.model.module.state_dict(), strict=True)

            trainer = build_trainer(components)
            trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))

    finally:
        close_resources(dist.destroy_process_group)


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    cfg, checkpoint = prepare_training(cfg)

    num_gpus = cfg.training.get("num_gpus")
    gpu_ids = resolve_gpu_ids(num_gpus, cfg.training.get("gpu_ids"), torch.cuda.device_count())
    print(f"Using GPUs: {gpu_ids}")

    os.environ.setdefault("MASTER_ADDR", "localhost")

    set_seed(cfg.training.seed)
    dataset, normalizer, data_description = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
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
        train_distributed,
        args=(cfg, gpu_ids, dataset, normalizer_state, data_description),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
