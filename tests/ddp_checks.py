"""Two-rank CPU/Gloo integration, not a substitute for multi-GPU NCCL checks."""

from contextlib import closing
from datetime import timedelta
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_training_checkpoint
from dexmani_policy.common.runtime import set_seed
from dexmani_policy.training.builder import build_training_components, build_trainer
from dexmani_policy.training.data import build_dataset_and_normalizer
from dexmani_policy.training.resume import restore_training_state
from tests.helpers import tiny_policy
from tests.test_data_config import zarr_fixture


def worker(rank, root, config):
    root = Path(root)
    dist.init_process_group('gloo', init_method=f'file://{root / "rendezvous"}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        for stage in ('first', 'resumed'):
            set_seed(42)
            checkpoint = None
            if stage == 'first':
                cfg = OmegaConf.create(config)
            else:
                source = (root / 'first/checkpoints/latest.pt').resolve()
                checkpoint = load_training_checkpoint(source)
                cfg = OmegaConf.create(checkpoint.effective_config)
                cfg.resume_from = str(source)
            cfg.workspace.output_dir = str(root / stage)
            ds, norm, facts = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint, verbose=False)
            components = build_training_components(cfg, torch.device('cpu'), ds, norm,
                data_description=facts, rank=rank, world_size=2, checkpoint=checkpoint)
            with closing(components):
                cfg = components.effective_config
                recipes = [None, None]
                dist.all_gather_object(recipes, OmegaConf.to_container(cfg, resolve=True))
                assert recipes[0] == recipes[1]
                resume = None if checkpoint is None else restore_training_state(
                    checkpoint, components, effective_config=cfg)
                if rank == 0:
                    components.workspace.save_hydra_config(cfg)
                    if stage == 'resumed':
                        assert len(list((root / stage / 'checkpoints').glob('*milestone*'))) == 1
                components.model = DistributedDataParallel(components.model,
                    static_graph=True, gradient_as_bucket_view=True, find_unused_parameters=False)
                if checkpoint is None and components.ema_model is not None:
                    components.ema_model.load_state_dict(components.model.module.state_dict(), strict=True)
                trainer = build_trainer(components)
                assert len(components.train_loader) % 3 == 1
                trainer.train(resume_state=resume, max_updates=4 if stage == 'first' else 2)
                expected = 4 if stage == 'first' else 6
                assert trainer.global_step == components.scheduler.last_epoch == expected
                if components.ema_updater is not None:
                    assert components.ema_updater.optimization_step == expected
                for parameter in trainer.raw_model.parameters():
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
            if rank == 0 and stage == 'first':
                next((root / stage / 'checkpoints').glob('*milestone=20pct.pt')).unlink()
            dist.barrier()
        if rank == 0:
            saved = load_training_checkpoint(root / 'resumed/checkpoints/latest.pt')
            assert saved.global_step == 6 and len(saved.rng_states) == 2
            assert len(list((root / 'resumed/checkpoints').glob('*milestone*'))) == 2
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path, group, _ = zarr_fixture(tmp)
        group.create_dataset('data/point_cloud', data=np.random.default_rng(4).normal(
            size=(26, 8, 6)).astype(np.float32), chunks=(4, 8, 6))
        _, cfg, _, _ = tiny_policy()
        with open_dict(cfg):
            cfg.zarr_path = cfg.dataset.zarr_path = str(path)
            cfg.dataset.augmentation_cfg = None
            cfg.dataloader.update(batch_size=4, num_workers=0, pin_memory=False,
                                  persistent_workers=False, prefetch_factor=None, drop_last=False)
            cfg.training.update(use_compile=False, use_bfloat16=False, lr_warmup_steps=0)
            cfg.training.loop.update(total_train_steps=10, gradient_accumulation_steps=3)
            cfg.workspace.wandb_cfg = None
        mp.spawn(worker, args=(tmp, OmegaConf.to_container(cfg, resolve=True)), nprocs=2, join=True)
    print('CPU/Gloo two-rank fresh/resume/tail accumulation/missing-milestone archive PASSED')


if __name__ == '__main__':
    main()
