"""CPU regressions for the resumable training state machine."""

from __future__ import annotations

import copy
import os
import random
import signal
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    TrainCheckpoint,
    validate_resume_contract,
)
from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state
from dexmani_policy.datasets.resumable_sampler import ResumableDistributedSampler
from dexmani_policy.training.ema_model import EMAModel
from dexmani_policy.training.resume import (
    build_train_loader,
    loader_options,
    restore_training_state,
    validate_gpu_ids,
)
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig


class _TinyDataset(Dataset):
    def __init__(self, length: int = 5):
        self.x = torch.arange(1, length + 1, dtype=torch.float32).unsqueeze(-1)
        self.y = 0.25 * self.x - 0.1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.callback = None
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        # Exercise every process RNG captured by get_rng_state().  The callback
        # requests a stop from inside a non-boundary micro-batch.
        scale = (
            torch.rand((), device=batch["x"].device)
            + random.random()
            + float(np.random.random())
        )
        if self.callback is not None:
            self.callback(self.forward_calls)
        prediction = self.linear(batch["x"]) * (0.5 + 0.05 * scale)
        loss = torch.square(prediction - batch["y"]).mean()
        return loss, {"loss": loss.detach()}


class _MetricModel(nn.Module):
    def __init__(self, *, reverse_metrics=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.reverse_metrics = reverse_metrics

    def forward(self, batch):
        value = batch["x"].mean()
        loss = torch.square(self.weight * value)
        items = [
            ("sample", value.detach()),
            ("double", 2 * value.detach()),
            ("loss", loss.detach()),
        ]
        if self.reverse_metrics:
            items.reverse()
        return loss, dict(items)


class _CheckpointWorkspace:
    """Small checkpoint-only workspace used by the training-loop regression."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_store = CheckpointStore(self.checkpoint_dir)
        self.logs = []

    def save_checkpoint(self, tag, checkpoint):
        return self.checkpoint_store.save(f"{tag}.pt", checkpoint)

    def save_latest(self, checkpoint_path):
        latest = self.checkpoint_dir / "latest.pt"
        temporary = self.checkpoint_dir / "latest.tmp.pt"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(checkpoint_path.name)
        os.replace(temporary, latest)
        return latest

    def load_checkpoint(self, tag_or_path):
        return self.checkpoint_store.load(self.checkpoint_store.resolve_path(tag_or_path))

    def log(self, data, step=None):
        self.logs.append((step, dict(data)))

    def close(self):
        pass


def _loader_config(*, seed=17, batch_size=1, shuffle=False, drop_last=False):
    return OmegaConf.create(
        {
            "training": {"seed": seed},
            "dataloader": {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": 0,
                # These are invalid DataLoader arguments with zero workers and
                # must be normalized by the shared builder.
                "persistent_workers": True,
                "prefetch_factor": 2,
                "drop_last": drop_last,
            },
        }
    )


def _resume_contract(batches_per_epoch, *, world_size=1, accumulation=2):
    return {
        "batches_per_epoch": batches_per_epoch,
        "world_size": world_size,
        "training": {"loop": {"gradient_accumulation_steps": accumulation}},
    }


def _make_trainer(
    output_dir,
    *,
    total_steps=4,
    world_size=1,
    rank=0,
    distributed=False,
    use_ema=True,
    model=None,
    log_interval_steps=1000,
    accumulation=2,
):
    dataset = _TinyDataset(5 if world_size == 1 else 8)
    cfg = _loader_config(seed=31, batch_size=1)
    loader = build_train_loader(cfg, dataset, rank=rank, world_size=world_size)
    model = _TinyModel() if model is None else model
    ema_model = copy.deepcopy(model) if use_ema else None
    ema_updater = EMAModel(ema_model, inv_gamma=2.0, power=0.75) if use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    workspace = _CheckpointWorkspace(output_dir) if rank == 0 else None
    contract = _resume_contract(
        len(loader), world_size=world_size, accumulation=accumulation
    )
    wrapped_model = (
        DistributedDataParallel(
            model,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if distributed
        else model
    )
    trainer = Trainer(
        device=torch.device("cpu"),
        model=wrapped_model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        workspace=workspace,
        train_loop_cfg=TrainLoopConfig(
            total_train_steps=total_steps,
            log_interval_steps=log_interval_steps,
            gradient_accumulation_steps=accumulation,
        ),
        use_ema_teacher_for_consistency=False,
        num_training_steps=total_steps,
        resume_contract=contract,
        max_grad_norm=1.0,
        is_main_process=rank == 0,
        distributed=distributed,
        train_sampler=loader.sampler,
    )
    return trainer


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _sample_all_rngs():
    return random.random(), float(np.random.random()), torch.rand(4)


def test_recursive_resume_contract_reports_every_difference():
    saved = {
        "nested": {"changed": 1, "old": 2},
        "sequence": [{"typed": 1}, "same"],
        "obsolete": True,
    }
    current = {
        "nested": {"changed": 3, "new": 4},
        "sequence": [{"typed": False}, "same", "added"],
    }

    with pytest.raises(ValueError) as exc_info:
        validate_resume_contract(saved, current)

    message = str(exc_info.value)
    expected_lines = {
        "resume_contract.nested.changed: saved=1, current=3",
        "resume_contract.nested.new: missing in checkpoint",
        "resume_contract.nested.old: unexpected checkpoint key",
        "resume_contract.obsolete: unexpected checkpoint key",
        "resume_contract.sequence: length saved=2, current=3",
        "resume_contract.sequence[0].typed: saved=1, current=False",
    }
    assert expected_lines <= set(message.splitlines())


def test_sampler_single_and_multi_rank_resume_preserves_seeded_order():
    dataset = list(range(13))
    kwargs = dict(
        batch_size=2,
        num_replicas=2,
        shuffle=True,
        seed=91,
        drop_last=False,
    )
    rank0 = ResumableDistributedSampler(dataset, rank=0, **kwargs)
    rank1 = ResumableDistributedSampler(dataset, rank=1, **kwargs)
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    full0, full1 = list(rank0), list(rank1)

    # DistributedSampler shards alternating positions of one shared epoch order.
    interleaved = [item for pair in zip(full0, full1) for item in pair]
    assert len(interleaved) == 14
    assert sorted(interleaved[:13]) == list(range(13))

    rank0.set_epoch(3, next_micro_step=2)
    rank1.set_epoch(3, next_micro_step=2)
    assert list(rank0) == full0[4:]
    assert list(rank1) == full1[4:]

    single = ResumableDistributedSampler(
        dataset,
        batch_size=2,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=91,
        drop_last=False,
    )
    single.set_epoch(3)
    single_full = list(single)
    single.set_epoch(3, next_micro_step=3)
    assert list(single) == single_full[6:]


def test_sampler_tracks_full_epoch_and_remaining_tail_with_drop_last():
    dataset = list(range(13))
    sampler = ResumableDistributedSampler(
        dataset,
        batch_size=2,
        num_replicas=2,
        rank=0,
        shuffle=False,
        seed=0,
        drop_last=False,
    )
    cfg_keep = _loader_config(batch_size=2, drop_last=False)
    cfg_drop = _loader_config(batch_size=2, drop_last=True)
    loader_keep = build_train_loader(cfg_keep, dataset, rank=0, world_size=2)
    loader_drop = build_train_loader(cfg_drop, dataset, rank=0, world_size=2)

    assert sampler.num_samples == 7
    assert sampler.full_num_batches(drop_last=False) == 4
    assert sampler.full_num_batches(drop_last=True) == 3
    assert len(loader_keep) == 4
    assert len(loader_drop) == 3

    loader_keep.sampler.set_epoch(0, next_micro_step=3)
    loader_drop.sampler.set_epoch(0, next_micro_step=3)
    assert len(loader_keep.sampler) == 1
    assert len(loader_keep) == 1
    assert len(loader_drop) == 0


def test_zero_worker_loader_uses_independent_generator_and_normalized_options():
    cfg = _loader_config(seed=23, batch_size=2, shuffle=True)
    assert loader_options(cfg) == {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
    }

    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    loader = build_train_loader(cfg, list(range(6)))
    iterator = iter(loader)
    next(iterator)
    assert torch.equal(torch.get_rng_state(), before)


def test_nonboundary_signal_checkpoint_resume_matches_continuous_training(tmp_path):
    # Continuous reference.
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    continuous = _make_trainer(tmp_path / "continuous")
    continuous.train()
    continuous_rng = get_rng_state()

    # Request interruption on the first micro-batch; accumulation=2 means the
    # checkpoint must be delayed until the complete optimizer boundary.
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    interrupted = _make_trainer(tmp_path / "split")
    interrupted.raw_model.callback = lambda call: (
        interrupted._signal_handler(signal.SIGTERM, None) if call == 1 else None
    )
    interrupted.train()

    checkpoint = interrupted.workspace.load_checkpoint("latest")
    assert checkpoint.global_step == 1
    assert checkpoint.epoch == 0
    assert checkpoint.next_micro_step == 2
    assert checkpoint.ema_updater_step == 1
    assert checkpoint.scheduler_state["last_epoch"] == 1

    # Construction may consume RNG, so load_for_resume must restore the state
    # captured after the completed accumulation group.
    resumed = _make_trainer(tmp_path / "split")
    resumed.train(resume_tag="latest")
    resumed_rng = get_rng_state()

    _assert_nested_equal(continuous.raw_model.state_dict(), resumed.raw_model.state_dict())
    _assert_nested_equal(continuous.ema_model.state_dict(), resumed.ema_model.state_dict())
    _assert_nested_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_nested_equal(continuous.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert continuous.ema_updater.optimization_step == resumed.ema_updater.optimization_step == 4
    assert continuous.ema_updater.decay == resumed.ema_updater.decay
    assert (continuous.global_step, continuous.current_epoch, continuous.next_micro_step) == (
        resumed.global_step,
        resumed.current_epoch,
        resumed.next_micro_step,
    ) == (4, 1, 2)
    _assert_nested_equal(continuous_rng, resumed_rng)


def test_metrics_mean_each_accumulation_group_and_reset_after_tail(tmp_path):
    trainer = _make_trainer(
        tmp_path / "metrics",
        total_steps=4,
        use_ema=False,
        model=_MetricModel(),
        log_interval_steps=1,
    )
    trainer.train()

    assert [step for step, _ in trainer.workspace.logs] == [1, 2, 3, 4]
    assert [metrics["train/sample"] for _, metrics in trainer.workspace.logs] == [
        1.5,
        3.5,
        5.0,
        1.5,
    ]
    assert [metrics["train/double"] for _, metrics in trainer.workspace.logs] == [
        3.0,
        7.0,
        10.0,
        3.0,
    ]


def test_restore_selects_rank_ordered_rng_state():
    torch.manual_seed(4)
    source_model = _TinyModel()
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, 1)
    contract = _resume_contract(5, world_size=2, accumulation=2)

    states = []
    expected = []
    for seed in (101, 202):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        states.append(get_rng_state())
        expected.append(_sample_all_rngs())

    checkpoint = TrainCheckpoint(
        epoch=0,
        global_step=1,
        next_micro_step=2,
        model_state=copy.deepcopy(source_model.state_dict()),
        ema_model_state=None,
        optimizer_state=copy.deepcopy(source_optimizer.state_dict()),
        scheduler_state=copy.deepcopy(source_scheduler.state_dict()),
        monitor={},
        resume_contract=contract,
        ema_updater_step=None,
        ema_decay=None,
        rng_states=states,
    )

    for rank in (0, 1):
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        result = restore_training_state(
            checkpoint,
            resume_contract=contract,
            model=model,
            ema_model=None,
            ema_updater=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            rank=rank,
        )
        assert result == (1, 0, 2)
        actual = _sample_all_rngs()
        assert actual[:2] == expected[rank][:2]
        torch.testing.assert_close(actual[2], expected[rank][2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_gpus,gpu_ids,available",
    [
        (True, None, 4),
        (1, None, 4),
        (2, [0], 4),
        (2, [0, 0], 4),
        (2, [-1, 0], 4),
        (2, [0, 4], 4),
        (2, [0, 1.0], 4),
        (2, [0, True], 4),
        (3, None, 2),
    ],
)
def test_validate_gpu_ids_rejects_invalid_configuration(num_gpus, gpu_ids, available):
    with pytest.raises(ValueError):
        validate_gpu_ids(num_gpus, gpu_ids, available)


def test_validate_gpu_ids_returns_explicit_or_default_mapping():
    assert validate_gpu_ids(2, [3, 1], 4) == [3, 1]
    assert validate_gpu_ids(3, None, 4) == [0, 1, 2]


def _gloo_interrupt_worker(rank, world_size, init_path, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    original_all_reduce = dist.all_reduce
    metric_collectives = [0]

    def counted_all_reduce(tensor, *args, **kwargs):
        if tensor.dtype == torch.float64 and tensor.numel() > 1:
            metric_collectives[0] += 1
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = counted_all_reduce
    try:
        random.seed(500 + rank)
        np.random.seed(500 + rank)
        torch.manual_seed(500 + rank)
        trainer = _make_trainer(
            output_dir,
            total_steps=2,
            world_size=world_size,
            rank=rank,
            distributed=True,
            use_ema=False,
        )
        if rank == 1:
            trainer.raw_model.callback = lambda call: (
                trainer._signal_handler(signal.SIGTERM, None) if call == 1 else None
            )
        trainer.train()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            metric_collectives[0], Path(output_dir) / f"metric_collectives_rank{rank}.pt"
        )
    finally:
        dist.all_reduce = original_all_reduce
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_gloo_signal_stops_symmetrically_and_gathers_rng(tmp_path):
    world_size = 2
    init_path = tmp_path / "gloo_init"
    output_dir = tmp_path / "gloo_run"
    mp.spawn(
        _gloo_interrupt_worker,
        args=(world_size, str(init_path), str(output_dir)),
        nprocs=world_size,
        join=True,
    )

    checkpoint = CheckpointStore(output_dir / "checkpoints").load(
        output_dir / "checkpoints" / "latest.pt"
    )
    assert checkpoint.global_step == 1
    assert checkpoint.epoch == 0
    assert checkpoint.next_micro_step == 2
    assert len(checkpoint.rng_states) == world_size
    assert not torch.equal(
        checkpoint.rng_states[0]["torch"], checkpoint.rng_states[1]["torch"]
    )
    assert [
        torch.load(
            output_dir / f"metric_collectives_rank{rank}.pt",
            weights_only=False,
        )
        for rank in range(world_size)
    ] == [0, 0]


def _gloo_metrics_worker(rank, world_size, init_path, output_dir, accumulation):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    original_all_reduce = dist.all_reduce
    metric_collectives = [0]

    def counted_all_reduce(tensor, *args, **kwargs):
        if tensor.dtype == torch.float64 and tensor.numel() > 1:
            metric_collectives[0] += 1
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = counted_all_reduce
    try:
        random.seed(700 + rank)
        np.random.seed(700 + rank)
        torch.manual_seed(700 + rank)
        trainer = _make_trainer(
            output_dir,
            total_steps=2,
            world_size=world_size,
            rank=rank,
            distributed=True,
            use_ema=False,
            model=_MetricModel(reverse_metrics=rank == 1),
            log_interval_steps=1,
            accumulation=accumulation,
        )
        # SGD without clipping exposes gradient scaling errors directly.
        trainer.max_grad_norm = 0
        trainer.optimizer = torch.optim.SGD(trainer.raw_model.parameters(), lr=0.001)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1, gamma=0.8)
        trainer.train()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            metric_collectives[0], Path(output_dir) / f"metric_collectives_rank{rank}.pt"
        )
        if rank == 0:
            torch.save(
                {
                    "logs": trainer.workspace.logs,
                    "model": trainer.raw_model.state_dict(),
                },
                Path(output_dir) / "rank0_result.pt",
            )
    finally:
        dist.all_reduce = original_all_reduce
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize("accumulation", [2, 3])
def test_two_rank_metrics_use_stable_keys_and_global_mean(tmp_path, accumulation):
    world_size = 2
    init_path = tmp_path / "metrics_gloo_init"
    output_dir = tmp_path / "metrics_gloo_run"
    mp.spawn(
        _gloo_metrics_worker,
        args=(world_size, str(init_path), str(output_dir), accumulation),
        nprocs=world_size,
        join=True,
    )

    result = torch.load(output_dir / "rank0_result.pt", weights_only=False)
    logs = result["logs"]
    assert [step for step, _ in logs] == [1, 2]
    means = [2.5, 6.5] if accumulation == 2 else [3.5, 7.5]
    assert [metrics["train/sample"] for _, metrics in logs] == means
    assert [metrics["train/double"] for _, metrics in logs] == [2 * x for x in means]
    assert [
        torch.load(
            output_dir / f"metric_collectives_rank{rank}.pt",
            weights_only=False,
        )
        for rank in range(world_size)
    ] == [2, 2]

    # Two DDP accumulation groups must match the equivalent global batches.
    reference = _MetricModel()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    split = 2 * accumulation
    for values in (list(range(1, split + 1)), list(range(split + 1, 9))):
        loss = torch.square(reference.weight * torch.tensor(values)).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(
        result["model"]["weight"], reference.state_dict()["weight"], rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize('legacy_format', ['simple.v1', 'simple.v2'])
def test_checkpoint_store_rejects_legacy_training_schema(tmp_path, legacy_format):
    path = tmp_path / 'legacy.pt'
    torch.save({'state': {}, 'weights': {}, '_format': legacy_format, '_saved_at': 0}, path)
    with pytest.raises(RuntimeError, match='Unsupported checkpoint format'):
        CheckpointStore(tmp_path).load(path)


def test_actual_contract_allows_runtime_changes_but_rejects_sampling_changes():
    from types import SimpleNamespace
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.resume import build_resume_contract

    cfg = load_config('dp3')
    model = SimpleNamespace(n_obs_steps=2, n_action_steps=8, horizon=16,
                            action_key='action', action_dim=19, control_action_dim=19)
    dataset = list(range(257))
    contract = build_resume_contract(cfg, model, build_train_loader(cfg, dataset))
    cfg.training.device = 'cuda:7'
    cfg.training.loop.log_interval_steps = 17
    cfg.dataloader.num_workers = 0
    runtime_contract = build_resume_contract(cfg, model, build_train_loader(cfg, dataset))
    validate_resume_contract(contract, runtime_contract)

    cfg.dataloader.batch_size = 64
    changed = build_resume_contract(cfg, model, build_train_loader(cfg, dataset))
    with pytest.raises(ValueError) as error:
        validate_resume_contract(contract, changed)
    assert 'resume_contract.loader.batch_size' in str(error.value)
    assert 'resume_contract.batches_per_epoch' in str(error.value)
