"""Scratch training -> strict resume -> verified deployment with real components."""

import copy

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from conftest import TASK_NAME, dataset_config, write_zarr
from test_flow_agents import small_config
from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.deployment.export import export_deployment_artifact
from dexmani_policy.deployment.runtime import load_experiment
from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
)
from dexmani_policy.training.resume import (
    build_resume_contract,
    build_train_loader,
    restore_training_state,
)
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("policy", ["sat", "maniflow"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_ema", [False, True])
def test_scratch_trainer_resume_and_deployment(policy, device, use_ema, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(22)
    cfg = load_config(policy)
    cfg.task_name = TASK_NAME
    cfg.agent = small_config(policy).agent
    cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps = 4, 2, 2
    cfg.agent.action_dim = cfg.agent.state_dim = 19
    cfg.agent.num_points = 1024
    if policy == "maniflow":
        cfg.agent.pc_encoder_config.num_points = 1024
    data_path = write_zarr(tmp_path / "data.zarr")
    zarr.open_group(str(data_path), mode="a").create_group("meta")["episode_ends"] = (
        np.array([8], dtype=np.int64)
    )
    cfg.dataset = OmegaConf.create(dataset_config(zarr_path=str(data_path)))
    cfg.normalization = {
        "joint_state": "limits",
        "point_cloud": "identity",
        "action": "limits",
    }
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.training.loop.total_train_steps = 4
    cfg.training.loop.gradient_accumulation_steps = 1
    cfg.training.lr_warmup_steps = 0
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset)
    batch = next(iter(loader))

    def build():
        model, ema, updater = build_model_and_ema(cfg, device, normalizer)
        optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, len(loader))
        contract = build_resume_contract(cfg, model, loader)
        trainer = Trainer(
            device=device,
            model=model,
            ema_model=ema,
            ema_updater=updater,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=loader,
            workspace=None,
            train_loop_cfg=TrainLoopConfig(**dict(cfg.training.loop)),
            num_training_steps=4,
            resume_contract=contract,
        )
        return trainer, contract

    trainer, contract = build()
    trainer.train_one_step(batch)
    checkpoint = TrainCheckpoint(
        epoch=0,
        global_step=1,
        next_micro_step=1,
        model_state=copy.deepcopy(trainer.model.state_dict()),
        ema_model_state=copy.deepcopy(trainer.ema_model.state_dict()),
        optimizer_state=copy.deepcopy(trainer.optimizer.state_dict()),
        scheduler_state=copy.deepcopy(trainer.scheduler.state_dict()),
        monitor={},
        resume_contract=contract,
        ema_updater_step=trainer.ema_updater.optimization_step,
        ema_decay=trainer.ema_updater.decay,
        rng_states=[get_rng_state()],
    )
    experiment = tmp_path / "experiment"
    store = CheckpointStore(experiment / "checkpoints")
    path = store.save("step1.pt", checkpoint)
    _, expected_logs = trainer.train_one_step(batch)
    resumed, resumed_contract = build()
    loaded = store.load(path)
    cursor = restore_training_state(
        loaded,
        resume_contract=resumed_contract,
        model=resumed.model,
        ema_model=resumed.ema_model,
        ema_updater=resumed.ema_updater,
        optimizer=resumed.optimizer,
        scheduler=resumed.scheduler,
        device=device,
    )
    assert cursor == (1, 0, 1)
    _, actual_logs = resumed.train_one_step(batch)
    assert_tree_equal(actual_logs["loss"], expected_logs["loss"])
    for name in ("model", "ema_model", "optimizer", "scheduler"):
        assert_tree_equal(
            getattr(trainer, name).state_dict(), getattr(resumed, name).state_dict()
        )
    assert (
        trainer.ema_updater.optimization_step == resumed.ema_updater.optimization_step
    )
    assert trainer.ema_updater.decay == resumed.ema_updater.decay

    cfg.eval.use_ema = use_ema
    cfg.eval.denoise_steps = 2
    OmegaConf.save(cfg, experiment / "config.yaml")
    export_deployment_artifact(experiment, checkpoint_selector="step1.pt")
    restored = load_experiment(experiment, device=device, inference_steps=1)
    observation = {key: value[0].numpy() for key, value in batch["obs"].items()}
    prediction = restored.predict(observation)
    assert np.isfinite(prediction).all()
    assert prediction.shape == (2, 19)
