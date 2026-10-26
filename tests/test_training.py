import copy
import os
import subprocess
import sys
from contextlib import closing

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.checkpoint import load_checkpoint, restore_policy_agent
from dexmani_policy.common.config import PROJECT_ROOT
from dexmani_policy.common.experiments import discover_milestone_checkpoints, milestone_steps
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.data import build_dataset_and_normalizer
from dexmani_policy.training.optimization import build_optimizer, get_scheduler
from dexmani_policy.training.trainer import PolicyTrainer


class LinearPolicy(BasePolicy):
    def __init__(self, horizon, n_obs_steps, n_action_steps, action_dim):
        encoder = nn.Linear(19, action_dim, bias=False)
        encoder.consumed_observation_fields = ("joint_state",)
        super().__init__(encoder, nn.Identity(), horizon, n_obs_steps, n_action_steps, action_dim)

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay):
        return [{"params": list(self.obs_encoder.parameters()), "lr": obs_lr, "weight_decay": obs_weight_decay}]

    def compute_loss(self, batch, *, log_metrics=True):
        prediction = self.obs_encoder(batch["obs"]["joint_state"][:, 0])
        loss = (prediction - batch["action"][:, 0]).square().mean()
        return loss, {"loss": loss.detach()} if log_metrics else {}

    def predict_action(self, obs):
        return {"future_actions": self.obs_encoder(obs["joint_state"])}


def training_config(path, output):
    return OmegaConf.create({
        "task_name": "test", "action_key": "action", "horizon": 1,
        "n_obs_steps": 1, "n_action_steps": 1, "action_dim": 19,
        "normalization": {"joint_state": "limits", "action": "limits"},
        "dataset": {
            "_target_": "dexmani_policy.datasets.robot_dataset.RobotDataset",
            "zarr_path": str(path), "sensor_modalities": ["joint_state"],
            "horizon": 1, "obs_horizon": 1, "max_train_episodes": 1,
        },
        "agent": {
            "_target_": f"{__name__}.LinearPolicy", "horizon": 1,
            "n_obs_steps": 1, "n_action_steps": 1, "action_dim": 19,
        },
        "optimizer": {"lr": 0.01, "weight_decay": 0.0},
        "ema": {"_target_": "dexmani_policy.training.optimization.EMA"},
        "dataloader": {"batch_size": 2, "num_workers": 0, "shuffle": False, "drop_last": True},
        "training": {
            "seed": 42, "device": "cpu", "use_ema": True, "max_grad_norm": 0.0,
            "lr_scheduler": "cosine", "lr_warmup_steps": 0,
            "loop": {"total_train_steps": 6, "log_interval_steps": 1, "gradient_accumulation_steps": 4},
        },
        "workspace": {"output_dir": str(output), "wandb_cfg": None},
    })


def test_accumulation_matches_full_batches_including_tail(demonstrations, tmp_path, one_thread):
    cfg = training_config(demonstrations, tmp_path / "run")
    dataset, normalizer, description = build_dataset_and_normalizer(cfg)
    with closing(PolicyTrainer(cfg, dataset, normalizer, description)) as trainer:
        reference = copy.deepcopy(trainer.model)
        optimizer = build_optimizer(reference, **cfg.optimizer)
        scheduler = get_scheduler(optimizer, "cosine", 0, 6)
        for batch in DataLoader(dataset, batch_size=8):
            loss, _ = reference.compute_loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        trainer.train(max_updates=2)
        torch.testing.assert_close(trainer.model.state_dict(), reference.state_dict())
        assert (trainer.global_step, trainer.current_epoch) == (2, 1)
        assert trainer.ema_updater.optimization_step == 2
        assert trainer.scheduler.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize("updates", [1, 2])
@pytest.mark.parametrize("fixed_decay", [None, 0.9])
def test_epoch_resume_and_checkpoint_selection(demonstrations, tmp_path, one_thread, updates, fixed_decay):
    cfg = training_config(demonstrations, tmp_path / "source")
    cfg.ema.fixed_decay = fixed_decay
    set_seed(42)
    dataset, normalizer, description = build_dataset_and_normalizer(cfg)
    with closing(PolicyTrainer(cfg, dataset, normalizer, description)) as trainer:
        trainer.train(max_updates=updates)
        path = (trainer.checkpoint_dir / "latest.pt").resolve()
        saved_epoch = trainer.current_epoch
        checkpoint = load_checkpoint(path)
        obs = next(iter(DataLoader(dataset, batch_size=2)))["obs"]
        for use_ema, model in ((False, trainer.model), (True, trainer.ema_model)):
            restored = restore_policy_agent(trainer.cfg, path, use_ema=use_ema, device="cpu")
            torch.testing.assert_close(restored.predict_action(obs), model.predict_action(obs))

    cfg = OmegaConf.create(checkpoint.effective_config)
    cfg.resume_from = str(path)
    cfg.workspace.output_dir = str(tmp_path / "resumed")
    dataset, normalizer, description = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
    assert normalizer is None
    with closing(PolicyTrainer(cfg, dataset, normalizer, description, checkpoint=checkpoint)) as trainer:
        assert (trainer.global_step, trainer.current_epoch) == (updates, saved_epoch)
        assert trainer.scheduler.state_dict() == checkpoint.scheduler_state
        assert trainer.ema_updater.optimization_step == checkpoint.ema_updater_step
        state = trainer.optimizer.state_dict()
        assert state["param_groups"] == checkpoint.optimizer_state["param_groups"]
        torch.testing.assert_close(state["state"], checkpoint.optimizer_state["state"])
        assert torch.equal(torch.get_rng_state(), checkpoint.rng_state["torch"])
        np.testing.assert_array_equal(np.random.get_state()[1], checkpoint.rng_state["numpy"][1])
        seen = []
        handle = trainer.model.register_forward_pre_hook(
            lambda module, args: seen.extend(args[0]["action"][:, 0, 0].tolist())
        )
        trainer.train()
        handle.remove()
        expected = [float(dataset[i]["action"][0, 0]) for i in range(8)]
        assert seen[:8] == expected
        assert trainer.global_step == trainer.ema_updater.optimization_step == 6
        candidates = discover_milestone_checkpoints(trainer.logger.output_dir)
        assert {item.global_step for item in candidates} == set(milestone_steps(6))


def test_nonfinite_loss_and_signal_stop_at_update(demonstrations, tmp_path, one_thread):
    cfg = training_config(demonstrations, tmp_path / "stop")
    dataset, normalizer, description = build_dataset_and_normalizer(cfg)
    with closing(PolicyTrainer(cfg, dataset, normalizer, description)) as trainer:
        def request_stop(module, inputs):
            trainer.stop_requested = True
        handle = trainer.model.register_forward_pre_hook(request_stop)
        trainer.train()
        handle.remove()
        assert trainer.global_step == 1 and trainer.current_epoch == 0
        assert load_checkpoint(trainer.checkpoint_dir / "latest.pt").global_step == 1
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        batch["action"].fill_(float("nan"))
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.train_microbatch(batch)
        assert trainer.global_step == 1


def test_dp3_cpu_cli_training_and_resume(demonstrations, tmp_path):
    source, resumed = tmp_path / "source", tmp_path / "resumed"
    command = [sys.executable, "-m", "dexmani_policy.train", "--config-name=dp3"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    overrides = [
        f"zarr_path={demonstrations}", f"hydra.run.dir={source}", "workspace.wandb_cfg=null",
        "horizon=4", "n_obs_steps=1", "n_action_steps=2", "dataset.pad_after=0",
        "dataset.augmentation_cfg=null", "agent.num_points=8", "agent.pc_out_dim=8",
        "agent.diffusion_step_embed_dim=16", "agent.down_dims=[8,16]", "agent.n_groups=4",
        "agent.num_training_steps=10", "agent.num_inference_steps=2",
        "training.device=cpu", "training.use_compile=false", "training.use_bfloat16=false",
        "training.lr_warmup_steps=0", "training.loop.total_train_steps=2", "+max_updates=1",
        "dataloader.batch_size=2", "dataloader.num_workers=0",
    ]
    first = subprocess.run(command + overrides, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    saved = load_checkpoint(source / "checkpoints/latest.pt")
    assert saved.global_step == 1
    second = subprocess.run(
        command + [f"+resume_from={source}", f"hydra.run.dir={resumed}"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    final = load_checkpoint(resumed / "checkpoints/latest.pt")
    assert final.global_step == final.ema_updater_step == 2
    assert any(not torch.equal(value, final.model_state[name]) for name, value in saved.model_state.items())
