import copy
import json
import math
import shutil
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.common.checkpoint import load_checkpoint, save_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.env_runner.dexjoco_runner import DexJoCoRunner
from dexmani_policy.training import resume, trainer as policy_training
from dexmani_policy.training.optimization import clip_gradients
from dexmani_policy.training.pretrain.obs import common as obs_training
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer, run_trainer

from tests.helpers import MeanDecoder, TinyPolicy, training_config


class CustomDexJoCoRunner(DexJoCoRunner):
    pass


class BatchNormPolicy(TinyPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        linear = self.obs_encoder.linear
        self.obs_encoder.linear = nn.Sequential(linear, nn.BatchNorm1d(linear.out_features))


def test_dexjoco_subclass_requires_observation_metadata(tmp_path, small_policy_config):
    cfg = small_policy_config(tmp_path / "run", f"env_runner._target_={__name__}.CustomDexJoCoRunner")
    with pytest.raises(ValueError, match="Missing DexJoCo observation processing metadata.*joint_state"):
        build_trainer(cfg)


@pytest.mark.parametrize("decoder_type", [RectifiedFlow, ConsistencyFlow])
def test_flow_decoder_ema_training_and_compile_routing(decoder_type):
    model = Policy(nn.Identity(), decoder_type(nn.Identity()), 4, 1, 4, 3)
    ema = copy.deepcopy(model)
    requires_ema = decoder_type is ConsistencyFlow
    assert model.requires_ema_for_loss == requires_ema
    batch = {"action": torch.zeros(2, 4, 3)}
    supplied, options = next(model.training_microbatches([batch], ema if requires_ema else None))
    assert supplied is batch
    assert options == ({"ema_decoder": ema.action_decoder} if requires_ema else {})
    trainer = policy_training.PolicyTrainer.__new__(policy_training.PolicyTrainer)
    trainer.model, trainer.ema_model = model, ema
    trainer.cfg = OmegaConf.create({"training": {"use_compile": True, "compile_mode": "default"}})
    with patch.object(model, "compile_backbone") as student, patch.object(ema, "compile_backbone") as teacher:
        trainer.compile_models()
    student.assert_called_once_with(mode="default")
    assert teacher.call_count == int(requires_ema)


def test_batch_norm_ema_fine_tuning_and_resume(tmp_path, small_policy_config):
    cfg = small_policy_config(tmp_path, f"agent._target_={__name__}.BatchNormPolicy")
    with closing(build_trainer(cfg)) as trainer:
        bn = trainer.model.obs_encoder.linear[1]
        ema_bn = trainer.ema_model.obs_encoder.linear[1]
        initial_weight = bn.weight.detach().clone()
        trainer.train(max_updates=2)
        assert not torch.equal(bn.weight, initial_weight)
        assert bn.num_batches_tracked.item() == 2
        torch.testing.assert_close(ema_bn.state_dict(), bn.state_dict(), rtol=0, atol=0)

    saved = load_checkpoint(tmp_path / "checkpoints/latest.pt")
    with closing(prepare_resume(tmp_path)) as trainer:
        linear, bn = trainer.model.obs_encoder.linear
        ema_linear, ema_bn = trainer.ema_model.obs_encoder.linear
        torch.testing.assert_close(trainer.ema_model.state_dict(), saved.ema_model_state)
        previous = ema_linear.weight.detach().clone()
        decay = trainer.ema_updater.get_decay(trainer.global_step)
        assert 0 < decay < 1
        trainer.train(max_updates=1)

        assert bn.num_batches_tracked.item() == 3
        assert not ema_bn.training
        torch.testing.assert_close(ema_bn.state_dict(), bn.state_dict(), rtol=0, atol=0)
        assert not torch.equal(ema_linear.weight, linear.weight)
        torch.testing.assert_close(ema_linear.weight, previous.lerp(linear.weight, 1 - decay))


def test_resume_and_warm_start_state_boundaries(tmp_path, small_policy_config, make_demo, monkeypatch):
    directory = tmp_path / "source"
    data = make_demo("source", episode_ends=(4, 8, 12))
    first_source, resumed_source = "gpu:/snapshots/first", "gpu:/snapshots/resumed"
    monkeypatch.setenv("DEXMANI_POLICY_REMOTE_SOURCE", first_source)
    with closing(build_trainer(small_policy_config(directory, f"zarr_path={data}"))) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    first_checkpoint = latest.resolve()
    saved = load_checkpoint(first_checkpoint)
    assert saved.effective_config["remote_source"] == first_source
    for override in ("action_key=action_ee", "agent._target_=dexmani_policy.agents.policies.base.Policy"):
        invalid = small_policy_config(tmp_path / "invalid", f"initialization.policy={first_checkpoint}", override)
        with pytest.raises(ValueError, match="same policy and action semantics"):
            build_trainer(invalid)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint restoration must not fit normalization")

    monkeypatch.setattr(LinearNormalizer, "fit_field", forbidden)
    monkeypatch.setattr("dexmani_policy.training.initialization.resolve_rgb_architecture", forbidden)
    monkeypatch.setenv("DEXMANI_POLICY_REMOTE_SOURCE", resumed_source)
    with closing(prepare_resume(directory)) as resumed:
        torch.testing.assert_close(resumed.model.state_dict(), saved.model_state)
        torch.testing.assert_close(resumed.ema_model.state_dict(), saved.ema_model_state)
        optimizer = resumed.optimizer.state_dict()
        torch.testing.assert_close(optimizer["state"], saved.optimizer_state["state"])
        assert optimizer["param_groups"] == saved.optimizer_state["param_groups"]
        assert resumed.scheduler.state_dict() == saved.scheduler_state
        assert resumed.global_step == 1
        actual_parameters = {id(p) for p in resumed.model.parameters() if p.requires_grad}
        assert {id(p) for group in resumed.optimizer.param_groups for p in group["params"]} == actual_parameters
        assert {id(p) for p in resumed.ema_updater.parameters} == actual_parameters
        resumed.train(max_updates=1)
    continued = load_checkpoint(latest)
    assert continued.global_step == continued.scheduler_state["last_epoch"] == 2
    for group, base_lr in zip(continued.optimizer_state["param_groups"], saved.scheduler_state["base_lrs"]):
        assert group["lr"] == pytest.approx(0.25 * base_lr)
    assert continued.effective_config["remote_source"] == resumed_source
    assert continued.effective_config["training_data"] == saved.effective_config["training_data"]
    assert "prior_data" not in continued.effective_config
    assert OmegaConf.load(directory / "config.yaml").remote_source == first_source

    new_data = tmp_path / "copied.zarr"
    shutil.copytree(data, new_data)
    warm_cfg = small_policy_config(
        tmp_path / "warm", f"initialization.policy={first_checkpoint}",
        f"zarr_path={data}", "dataset.max_train_episodes=1",
    )
    with pytest.raises(ValueError, match="same-source prior exceeds"):
        build_trainer(warm_cfg)
    warm_cfg.zarr_path = warm_cfg.dataset.zarr_path = str(new_data)
    with closing(build_trainer(warm_cfg)) as warm:
        torch.testing.assert_close(warm.model.state_dict(), saved.model_state)
        assert warm.global_step == 0
        assert warm.scheduler.last_epoch == 0
        assert warm.optimizer.state_dict()["state"] == {}
        assert warm.logger.output_dir == tmp_path / "warm"
        assert warm.cfg.training_data.source == str(new_data)
        assert len(warm.cfg.training_data.train_source_episodes) == 1
        prior, = warm.cfg.prior_data
        assert prior.source == saved.effective_config["training_data"]["source"]
        assert prior.train_source_episodes == [0, 1, 2]
        assert not prior.budget_verified
        torch.testing.assert_close(warm.ema_model.state_dict(), warm.model.state_dict())
        warm.train(max_updates=1)

    warm_checkpoint = tmp_path / "warm/checkpoints/latest.pt"
    warm_saved = load_checkpoint(warm_checkpoint)
    repeated_cfg = OmegaConf.create(warm_saved.effective_config)
    repeated_cfg.workspace.output_dir = str(tmp_path / "warm_again")
    repeated_cfg.initialization.policy = str(warm_checkpoint)
    with closing(build_trainer(repeated_cfg)) as repeated:
        assert repeated.cfg.training_data == warm_saved.effective_config["training_data"]
        prior, = repeated.cfg.prior_data
        assert prior == {
            **warm_saved.effective_config["training_data"],
            "artifact": str(warm_checkpoint.resolve()), "budget_verified": False,
        }

    resume_cfg = OmegaConf.create(warm_saved.effective_config)
    alias = tmp_path / "alias.zarr"
    alias.symlink_to(new_data)
    resume_cfg.dataset.zarr_path = str(alias)
    del resume_cfg.training_data
    resume_cfg.prior_data[0].budget_verified = True
    resume_cfg.control_dt_s = 0.1
    with closing(build_trainer(resume_cfg, checkpoint=warm_saved)) as resumed:
        assert resumed.cfg.zarr_path == resumed.cfg.dataset.zarr_path == str(new_data)
        assert resumed.cfg.control_dt_s == warm_saved.effective_config["control_dt_s"]
        assert resumed.cfg.training_data == warm_saved.effective_config["training_data"]
        assert resumed.cfg.prior_data == warm_saved.effective_config["prior_data"]


class Samples(Dataset):
    sensor_modalities = ("target",)

    def __init__(self, targets, episode_store):
        self.targets = targets
        self.obs_horizon = 1
        self.episode_store = episode_store
        self.train_mask, self.val_mask = np.array([True]), np.array([False])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        value = self.targets[index].reshape(1, 1)
        return {"obs": {"target": value}, "action": value}

    def get_validation_dataset(self):
        return None

    def training_summary(self):
        return "deterministic scalar regression"


class ObservationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MeanDecoder()
        self.normalizer = LinearNormalizer()

    def forward(self, batch):
        return self.encoder.compute_loss(None, batch["obs"]["target"])


@pytest.mark.parametrize("count,batch_size,accumulation", [
    (4, 2, 2), (5, 4, 2), (9, 2, 3), (9, 4, 2),
    (1, 128, 4), (5, 2, 1), (129, 128, 2),
])
@pytest.mark.parametrize("loop", ["policy", "observation"])
def test_sample_weighted_updates(tmp_path, monkeypatch, make_demo, count, batch_size, accumulation, loop):
    targets = torch.arange(count).float() / count
    if count == 129:
        targets.zero_()
        targets[-1] = 1
    updates = math.ceil(math.ceil(count / batch_size) / accumulation)
    cfg = training_config(tmp_path, batch_size, accumulation, updates)
    dataset = Samples(targets, EpisodeStore(make_demo(), keys=()))
    actual_gradients, optimizers, schedulers = [], [], []

    def capture_clip(parameters, finite_loss, max_grad_norm=0, *, step):
        parameters = list(parameters)
        actual_gradients.append(parameters[0].grad.clone())
        return clip_gradients(parameters, finite_loss, max_grad_norm, step=step)

    def sgd(parameters, **kwargs):
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        optimizers.append(optimizer)
        return optimizer

    def scheduler(optimizer, *args, **kwargs):
        result = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        schedulers.append(result)
        return result

    if loop == "policy":
        model = Policy(nn.Identity(), MeanDecoder(), 1, 1, 1, 1)
        model.configure_normalizer({"action": 1}, {"action": (1,)})
        monkeypatch.setattr(policy_training, "build_optimizer", lambda model, **kw: sgd(model.parameters()))
        monkeypatch.setattr(policy_training, "build_scheduler", scheduler)
        monkeypatch.setattr(policy_training, "clip_gradients", capture_clip)
        trainer = policy_training.PolicyTrainer(cfg, model, dataset)
        ema_steps = []
        ema_step = trainer.ema_updater.step

        def update_ema(step):
            ema_steps.append(step)
            ema_step(step)

        monkeypatch.setattr(trainer.ema_updater, "step", update_ema)
        try:
            trainer.train()
            weight = model.action_decoder.weight.detach()
            assert trainer.global_step == updates
            assert ema_steps == list(range(updates))
        finally:
            trainer.close()
    else:
        monkeypatch.setattr(obs_training.hydra.utils, "instantiate", lambda *args, **kwargs: dataset)
        monkeypatch.setattr(obs_training.torch.optim, "AdamW", sgd)
        monkeypatch.setattr(obs_training, "build_scheduler", scheduler)
        monkeypatch.setattr(obs_training, "clip_gradients", capture_clip)
        obs_training.train(cfg, lambda dataset: ObservationModel())
        weight = optimizers[0].param_groups[0]["params"][0].detach()

    reference = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([reference], lr=0.1)
    expected_losses, expected_gradients, expected_norms = [], [], []
    for group in targets.split(batch_size * accumulation):
        optimizer.zero_grad()
        loss = (reference - group).square().mean()
        loss.backward()
        expected_losses.append(loss.detach())
        expected_gradients.append(reference.grad.clone())
        expected_norms.append(torch.nn.utils.clip_grad_norm_([reference], 0.1 if loop == "policy" else float("inf")))
        optimizer.step()
    torch.testing.assert_close(actual_gradients, expected_gradients)
    torch.testing.assert_close(weight, reference.detach())
    assert schedulers[0].last_epoch == updates
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == list(range(1, updates + 1))
    for record, loss, norm in zip(records, expected_losses, expected_norms):
        assert record["train/loss"] == pytest.approx(float(loss))
        assert record["train/lr"] == 0.1
        if loop == "policy":
            assert record["train/grad_norm"] == pytest.approx(float(norm))
    if count == 129:
        assert records[0]["train/loss"] == pytest.approx(1 / 129)
        assert float(actual_gradients[0]) == pytest.approx(-2 / 129)


def test_nonfinite_gradient_rejected():
    parameter = nn.Parameter(torch.zeros(()))
    parameter.grad = torch.tensor(float("inf"))
    with pytest.raises(FloatingPointError):
        clip_gradients([parameter], torch.tensor(True), step=0)
    parameter.grad.zero_()
    with pytest.raises(FloatingPointError):
        clip_gradients([parameter], torch.tensor(False), step=0)


def test_policy_artifact_paths(output_path, small_policy_config):
    value, expected = output_path
    cfg = small_policy_config(value)
    with closing(build_trainer(cfg)) as trainer:
        assert trainer.logger.output_dir == expected
        assert trainer.cfg.workspace.output_dir == str(expected)
        trainer.train(max_updates=1)
    saved_cfg = OmegaConf.load(expected / "config.yaml")
    saved = load_checkpoint(expected / "checkpoints/latest.pt")
    assert saved_cfg.workspace.output_dir == saved.effective_config["workspace"]["output_dir"] == str(expected)
    original_config = (expected / "config.yaml").read_bytes()
    original_metrics = (expected / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_trainer(cfg)
    assert (expected / "config.yaml").read_bytes() == original_config
    assert (expected / "metrics.jsonl").read_bytes() == original_metrics
    with closing(prepare_resume(expected)) as resumed:
        assert resumed.logger.output_dir == expected
        assert resumed.global_step == 1


@pytest.mark.parametrize("mode,benchmark", [(None, False), ("default", True)])
def test_training_options_and_resume(tmp_path, small_policy_config, monkeypatch, mode, benchmark):
    overrides = [] if mode is None else [f"training.compile_mode={mode}"]
    cfg = small_policy_config(
        tmp_path / "run", "training.use_compile=true",
        f"training.cudnn_benchmark={str(benchmark).lower()}", *overrides,
    )
    expected = "reduce-overhead" if mode is None else mode
    compiled = []
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = not benchmark
    torch.set_float32_matmul_precision("medium")
    with closing(build_trainer(cfg)) as trainer:
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark == benchmark
        assert torch.get_float32_matmul_precision() == "medium"
        monkeypatch.setattr(trainer.model, "compile_backbone", lambda *, mode: compiled.append(mode))
        trainer.compile_models()
        trainer.train(max_updates=1)
        assert torch.get_float32_matmul_precision() == "highest"
    with closing(prepare_resume(tmp_path / "run")) as resumed:
        assert resumed.cfg.training.compile_mode == expected
        assert resumed.cfg.training.cudnn_benchmark == benchmark
        monkeypatch.setattr(resumed.model, "compile_backbone", lambda *, mode: compiled.append(mode))
        resumed.compile_models()
    assert compiled == [expected, expected]


def test_training_failure_closes_trainer():
    closed = []

    def fail():
        raise ValueError("training failed")

    trainer = SimpleNamespace(
        compile_models=fail,
        close=lambda: closed.append(True),
    )
    with pytest.raises(ValueError, match="training failed"):
        run_trainer(trainer)
    assert closed == [True]


def test_training_preserves_failure_when_workers_and_logger_fail(tmp_path, small_policy_config, monkeypatch, caplog):
    trainer = build_trainer(small_policy_config(tmp_path / "run"))
    failure = ValueError("backward failed")
    monkeypatch.setattr(trainer, "_backward_group", Mock(side_effect=failure))
    workers = Mock(side_effect=OSError("worker cleanup failed"))
    monkeypatch.setattr(policy_training, "close_data_loader", workers)
    logger = Mock(side_effect=OSError("logger cleanup failed"))
    trainer.logger._resources.callback(logger)
    with pytest.raises(ValueError) as caught:
        run_trainer(trainer, max_updates=1)
    assert caught.value is failure
    workers.assert_called_once()
    logger.assert_called_once()
    assert "worker cleanup failed" in caplog.text and "logger cleanup failed" in caplog.text
    assert not trainer.model.training


def test_resume_uses_saved_zarr_and_groups(tmp_path, small_policy_checkpoint, make_demo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = load_checkpoint(small_policy_checkpoint)
    with pytest.raises(SystemExit) as rejected:
        resume.main(["--run", "policy_run", "--zarr-path", "other.zarr"])
    assert rejected.value.code == 2

    cfg = OmegaConf.create(saved.effective_config)
    other_data = make_demo("other")
    cfg.dataset.zarr_path = str(other_data)
    with pytest.raises(ValueError, match="Resume Zarr path mismatch"):
        build_trainer(cfg, checkpoint=saved)
    other_store = zarr.open_group(str(other_data), mode="a")
    other_store.attrs["source_zarr_path"] = saved.effective_config["training_data"]["source"]
    with pytest.raises(ValueError, match="Resume Zarr path mismatch"):
        build_trainer(cfg, checkpoint=saved)

    ends = zarr.open_group(saved.effective_config["dataset"]["zarr_path"], mode="a")["meta/episode_ends"]
    ends.resize(2)
    ends[:] = [6, 12]
    with pytest.raises(ValueError, match="Resume train source groups mismatch"):
        resume.main(["--run", "policy_run", "--max-updates", "1"])
    ends.resize(1)
    ends[:] = [12]
    assert load_checkpoint(small_policy_checkpoint).global_step == 1

    layout = saved.effective_config["episode_layout"]
    saved.effective_config["episode_layout"] = {**layout, "episode_ends": [11]}
    save_checkpoint(small_policy_checkpoint, saved)
    with pytest.raises(ValueError, match="Resume episode layout mismatch"):
        resume.main(["--run", "policy_run", "--max-updates", "1"])
    assert load_checkpoint(small_policy_checkpoint).global_step == 1

    saved.effective_config["episode_layout"] = layout
    save_checkpoint(small_policy_checkpoint, saved)
    resume.main(["--run", "policy_run", "--max-updates", "1"])
    assert Path.cwd() == tmp_path
    checkpoint = load_checkpoint(small_policy_checkpoint)
    assert checkpoint.global_step == 2
    assert checkpoint.effective_config["workspace"]["output_dir"] == str(tmp_path / "policy_run")
    assert checkpoint.effective_config["dataset"]["zarr_path"] == str(tmp_path / "demo.zarr")
    assert checkpoint.effective_config["training_data"] == saved.effective_config["training_data"]
    assert checkpoint.effective_config["episode_layout"] == layout
