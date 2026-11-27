import copy
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.agents.policies import r3d
from dexmani_policy.common.checkpoint import load_checkpoint, save_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.variants.r3d import R3DDataset
from dexmani_policy.deployment import load_policy
from dexmani_policy.training.config import load_policy_config
from dexmani_policy.training.data import build_dataset, fit_action_normalizer
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer

from tests.helpers import check_recipe_composition


@pytest.mark.parametrize("pc_dim", [3, 6])
def test_r3d_limits_config_and_restoration(tmp_path, make_demo, monkeypatch, pc_dim):
    class SmallUni3D(nn.Module):
        def __init__(self, embed_dim, **kwargs):
            super().__init__()
            self.out_dim = embed_dim
            self.project = nn.Linear(3, embed_dim)

        def initialize_weights(self, source):
            assert source is None

        def forward(self, cloud):
            projected = self.project(cloud[..., :3])
            return projected, torch.zeros_like(projected)

    monkeypatch.setattr(r3d, "Uni3DEncoder", SmallUni3D)
    data = make_demo()
    directory = tmp_path / "r3d"
    cfg = load_policy_config("r3d", [
        f"zarr_path={data}", "task_name=test", f"workspace.output_dir={directory}",
        "workspace.wandb_cfg=null", "initialization.uni3d=null", "dataset.augmentation_cfg=null",
        f"+dataset.pc_dim={pc_dim}",
        "training.device=cpu", "training.use_compile=false", "training.loop.total_train_steps=3",
        "dataloader.num_workers=0", "dataloader.batch_size=2", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.pc_encoder_config.embed_dim=16", "agent.embedding_dim=16", "agent.num_heads=2", "agent.depth=1",
        "agent.state_out_dim=8", "agent.mlp_dim=32", "agent.timestep_embed_dim=8",
    ])
    cfg.policy_name = "renamed_experiment"
    unsupported = copy.deepcopy(cfg)
    unsupported.normalization.point_cloud = "mean_std"
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(unsupported)
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    checkpoint = load_checkpoint(latest)
    assert checkpoint.effective_config["normalization"]["point_cloud"] == "limits"

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not rescan training arrays")

    with monkeypatch.context() as patch:
        patch.setattr(EpisodeStore, "iter_chunks", forbidden)
        with closing(prepare_resume(directory)) as trainer:
            assert trainer.global_step == 1
            torch.testing.assert_close(trainer.model.normalizer["point_cloud"].scale,
                                       checkpoint.model_state["normalizer.fields.point_cloud.scale"])
    cfg.workspace.output_dir = str(tmp_path / "warm")
    cfg.initialization.policy = str(latest)
    warm_data = make_demo("warm")
    cfg.zarr_path = cfg.dataset.zarr_path = str(warm_data)
    cloud = zarr.open_group(str(warm_data), mode="r+")["data/point_cloud"]
    if pc_dim == 3:
        cloud[6, 2, 5] = np.inf
    with closing(build_trainer(cfg)) as trainer:
        assert trainer.global_step == 0
    for weights in ("raw", "ema"):
        assert load_policy(latest, weights=weights, device="cpu").agent.normalization_dims["point_cloud"] == pc_dim
    cfg.workspace.output_dir = str(tmp_path / "warm_bad")
    column = pc_dim - 1
    value = cloud[6, 2, column]
    cloud[6, 2, column] = np.inf
    with pytest.raises(ValueError, match=rf"field=point_cloud,.*row=6, channel=\(2, {column}\)"):
        build_trainer(cfg)
    assert load_checkpoint(latest).global_step == 1
    assert not Path(cfg.workspace.output_dir).exists()
    cloud[6, 2, column] = value
    checkpoint.effective_config["normalization"]["point_cloud"] = "mean_std"
    bad = save_checkpoint(directory / "checkpoints/bad.pt", checkpoint)
    cfg.initialization.policy = str(bad)
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(cfg)
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(OmegaConf.create(checkpoint.effective_config), checkpoint=checkpoint)
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        load_policy(bad, device="cpu")


class SmallObservationEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(19, 32)
        self.out_dim = 32
        self.pc_pe_dim = 16

    def forward(self, obs):
        return self.projection(obs["joint_state"]).unsqueeze(1)


class ConstantPrediction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.values = nn.Parameter(torch.linspace(-0.5, 0.5, dim))

    def forward(self, x, timestep, context):
        return self.values.expand_as(x) + context.mean((1, 2))[:, None, None]


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")])
def test_r3d_rejects_invalid_auxiliary_loss_weight(weight):
    with pytest.raises(ValueError, match="aux_loss_weight must be finite and nonnegative"):
        r3d.R3D(
            horizon=4, n_obs_steps=1, n_action_steps=4, action_dim=28, state_dim=19,
            use_aux_ee=True, aux_loss_weight=weight,
        )


@pytest.mark.parametrize("auxiliary,weight", [(False, 1), (True, 0), (True, 0.7)])
@pytest.mark.parametrize("log_metrics", [False, True])
def test_r3d_group_loss_and_gradients(monkeypatch, auxiliary, weight, log_metrics):
    monkeypatch.setattr(r3d, "R3DEncoder", SmallObservationEncoder)
    dim = 28 if auxiliary else 19
    model = r3d.R3D(
        horizon=4, n_obs_steps=1, n_action_steps=4, action_dim=dim, state_dim=19,
        use_aux_ee=auxiliary, aux_loss_weight=weight, embedding_dim=16, depth=1,
        num_heads=2, mlp_dim=32, timestep_embed_dim=8, num_inference_steps=2,
    )
    model.action_decoder.model = ConstantPrediction(dim)
    model.configure_normalizer({"action": dim}, {"action": (dim,)})
    model.normalizer["action"].scale.copy_(torch.linspace(1, 2, dim))
    model.normalizer["action"].offset.fill_(-0.25)
    actions = torch.randn(2, 4, dim)
    batch = {"obs": {"joint_state": torch.randn(2, 1, 19)}, "action": actions}
    token = object()
    compute_loss = model.action_decoder.compute_loss

    def with_extra_kwarg(*args, probe, **kwargs):
        assert probe is token
        return compute_loss(*args, **kwargs)

    monkeypatch.setattr(model.action_decoder, "compute_loss", with_extra_kwarg)
    loss, metrics = model.compute_loss(batch, log_metrics=log_metrics, probe=token)
    condition = model._encode_obs(batch["obs"])
    prediction = model.action_decoder.model(actions, 0, condition)
    error = (prediction - model._encode_actions(actions)).square()
    joint = error[..., :19].mean()
    expected = joint + weight * error[..., 19:].mean() if auxiliary else error.mean()
    torch.testing.assert_close(loss, expected)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(loss, parameters)
    expected_gradients = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    if auxiliary and weight == 0:
        assert torch.count_nonzero(actual_gradients[-1][19:]) == 0
    if log_metrics:
        torch.testing.assert_close(metrics["loss"], expected.detach())
        if auxiliary:
            torch.testing.assert_close(metrics["loss_joint"], joint.detach())
            torch.testing.assert_close(metrics["loss_ee"], error[..., 19:].mean().detach())
        else:
            assert set(metrics) == {"loss"}
    else:
        assert metrics == {}
    assert model.predict_action(batch["obs"])["future_actions"].shape == (2, 4, 19)


class CustomR3DDataset(R3DDataset):
    pass


@pytest.mark.parametrize("auxiliary,action_key,width", [
    (False, "action", 19), (True, "action", 28), (False, "action_ee", 21),
])
def test_r3d_sampling_and_normalization(make_demo, auxiliary, action_key, width):
    path = make_demo(auxiliary=auxiliary or action_key == "action_ee")
    dataset = R3DDataset(
        path, use_aux_ee=auxiliary, action_key=action_key,
        sensor_modalities=["joint_state"], horizon=2, obs_horizon=1,
    )
    values = dataset.episode_store.read(action_key, slice(None))
    if auxiliary:
        values = np.concatenate((values, dataset.episode_store.read("action_ee", slice(None))[:, :9]), -1)
        assert dataset.consumed_fields["action_ee"] == slice(0, 9)
    elif action_key == "action":
        assert "action_ee" not in dataset.episode_store.keys()
        assert "action_ee" not in dataset.consumed_fields
    assert dataset.action_dim == width
    np.testing.assert_array_equal(dataset[0]["action"], values[:2])
    np.testing.assert_array_equal(np.concatenate(list(dataset.iter_normalization_data("action"))), values)
    normalizer = LinearNormalizer({"action": (width,)})
    fit_action_normalizer(dataset, normalizer)
    scale = 2 / (values.max(0) - values.min(0))
    offset = -1 - scale * values.min(0)
    if action_key == "action_ee":
        scale[3:9], offset[3:9] = 1, 0
    torch.testing.assert_close(normalizer["action"].scale, torch.from_numpy(scale))
    torch.testing.assert_close(normalizer["action"].offset, torch.from_numpy(offset))


@pytest.mark.parametrize("action_key", ["action_ee", "action_rotvec"])
def test_auxiliary_requires_joint_action(make_demo, action_key):
    with pytest.raises(ValueError, match="joint action"):
        R3DDataset(make_demo(), use_aux_ee=True, action_key=action_key,
                   sensor_modalities=["joint_state"], horizon=2, obs_horizon=1)


@pytest.mark.parametrize("auxiliary,width", [(False, 19), (True, 28)])
def test_r3d_explicit_dimensions_and_custom_target(make_demo, auxiliary, width):
    target = f"{__name__}.CustomR3DDataset"
    cfg = load_policy_config("r3d", [
        f"zarr_path={make_demo()}", "task_name=test", "dataset.augmentation_cfg=null",
        f"++dataset._target_={target}", "++agent.action_dim=20",
        f"use_aux_ee={str(auxiliary).lower()}",
    ])
    assert cfg.dataset._target_ == target
    assert cfg.agent.action_dim == 20
    with pytest.raises(ValueError, match="action width"):
        build_dataset(cfg)
    cfg.agent.action_dim = None
    dataset, description = build_dataset(cfg)
    assert isinstance(dataset, CustomR3DDataset)
    assert cfg.agent.action_dim == description["field_dims"]["action"] == width


def test_recipe_composition():
    cfg = check_recipe_composition("r3d")
    assert cfg.dataset.use_aux_ee == cfg.agent.use_aux_ee == cfg.use_aux_ee
