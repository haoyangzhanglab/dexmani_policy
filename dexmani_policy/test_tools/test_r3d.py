import numpy as np
import pytest
import torch
from torch import nn

from dexmani_policy.agents.policies import r3d
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.variants.r3d import R3DDataset
from dexmani_policy.training.config import load_config
from dexmani_policy.training.data import build_dataset, fit_action_normalizer


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
    target = "dexmani_policy.test_tools.test_r3d.CustomR3DDataset"
    cfg = load_config("r3d", [
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


@pytest.mark.parametrize("auxiliary,weight", [(False, 1), (True, 0), (True, 0.7)])
@pytest.mark.parametrize("log_metrics", [False, True])
def test_r3d_group_loss_and_gradients(monkeypatch, auxiliary, weight, log_metrics):
    monkeypatch.setattr(r3d, "R3DEncoder", SmallObservationEncoder)
    dim = 28 if auxiliary else 19
    model = r3d.R3D(
        horizon=4, n_obs_steps=1, n_action_steps=4, action_dim=dim, state_dim=19,
        use_aux_ee=auxiliary, aux_loss_weight=weight, embedding_dim=16, depth=1,
        num_heads=2, mlp_dim=32, timestep_embed_dim=8, num_inference_steps=1,
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
