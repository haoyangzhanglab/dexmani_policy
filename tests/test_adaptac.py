import json
import math
from contextlib import closing
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import default_collate

from dexmani_policy.agents.action_decoders.diffusion import ForceDiffusion
from dexmani_policy.agents.obs_encoder.pointcloud import point_transformer
from dexmani_policy.agents.policies.adaptac import AdapTac
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.common import config as common_config
from dexmani_policy.datasets.variants.tactile import AdapTacDataset
from dexmani_policy.training import trainer as policy_training
from dexmani_policy.training.config import load_policy_config, validate_config
from dexmani_policy.training.optimization import clip_gradients

from tests.helpers import (
    MeanDecoder, assert_loader_matches_items, check_observation_identity_constraints,
    check_recipe_composition, training_config,
)


@pytest.mark.parametrize("mode", ["force", "taxel"])
def test_adaptac_padded_force_invariance(monkeypatch, mode):
    # CPU reference grouping keeps this loss test independent of the native point-cloud extension.
    def group(points, num_patches, group_size, *args):
        centers = points[:, :num_patches, :3]
        return centers, torch.cdist(centers, points[..., :3]).topk(group_size, largest=False).indices

    monkeypatch.setattr(point_transformer, "sample_point_patches", group)
    cfg = load_policy_config("adaptac", [
        f"tactile={mode}", "agent.action_dim=19", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.feature_dim=32", "agent.state_out_dim=16", "agent.num_points=16",
        "agent.pc_encoder_config.channels=48", "agent.pc_encoder_config.num_patches=4",
        "agent.pc_encoder_config.group_size=4", "agent.pc_encoder_config.num_heads=4",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=32",
        "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=32",
    ])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    shape = common_config.TACTILE_FEATURE_SHAPES[model.force_key]
    model.configure_normalizer(
        {"action": 19, "point_cloud": 6, "joint_state": 19, model.force_key: math.prod(shape)},
        {"action": (19,), "joint_state": (19,)},
    )
    batch = {"obs": {"point_cloud": torch.randn(2, 1, 16, 6), "joint_state": torch.randn(2, 1, 19),
                     model.force_key: torch.randn(2, 1, *shape)}, "action": torch.randn(2, 4, 19)}
    force = torch.randn(2, 4, *shape)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    conditions = []
    action_loss = model.action_decoder.compute_loss

    def capture(condition, *args, **kwargs):
        conditions.append(condition.detach())
        return action_loss(condition, *args, **kwargs)

    monkeypatch.setattr(model.action_decoder, "compute_loss", capture)

    def run(target, mask):
        torch.manual_seed(11)
        loss, metrics = model.compute_loss(dict(batch, future_force=target, future_force_valid=mask))
        return conditions[-1], metrics, torch.autograd.grad(loss, params, allow_unused=True)

    expected = run(force, valid)
    for invalid in (0.0, -1e9, float("nan"), float("inf"), -float("inf")):
        target = force.clone()
        target[~valid] = invalid
        torch.testing.assert_close(run(target, valid), expected, rtol=0, atol=0)

    model.force_loss_weight = 0
    _, metrics, grads = run(force, valid)
    assert metrics["loss_force_weighted"] == 0
    torch.testing.assert_close(metrics["loss"], metrics["loss_action"])
    force_parameters = {id(parameter) for parameter in model.force_predictor.parameters()}
    assert any(grad is not None and grad.abs().max() > 0
               for parameter, grad in zip(params, grads) if id(parameter) in force_parameters)

    model.force_loss_weight = 1
    all_valid = torch.ones_like(valid)
    actual = run(force, all_valid)

    def original(condition, target, mask):
        prediction, target = model.force_predictor._training_prediction(condition, target.float())
        error = (prediction.float() - target).square() * mask[..., None]
        return error.sum() / (target.shape[-1] * mask.float().sum()), prediction.float()

    monkeypatch.setattr(model.force_predictor, "compute_loss_and_prediction", original)
    torch.testing.assert_close(run(force, all_valid), actual, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("tactile", ["force", "taxel"])
def test_adaptac_tactile_recipe(tactile):
    cfg = load_policy_config("adaptac", [f"tactile={tactile}"])
    validate_config(cfg)
    encoder = hydra.utils.instantiate(cfg.agent.tactile_encoder)
    assert encoder.input_key in cfg.dataset.sensor_modalities
    assert cfg.normalization[encoder.input_key] == "identity"


@pytest.mark.parametrize("key", ["finger_tactile_force", "taxel_tactile_force"])
def test_adaptac_batch_force_padding(segments, monkeypatch, key):
    path, fields = segments
    if key == "taxel_tactile_force":
        values = np.repeat(fields["finger_tactile_force"][:, :, None, :], 120, axis=2)
        zarr.open_group(str(path), mode="a").create_dataset(f"data/{key}", data=values)
    dataset = AdapTacDataset(path, sensor_modalities=[key], horizon=4, obs_horizon=2, pad_before=2, pad_after=3)
    batch = assert_loader_matches_items(dataset, monkeypatch)
    assert not batch["future_force"][~batch["future_force_valid"]].any()


@pytest.mark.parametrize("force_weight", [0.0, 0.7])
def test_adaptac_accumulation_matches_combined_force_and_action_means(tmp_path, monkeypatch, force_weight):
    class FixedForce(ForceDiffusion):
        def _training_prediction(self, condition, target):
            return self.model.weight.expand_as(target), target

    class ActionDecoder(MeanDecoder):
        def compute_loss(self, condition, target, *, log_metrics):
            loss = (self.weight + condition - target).square().mean()
            return loss, {}

    class SmallAdapTac(AdapTac):
        def __init__(self):
            Policy.__init__(self, nn.Identity(), ActionDecoder(), 4, 1, 4, 1)
            Policy.configure_normalizer(self, {"action": 1}, {"action": (1,)})
            self.force_predictor = FixedForce(MeanDecoder())
            self.force_fusion = nn.Identity()
            self.force_loss_weight = force_weight

        def _encode_force_inputs(self, obs):
            return None, None, None

        def _action_condition(self, features, current_force, prediction):
            return prediction

    dataset = [
        {"obs": {}, "action": torch.full((4, 1), action),
         "future_force": torch.full((4, 1), force), "future_force_valid": torch.arange(4) < valid}
        for action, force, valid in [(0.5, 0.0, 1), (0.7, 0.0, 1), (0.9, 1.0, 4)]
    ]
    model = SmallAdapTac()
    expected_loss, expected_metrics = model.compute_loss(default_collate(dataset))
    expected_gradients = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    updates = []

    def capture_clip(parameters, finite_loss, max_grad_norm=0, *, step):
        updates.append([p.grad.clone() for p in model.parameters()])
        return clip_gradients(parameters, finite_loss, max_grad_norm, step=step)

    monkeypatch.setattr(policy_training, "clip_gradients", capture_clip)
    cfg = training_config(tmp_path, batch_size=2, accumulation=2, updates=1)
    with closing(policy_training.PolicyTrainer(cfg, model, dataset)) as trainer:
        trainer.train()
    assert len(updates) == 1
    torch.testing.assert_close(updates[0], list(expected_gradients))
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    for key, value in expected_metrics.items():
        assert record[f"train/{key}"] == pytest.approx(float(value))


def test_recipe_composition():
    check_recipe_composition("adaptac")
    check_observation_identity_constraints("adaptac", ['point_cloud'])


def test_tactile_temporal_targets(segments):
    path, fields = segments
    group = zarr.open_group(str(path), mode="a")
    group["meta/episode_ends"][:] = [3, 6, 9, 12]
    tactile = AdapTacDataset(path, sensor_modalities=["finger_tactile_force"],
                             horizon=4, obs_horizon=2, pad_before=1, pad_after=3)
    assert len(tactile) == 8
    for episode in range(4):
        start = episode * 3
        for offset in range(2):
            sample = tactile[episode * 2 + offset]
            expected = [start + 1, start + 2, 0] if offset == 0 else [start + 2, 0, 0]
            np.testing.assert_array_equal(sample["future_force"][:, 0, 0], expected)
            np.testing.assert_array_equal(sample["future_force_valid"], [True, offset == 0, False])


def test_ddim_force_recipe_rejects_one_step():
    cfg = load_policy_config("adaptac", ["agent.action_dim=19", "agent.force_num_inference_steps=1"])
    with pytest.raises(hydra.errors.InstantiationException) as error:
        hydra.utils.instantiate(cfg.agent)
    assert isinstance(error.value.__cause__, ValueError)
    assert "leading DDIM" in str(error.value.__cause__)


def test_warm_start_requires_same_tactile_encoder(monkeypatch):
    cfg = load_policy_config("adaptac", ["policy_name=renamed_experiment"])
    saved = OmegaConf.to_container(cfg, resolve=False)
    checkpoint = SimpleNamespace(effective_config=saved)
    expected = object()
    monkeypatch.setattr(Policy, "build_for_training", classmethod(lambda cls, *args, **kwargs: expected))
    assert AdapTac.build_for_training(cfg, None, {}, checkpoint=checkpoint) is expected
    cfg.agent.tactile_encoder._target_ = "different.encoder"
    with pytest.raises(ValueError, match="Tactile encoder"):
        AdapTac.build_for_training(cfg, None, {}, checkpoint=checkpoint)
