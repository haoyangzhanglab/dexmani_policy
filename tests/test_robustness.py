"""CPU regressions for deployment and training-boundary robustness."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.agents.action_decoders.flowmatch import FlowMatchWithConsistency
from dexmani_policy.common.normalizer import SingleFieldLinearNormalizer
from dexmani_policy.deployment.contract import DeploymentSpec, ObservationFieldSpec
from dexmani_policy.deployment.restore import RestoredDeployment, prediction_snapshot
from dexmani_policy.training.build_utils import (
    _validate_augmentation_consistency,
    validate_config,
)


def _deployment_spec() -> DeploymentSpec:
    return DeploymentSpec(
        action_key="action",
        action_dim=19,
        horizon=2,
        n_obs_steps=1,
        n_action_steps=1,
        denoise_steps=1,
        observation_fields=(
            ObservationFieldSpec("joint_state", (1,), "float32", {}),
        ),
        control_dt_s=0.02,
        requires_hand=True,
        rgb_preprocessing=None,
    )


class _MetaDeviceAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0, device="meta"))
        self.seen_devices: list[torch.device] = []

    def predict_action(self, observation, denoise_timesteps=None):
        del denoise_timesteps
        self.seen_devices = [value.device for value in observation.values()]
        batch_size = next(iter(observation.values())).shape[0]
        pred = torch.zeros(batch_size, 2, 19)
        return {"pred_action": pred, "control_action": pred[:, :1]}


def test_prediction_snapshot_uses_the_agent_current_device_for_inputs():
    spec = _deployment_spec()
    agent = _MetaDeviceAgent()
    restored = RestoredDeployment(agent=agent, spec=spec)
    cpu_observation = {"joint_state": torch.ones(1, 1, 1)}
    synthetic_devices = []

    def synthetic(spec, *, batch_size=1, device="cpu"):
        del spec, batch_size
        synthetic_devices.append(torch.device(device))
        return cpu_observation

    with patch(
        "dexmani_policy.deployment.restore.deterministic_observation",
        side_effect=synthetic,
    ):
        prediction_snapshot(restored)
    assert synthetic_devices == [torch.device("meta")]
    assert agent.seen_devices == [torch.device("meta")]

    prediction_snapshot(restored, observation=cpu_observation)
    assert agent.seen_devices == [torch.device("meta")]


def test_manual_normalizer_parameters_are_frozen_on_creation():
    normalizer = SingleFieldLinearNormalizer.create_manual(
        scale=torch.tensor([2.0, 3.0]), offset=torch.tensor([-1.0, 4.0])
    )
    assert all(not parameter.requires_grad for parameter in normalizer.parameters())


@pytest.mark.parametrize("augmentation", ["color", "color_noise"])
def test_pc_color_config_errors_are_value_errors(augmentation):
    cfg = OmegaConf.create(
        {
            "agent": {"pc_dim": 3},
            "dataset": {"augmentation_cfg": {"pc": {augmentation: {}}}},
        }
    )
    with pytest.raises(ValueError, match="pc_dim >= 6"):
        _validate_augmentation_consistency(cfg)


def test_negative_obs_lr_is_a_value_error():
    cfg = OmegaConf.create(
        {
            "training": {},
            "n_obs_steps": 1,
            "n_action_steps": 1,
            "horizon": 2,
            "optimizer": {"obs_lr": -1.0},
            "agent": {"pc_dim": 6},
            "dataset": {"action_key": "action"},
            "action_key": "action",
            "env_runner": {"env_kwargs": {"control_mode": "joint"}},
        }
    )
    with pytest.raises(ValueError, match="optimizer.obs_lr"):
        validate_config(cfg)


class _RecordingTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep = None
        self.target_t = None

    def forward(self, x, timestep, target_t, context):
        del context
        self.timestep = timestep.detach().clone()
        self.target_t = target_t.detach().clone()
        return torch.zeros_like(x)


def test_maniflow_relative_target_t_keeps_dt_after_t_next_is_clamped():
    decoder = FlowMatchWithConsistency(nn.Identity(), target_t_sample_mode="relative")
    teacher = _RecordingTeacher()
    samples = iter((torch.tensor([0.8]), torch.tensor([0.5])))
    decoder.sampler.sample = lambda batch_size, mode, device: next(samples).to(device)

    targets = decoder.get_consistency_velocity(
        torch.ones(1, 2, 19), torch.zeros(1, 1, 3), teacher
    )

    torch.testing.assert_close(targets["t"].flatten(), torch.tensor([0.8]))
    torch.testing.assert_close(targets["target_t"].flatten(), torch.tensor([0.5]))
    torch.testing.assert_close(teacher.timestep, torch.tensor(1.0))
    torch.testing.assert_close(teacher.target_t, torch.tensor([0.5]))
