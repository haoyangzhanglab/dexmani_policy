"""Focused regressions for checkpoint metadata and evaluation state selection."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dexmani_policy.common.checkpoint_io import TrainCheckpoint, build_train_params
from dexmani_policy.training.eval_utils import load_ckpt_for_inference


class _FittedNormalizer:
    def is_fitted(self, required_keys):
        return required_keys == ["action"]


class _Agent:
    normalizer = _FittedNormalizer()

    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict, strict):
        self.loaded_state = state_dict
        self.strict = strict


class _CheckpointStore:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def load(self, path):
        return self.checkpoint


def _checkpoint(train_params=None, ema_model_state=None):
    return TrainCheckpoint(
        epoch=0,
        global_step=1,
        model_state={"weight": torch.tensor(1.0)},
        ema_model_state=ema_model_state,
        optimizer_state={},
        scheduler_state={},
        monitor={},
        train_params=train_params,
    )


class CheckpointCorrectnessTest(unittest.TestCase):
    def test_train_params_records_aux_ee_semantics(self):
        model = SimpleNamespace(
            n_obs_steps=2,
            n_action_steps=8,
            action_dim=28,
            horizon=16,
            action_key="action",
            tcp_dim=7,
            hand_dim=12,
            control_action_dim=19,
            use_aux_ee=True,
        )

        params = build_train_params(model, num_training_steps=100)

        self.assertIs(params["use_aux_ee"], True)

    def test_train_params_defaults_aux_ee_to_false(self):
        model = SimpleNamespace(
            n_obs_steps=2,
            n_action_steps=8,
            action_dim=19,
            horizon=16,
            control_action_dim=19,
        )

        params = build_train_params(model)

        self.assertIs(params["use_aux_ee"], False)

    def test_eval_loads_model_state_when_metadata_is_absent(self):
        agent = _Agent()
        checkpoint = _checkpoint(train_params=None)

        load_ckpt_for_inference(
            agent,
            _CheckpointStore(checkpoint),
            Path("unused.pt"),
            use_ema=False,
        )

        self.assertTrue(torch.equal(agent.loaded_state["weight"], torch.tensor(1.0)))
        self.assertTrue(agent.strict)

    def test_eval_selects_ema_independently_of_metadata(self):
        agent = _Agent()
        checkpoint = _checkpoint(train_params=None, ema_model_state={"weight": torch.tensor(2.0)})

        load_ckpt_for_inference(
            agent,
            _CheckpointStore(checkpoint),
            Path("unused.pt"),
            use_ema=True,
        )

        self.assertTrue(torch.equal(agent.loaded_state["weight"], torch.tensor(2.0)))
