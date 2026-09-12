"""CPU regressions for training-checkpoint inference consumers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.deployment import export as deployment_export
from dexmani_policy.deployment.export import InvalidCheckpointError
from dexmani_policy.training.eval_utils import load_ckpt_for_inference


def _agent_contract() -> dict[str, object]:
    return {
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "action_dim": 19,
        "horizon": 16,
        "action_key": "action",
        "tcp_dim": None,
        "hand_dim": None,
        "control_action_dim": 19,
        "use_aux_ee": False,
    }


class _Normalizer:
    def is_fitted(self, required_keys):
        return required_keys == ["action"]


class _Agent:
    n_obs_steps = 2
    n_action_steps = 8
    action_dim = 19
    horizon = 16
    action_key = "action"
    tcp_dim = None
    hand_dim = None
    control_action_dim = 19
    use_aux_ee = False

    def __init__(self):
        self.normalizer = _Normalizer()
        self.loaded_state = None

    def load_state_dict(self, state, strict):
        self.loaded_state = state, strict


def _save_checkpoint(store, agent_contract):
    return store.save(
        "checkpoint.pt",
        TrainCheckpoint(
            epoch=0,
            global_step=1,
            next_micro_step=0,
            model_state={"weight": torch.ones(1)},
            ema_model_state=None,
            optimizer_state={},
            scheduler_state={},
            monitor={},
            resume_contract={
                "agent": agent_contract,
                "training": {"num_training_steps": 100},
            },
            ema_updater_step=None,
            ema_decay=None,
            rng_states=[{}],
        ),
    )


class CheckpointConsumerTest(unittest.TestCase):
    def test_eval_and_export_read_v3_agent_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(Path(tmpdir))
            path = _save_checkpoint(store, _agent_contract())
            agent = _Agent()

            load_ckpt_for_inference(agent, store, path, False)
            self.assertEqual(agent.loaded_state[1], True)
            self.assertTrue(
                torch.equal(agent.loaded_state[0]["weight"], torch.ones(1))
            )

            cfg = {
                "action_key": "action",
                "action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "use_aux_ee": False,
                "agent": {},
            }
            loaded = store.load(path)
            train, provenance, retrofitted = deployment_export._reconcile_train_params(
                loaded, cfg
            )
            self.assertEqual(train, _agent_contract())
            self.assertEqual(provenance, "native")
            self.assertEqual(retrofitted, [])

    def test_eval_rejects_missing_and_changed_agent_contract_values(self):
        cases = (
            ("hand_dim", None, "resume_contract.agent.hand_dim: missing in checkpoint"),
            ("control_action_dim", 18, "resume_contract.agent.control_action_dim"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmpdir:
                contract = _agent_contract()
                if value is None:
                    del contract[key]
                else:
                    contract[key] = value
                store = CheckpointStore(Path(tmpdir))
                path = _save_checkpoint(store, contract)
                with self.assertRaisesRegex(ValueError, message):
                    load_ckpt_for_inference(_Agent(), store, path, False)

    def test_export_requires_use_aux_ee_in_v3_agent_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract = _agent_contract()
            del contract["use_aux_ee"]
            store = CheckpointStore(Path(tmpdir))
            path = _save_checkpoint(store, contract)
            cfg = {
                "action_key": "action",
                "action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "use_aux_ee": False,
                "agent": {},
            }
            with self.assertRaisesRegex(
                InvalidCheckpointError, "resume_contract.agent is missing use_aux_ee"
            ):
                deployment_export._reconcile_train_params(store.load(path), cfg)


if __name__ == "__main__":
    unittest.main()
