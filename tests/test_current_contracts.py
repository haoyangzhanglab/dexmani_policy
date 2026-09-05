from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager
from dexmani_policy.agents.vq_hand.vqvae import VQVAEHand
from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    TrainCheckpoint,
    validate_training_steps,
)
from dexmani_policy.common.config import validate_action_key_consistency
from dexmani_policy.common.pytorch_util import get_rng_state


class CurrentContractsTest(unittest.TestCase):
    def test_action_key_is_explicit(self) -> None:
        for payload in ({}, {"action_mode": "joint"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_action_key_consistency(OmegaConf.create(payload))

    def test_training_checkpoint_requires_complete_schema(self) -> None:
        checkpoint = TrainCheckpoint(
            epoch=1,
            global_step=2,
            model_state={"weight": torch.ones(1)},
            ema_model_state=None,
            optimizer_state={},
            scheduler_state={},
            monitor={},
            train_params={"num_training_steps": 10},
            ema_updater_step=None,
            ema_decay=None,
            rng_state=get_rng_state(),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory))
            path = store.save("checkpoint.pt", checkpoint)
            loaded = store.load(path)
            self.assertEqual(loaded.global_step, 2)
            validate_training_steps(loaded, 10)
            with self.assertRaises(ValueError):
                validate_training_steps(loaded, 11)

            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["state"]["rng_state"]
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                store.load(path)

    def test_vq_checkpoint_and_codebook_require_current_formats(self) -> None:
        model = VQVAEHand(
            hand_dim=2,
            loss_weight=[1.0, 1.0],
            latent_dim=4,
            hidden_dim=8,
            num_groups=2,
            codebook_size=2,
            num_layers=1,
            kmeans_iters=3,
        )
        checkpoint = {
            "format_version": 3,
            "model_config": {
                "hand_dim": 2,
                "loss_weight": [1.0, 1.0],
                "latent_dim": 4,
                "hidden_dim": 8,
                "num_groups": 2,
                "codebook_size": 2,
                "num_layers": 1,
                "act_scale": 1.0,
                "vq_decay": 0.8,
                "threshold_ema_dead_code": 0,
                "kmeans_iters": 3,
            },
            "model_state_dict": model.state_dict(),
        }
        restored = VQVAEHand.from_checkpoint(checkpoint)
        self.assertEqual(restored.num_layers, 1)

        checkpoint["format_version"] = 2
        with self.assertRaises(ValueError):
            VQVAEHand.from_checkpoint(checkpoint)

        manager = CodebookManager(hand_dim=2, num_groups=2, codebook_size=2)
        manager.sorted_hand_poses = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        manager.pca_permutation = torch.arange(4)
        manager.layer_weights = torch.ones(2)
        manager.set_hand_normalizer(np.ones(2), np.zeros(2))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codebook.npz"
            manager.save(path)
            loaded = CodebookManager(hand_dim=2, num_groups=2, codebook_size=2)
            loaded.load(path)
            torch.testing.assert_close(
                loaded.sorted_hand_poses, manager.sorted_hand_poses
            )

            npy_path = Path(directory) / "codebook.npy"
            np.save(npy_path, np.zeros((4, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                loaded.load(npy_path)


if __name__ == "__main__":
    unittest.main()
