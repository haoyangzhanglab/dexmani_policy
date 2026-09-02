"""Regression tests for §5.6 (fail-closed pretrained weight loading).

Uni3D must raise on a missing path / download failure / large-scale key mismatch
unless ``allow_random_init=True`` explicitly permits random initialization.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DPointcloudEncoder


def _make_encoder(**kwargs):
    cfg = dict(
        pc_model="eva02_tiny_patch14_224",
        embed_dim=64,
        num_group=32,
        group_size=32,
        pc_in_channels=6,
        feature_mode="pointsam",
    )
    cfg.update(kwargs)
    return Uni3DPointcloudEncoder(**cfg)


class TestPretrainedWeightsFailClosed(unittest.TestCase):
    def test_missing_path_fail_closed(self):
        with self.assertRaises(ValueError):
            _make_encoder(
                use_pretrained_weights=True,
                pretrained_weights_path=None,
                allow_random_init=False,
            )

    def test_missing_path_allow_random_init(self):
        # Explicit opt-in to random init must not raise.
        _make_encoder(
            use_pretrained_weights=True,
            pretrained_weights_path=None,
            allow_random_init=True,
        )

    def test_large_scale_key_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            safetensors_path = os.path.join(d, "model.safetensors")
            from safetensors.torch import save_file

            save_file({"totally_wrong_key": torch.zeros(1)}, safetensors_path)
            with self.assertRaises(ValueError):
                _make_encoder(
                    use_pretrained_weights=True,
                    pretrained_weights_path=d,
                    allow_random_init=False,
                )


if __name__ == "__main__":
    unittest.main()
