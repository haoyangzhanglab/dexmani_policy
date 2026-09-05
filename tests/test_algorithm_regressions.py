from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr

from dexmani_policy.agents.action_decoders.flowmatch import FlowMatchWithConsistency
from dexmani_policy.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from dexmani_policy.datasets.augmentation import PointDropout
from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
from dexmani_policy.datasets.replay_buffer import ReplayBuffer


class _FixedSampler:
    def __init__(self, *samples: torch.Tensor) -> None:
        self._samples = list(samples)

    def sample(self, batch_size: int, mode: str, device: torch.device) -> torch.Tensor:
        sample = self._samples.pop(0)
        assert sample.shape == (batch_size,)
        return sample.to(device=device)


class _RecordingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(
        self,
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        target_t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append((timestep.clone(), target_t.clone()))
        return torch.zeros_like(x)


class AlgorithmRegressionTests(unittest.TestCase):
    def _consistency_targets(self, mode: str):
        teacher = _RecordingVelocity()
        decoder = FlowMatchWithConsistency(
            model=nn.Identity(),
            denoise_timesteps=10,
            target_t_sample_mode=mode,
        )
        decoder.sampler = _FixedSampler(
            torch.tensor([0.2, 0.8]), torch.tensor([0.1, 0.4])
        )

        targets = decoder.get_consistency_velocity(
            actions=torch.ones(2, 1, 1),
            cond=torch.zeros(2, 1),
            ema_model=teacher,
        )
        return targets, teacher.calls

    def test_absolute_consistency_uses_next_state_for_student_target(self) -> None:
        targets, teacher_calls = self._consistency_targets("absolute")

        torch.testing.assert_close(
            targets["target_t"], torch.tensor([0.3, 1.0]), atol=1e-6, rtol=0
        )
        self.assertEqual(len(teacher_calls), 1)
        teacher_timestep, teacher_target_t = teacher_calls[0]
        torch.testing.assert_close(
            teacher_timestep, torch.tensor([0.3, 1.0]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            teacher_target_t, torch.tensor([0.4, 1.0]), atol=1e-6, rtol=0
        )

    def test_relative_consistency_targets_remain_relative_dt(self) -> None:
        targets, teacher_calls = self._consistency_targets("relative")

        expected_dt = torch.tensor([0.1, 0.4])
        torch.testing.assert_close(targets["target_t"], expected_dt)
        self.assertEqual(len(teacher_calls), 1)
        torch.testing.assert_close(teacher_calls[0][1], expected_dt)

    def test_flowmatch_rejects_invalid_core_arguments(self) -> None:
        model = nn.Identity()
        invalid_kwargs = [
            {"denoise_timesteps": 0},
            {"flow_batch_ratio": 0},
            {"flow_batch_ratio": 1},
            {"target_t_sample_mode": "invalid"},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FlowMatchWithConsistency(model, **kwargs)

    def test_point_dropout_zero_is_exact_noop_and_ratio_is_bounded(self) -> None:
        points = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        expected = points.copy()

        PointDropout(dropout_ratio=0)._augment(points)

        np.testing.assert_array_equal(points, expected)
        for ratio in (-0.01, 1.01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                PointDropout(dropout_ratio=ratio)

    def test_normalizer_field_views_are_invalidated_by_fit_and_setitem(self) -> None:
        normalizer = LinearNormalizer()
        normalizer.fit({"action": np.array([[0.0], [1.0]])})
        before_fit = normalizer["action"]
        normalizer.fit({"action": np.array([[10.0], [20.0]])})
        after_fit = normalizer["action"]
        self.assertIsNot(before_fit, after_fit)

        before_setitem = normalizer["action"]
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        after_setitem = normalizer["action"]
        self.assertIsNot(before_setitem, after_setitem)
        torch.testing.assert_close(
            after_setitem.params_dict["scale"], torch.tensor([1.0])
        )

    def test_replay_buffer_preserves_root_attrs_when_loading_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = zarr.open_group(Path(directory), mode="w")
            group.attrs["source"] = "algorithm-regression"
            meta = group.create_group("meta")
            meta.create_dataset("episode_ends", data=np.array([2], dtype=np.int64))
            data = group.create_group("data")
            data.create_dataset(
                "joint_state", data=np.zeros((2, 1), dtype=np.float32)
            )

            replay_buffer = ReplayBuffer.copy_from_path(directory)

        self.assertEqual(replay_buffer.attrs, {"source": "algorithm-regression"})

    def test_multi_task_dataset_rejects_invalid_constructor_invariants(self) -> None:
        with self.assertRaises(ValueError):
            MultiTaskDataset([], ["task"])
        with self.assertRaises(ValueError):
            MultiTaskDataset([object()], ["task"], task_texts=["one", "two"])
        with self.assertRaises(ValueError):
            MultiTaskDataset([object()], ["task"], sampling_strategy="unknown")
        with self.assertRaises(ValueError):
            MultiTaskDataset([object()], ["task"], normalizer_mode="unknown")

        for weights in ([0.0], [-1.0], [np.nan], [np.inf], [1.0, 1.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MultiTaskDataset(
                    [object()],
                    ["task"],
                    sampling_strategy="weighted",
                    task_weights=weights,
                )


if __name__ == "__main__":
    unittest.main()
