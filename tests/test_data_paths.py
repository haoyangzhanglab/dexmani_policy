import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.datasets.common.sampler import SequenceSampler, create_sequence_indices
from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
from dexmani_policy.training.common.data import build_dataset_and_normalizer, build_normalizer
from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_policy_data


def expected_windows(ends, length, before, after, mask):
    rows = []
    start = 0
    for end, enabled in zip(ends, mask):
        if enabled:
            for offset in range(-before, end - start - length + after + 1):
                rows.append(np.clip(start + offset + np.arange(length), start, end - 1))
        start = end
    return np.asarray(rows, dtype=np.int64).reshape(-1, length)


class DataPathsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / "data.zarr")
        self.ends = np.array([2, 14, 26])
        rng = np.random.default_rng(5)
        self.action = rng.normal(size=(26, 19)).astype(np.float32)
        self.state = rng.normal(size=(26, 19)).astype(np.float32)
        self.action[:14, 7] = 3.0
        self.action[:14, 8] = np.linspace(1.0, 1.00005, 14)
        self.action[14:, 7:] += 100
        self.action[5, 0] = np.nan
        self.state[8, 0] = np.nan
        group = zarr.open_group(self.path, mode="w")
        data = group.create_group("data")
        for key, value in (("action", self.action), ("joint_state", self.state)):
            data.create_dataset(key, data=value, chunks=(4, 19))
        group.create_group("meta").create_dataset("episode_ends", data=self.ends)
        group.attrs.update(dt=0.0625, source_id="test")
        self.manifest = dict(
            source_id="test", zarr_path=self.path, episode_ids=None, trial_ids=None,
            episode_ends=self.ends.tolist(), train_indices=[0, 1], val_indices=[2], exclusions=[],
        )
        self.options = dict(
            zarr_path=self.path, sensor_modalities=["joint_state"], horizon=3,
            obs_horizon=2, pad_before=1, split_manifest=self.manifest,
        )

    def test_sampler_padding_filtering_and_populations(self):
        buffer = ReplayBuffer(self.path)
        mask = np.array([True, True, False])
        obs_valid = np.isfinite(self.state).all(axis=1)
        action_valid = np.isfinite(self.action).all(axis=1)
        for length, before, after, offset in ((3, 0, 0, 0), (3, 1, 2, 0), (4, 1, 0, 1)):
            with self.subTest(length=length, before=before, after=after):
                sampler = SequenceSampler(buffer, length, before, after, mask)
                rows = expected_windows(self.ends, length, before, after, mask)
                self.assertEqual(sampler.indices.shape, (len(rows), 3))
                np.testing.assert_array_equal(sampler.source_rows(), rows)
                obs_ok = obs_valid[rows[:, :2]].all(axis=1)
                action_ok = action_valid[rows[:, offset:]].all(axis=1)
                kept = rows[obs_ok & action_ok]
                sampler.filter_valid(obs_valid, action_valid, 2, action_offset=offset)
                np.testing.assert_array_equal(sampler.source_rows(), kept)
                np.testing.assert_array_equal(sampler.observation_source_rows, np.unique(kept[:, :2]))
                np.testing.assert_array_equal(sampler.action_source_rows, np.unique(kept[:, offset:]))
                self.assertEqual(sampler.validity_summary, dict(
                    candidate=len(rows), observation=int((~obs_ok).sum()),
                    action=int((~action_ok).sum()), valid=len(kept),
                ))
                for index, row in enumerate(kept):
                    sample = sampler.sample_sequence(
                        index, key_lengths={"joint_state": 2, "action": length - offset},
                        key_offsets={"action": offset},
                    )
                    np.testing.assert_array_equal(sample["joint_state"], self.state[row[:2]])
                    np.testing.assert_array_equal(sample["action"], self.action[row[offset:]])
        short = SequenceSampler(buffer, 3, 1, 2, np.array([True, False, False]))
        np.testing.assert_array_equal(short.source_rows(), [[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        empty = create_sequence_indices(self.ends, 40, mask)
        self.assertEqual(empty.shape, (0, 3))
        self.assertTrue(create_sequence_indices.nopython_signatures)
        with self.assertRaisesRegex(ValueError, "No sequence windows"):
            SequenceSampler(buffer, 40)
        with self.assertRaisesRegex(ValueError, "Zero valid"):
            short.filter_valid(np.zeros(26, dtype=bool), action_valid, 2)

    def test_hand_prior_consumers_preserve_future_and_history_rows(self):
        chunks = HandChunkDataset(
            self.path, data_semantics=HAND_ACTION_SEMANTICS, horizon=3,
            split_manifest=self.manifest,
        )
        for split, start, end in ((chunks, 2, 14), (chunks.get_validation_dataset(), 14, 26)):
            rows = np.arange(start, end - 2)[:, None] + np.arange(3)
            actions, reference = split.raw_hand_batch(np.arange(len(split)))
            np.testing.assert_array_equal(actions, self.action[rows, 7:])
            np.testing.assert_array_equal(reference, self.state[rows[:, 0], 7:])

        lamp = LAMPDataset(
            self.path, sensor_modalities=["joint_state"], horizon=1, obs_horizon=1,
            history_steps=3, split_manifest=self.manifest,
        )
        for split in (lamp, lamp.get_validation_dataset()):
            for index in range(len(split)):
                query = split.sampler.sample_source_rows(index)[0]
                episode = np.searchsorted(self.ends, query, side="right")
                start = self.ends[episode - 1] if episode else 0
                history = np.stack([
                    self.action[row, 7:] if row >= start else self.state[start, 7:]
                    for row in range(query - 3, query)
                ])
                sample = split[index]
                np.testing.assert_array_equal(sample["action_context"]["hand_history"], history)
                np.testing.assert_array_equal(sample["action"], self.action[query : query + 1])

    def test_latent_flow_lifetime_windows_and_fresh_statistics(self):
        read = ReplayBuffer.read
        full_reads, arrays = [], []

        def track(buffer, key, rows, **kwargs):
            value = read(buffer, key, rows, **kwargs)
            if rows == slice(None):
                full_reads.append(key)
                arrays.append(weakref.ref(value))
            return value

        with patch.object(ReplayBuffer, "read", track):
            dataset = LatentFlowDataset(**self.options, data_semantics=HAND_ACTION_SEMANTICS)
            validation = dataset.get_validation_dataset()
            self.assertEqual(full_reads, [])
            expected = []
            for query in range(2, 14 - 3 + 1):
                history = [max(2, query - 1), query]
                future = np.arange(query, query + 3)
                if np.isfinite(self.state[history]).all() and np.isfinite(self.action[future]).all():
                    expected.append((history, future))
            self.assertEqual(len(dataset), len(expected))
            relative = []
            for i, (history, future) in enumerate(expected):
                sample = dataset[i]
                np.testing.assert_array_equal(sample["obs"]["joint_state"], self.state[history])
                np.testing.assert_array_equal(sample["action"], self.action[future])
                np.testing.assert_array_equal(sample["action_context"]["reference_state"], self.state[future[0]])
                relative.append(self.action[future] - self.state[future[0]])
            old = LinearNormalizer()
            old.fit_field("action", [np.concatenate(relative)], mode="limits")
            new = build_normalizer(dataset, {"action": "limits", "joint_state": "gaussian"}, "action")
            self.assertEqual(full_reads, ["action", "joint_state"])
            for name, value in old.state_dict().items():
                torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
            for name, value in old["action"].input_stats.items():
                torch.testing.assert_close(new["action"].input_stats[name], value)
            obs = LinearNormalizer()
            obs.fit_field("joint_state", [self.state[dataset.sampler.observation_source_rows]], mode="gaussian")
            torch.testing.assert_close(new["joint_state"].scale, obs["joint_state"].scale)
            self.assertTrue(all(ref() is None for ref in arrays))
            for split in (dataset, validation):
                self.assertFalse(hasattr(split, "commands"))
                self.assertFalse(hasattr(split, "states"))
            with self.assertRaisesRegex(ValueError, "reuse the training normalizer"):
                next(validation.iter_normalization_data("action"))

    def test_latent_flow_resume_skips_full_reads_and_fit(self):
        cfg = OmegaConf.create(dict(
            dataset=dict(self.options, data_semantics=HAND_ACTION_SEMANTICS,
                         _target_="dexmani_policy.datasets.hand_prior.latent_flow_dataset.LatentFlowDataset"),
            agent=dict(_target_="unused", action_dim=19), task_name="test", action_key="action",
            normalization={"action": "limits", "joint_state": "gaussian"}, resume_from="saved.pt",
        ))
        checkpoint = SimpleNamespace(effective_config={"data_recipe": [{"split_manifest": self.manifest}]})
        agent_class = SimpleNamespace(requires_split_manifest=True, requires_hand_pretraining=True)
        read = ReplayBuffer.read

        def reject_full(buffer, key, rows, **kwargs):
            self.assertNotEqual(rows, slice(None))
            return read(buffer, key, rows, **kwargs)

        with patch.object(ReplayBuffer, "read", reject_full), patch(
            "hydra.utils.get_class", return_value=agent_class
        ), patch("dexmani_policy.training.common.data.build_normalizer") as fit:
            dataset, normalizer, _ = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint, verbose=False)
        fit.assert_not_called()
        self.assertIsNone(normalizer)
        self.assertIsNotNone(dataset.get_validation_dataset())

    def test_vq_reuses_qualified_hand_rows_and_matches_sliced_normalizer(self):
        dataset = RobotDataset(**self.options, pad_after=2)
        validation = dataset.get_validation_dataset()
        old = build_normalizer(dataset, {"action": "auto"}, "action")["action"]
        rows = dataset.sampler.action_source_rows
        val_rows = validation.sampler.action_source_rows
        self.assertNotIn(5, rows)
        self.assertFalse(set(rows) & set(val_rows))
        read = ReplayBuffer.read
        reads = []

        def track(buffer, key, selection, **kwargs):
            reads.append((key, selection, kwargs.get("columns")))
            return read(buffer, key, selection, **kwargs)

        for mode in ("auto", "limits"):
            cfg = OmegaConf.create(dict(
                normalization={"action": mode, "joint_state": "limits"},
                action_key="action", agent={"tcp_dim": 7, "action_dim": 19}, dataset=self.options,
            ))
            reads.clear()
            with patch("dexmani_policy.training.pretrain.hand_prior.vqvae.build_policy_dataset", return_value=dataset), patch.object(ReplayBuffer, "read", track):
                train, val, normalizer, metadata = prepare_policy_data(cfg)
            self.assertEqual(len(reads), sum(len(np.unique(r // 4)) for r in (rows, val_rows)))
            self.assertTrue(all(key == "action" and columns == slice(7, 19) for key, _, columns in reads))
            hand = normalizer["hand"]
            torch.testing.assert_close(hand.scale, old.scale[7:], rtol=0, atol=0)
            torch.testing.assert_close(hand.offset, old.offset[7:], rtol=0, atol=0)
            for name, value in old.input_stats.items():
                torch.testing.assert_close(hand.input_stats[name], value[7:])
            np.testing.assert_array_equal(train, old.normalize(self.action[rows]).numpy()[:, 7:])
            np.testing.assert_array_equal(val, old.normalize(self.action[val_rows]).numpy()[:, 7:])
            self.assertTrue((val[:, :2] > 1).all())
            torch.testing.assert_close(hand.scale[:2], torch.ones(2))
            self.assertEqual(metadata["split_manifest"], self.manifest)
            self.assertEqual(metadata["train_windows"], dataset.sampler.validity_summary)
            self.assertEqual(metadata["val_windows"], validation.sampler.validity_summary)
            self.assertEqual(set(normalizer.state_dict()), {"fields.hand.scale", "fields.hand.offset"})
            restored = LinearNormalizer({"hand": 12})
            restored.load_state_dict(normalizer.state_dict(), strict=True)
            np.testing.assert_array_equal(restored["hand"].normalize(self.action[rows, 7:]), train)
            np.testing.assert_array_equal(restored["hand"].normalize(self.action[val_rows, 7:]), val)
        cfg.normalization.action = "gaussian"
        with self.assertRaisesRegex(ValueError, "auto or limits"):
            prepare_policy_data(cfg)


if __name__ == "__main__":
    unittest.main()
