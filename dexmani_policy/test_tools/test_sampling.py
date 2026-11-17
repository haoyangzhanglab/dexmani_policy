import numpy as np
import pytest
import zarr

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.variants.ode import ODEDataset
from dexmani_policy.datasets.variants.tactile import AdapTacDataset


@pytest.fixture
def segments(tmp_path):
    path = tmp_path / "segments.zarr"
    group = zarr.open_group(str(path), mode="w")
    rows = np.arange(12, dtype=np.float32)
    fields = {
        "action": np.broadcast_to(rows[:, None] + 100, (12, 19)).copy(),
        "joint_state": np.broadcast_to(rows[:, None] + 200, (12, 19)).copy(),
        "point_cloud": np.arange(12 * 2 * 6, dtype=np.float64).reshape(12, 2, 6),
        "finger_tactile_force": np.broadcast_to(rows[:, None, None], (12, 5, 3)).copy(),
        "rgb": np.arange(12 * 2 * 3 * 3, dtype=np.uint8).reshape(12, 2, 3, 3),
    }
    for key, value in fields.items():
        group.create_dataset(f"data/{key}", data=value, chunks=(2, *value.shape[1:]))
    group.create_dataset("meta/episode_ends", data=np.array([1, 3, 8, 12]))
    return path, fields


@pytest.mark.parametrize("horizon,before,after", [(1, 0, 0), (4, 0, 0), (4, 3, 3), (6, 2, 5)])
@pytest.mark.parametrize("obs_horizon", [1, 2])
def test_window_boundaries_and_short_observations(segments, horizon, before, after, obs_horizon):
    path, fields = segments
    obs_horizon = min(obs_horizon, horizon)
    store = EpisodeStore(path)
    mask = np.array([True, False, True, True])
    sampler = SequenceSampler(
        store, horizon, before, after, episode_mask=mask,
        fields={"action": None, "point_cloud": slice(0, 3), "rgb": None},
        field_horizons={"point_cloud": obs_horizon, "rgb": obs_horizon},
    )
    expected_rows = []
    for start, end, selected in zip([0, 1, 3, 8], [1, 3, 8, 12], mask):
        if selected:
            for offset in range(-before, end - start - horizon + after + 1):
                expected_rows.append([min(max(start + offset + t, start), end - 1) for t in range(horizon)])
    assert len(sampler) == len(expected_rows)
    for index, rows in enumerate(expected_rows):
        sample = sampler.sample_sequence(index)
        np.testing.assert_array_equal(sampler.sample_source_rows(index), rows)
        np.testing.assert_array_equal(sample["action"], fields["action"][rows])
        np.testing.assert_array_equal(sample["point_cloud"], fields["point_cloud"][rows[:obs_horizon], :, :3])
        np.testing.assert_array_equal(sample["rgb"], fields["rgb"][rows[:obs_horizon]])
        assert sample["point_cloud"].dtype == np.float32
        assert sample["rgb"].dtype == np.uint8
        sample["action"][:] = -1
        np.testing.assert_array_equal(sampler.sample_sequence(index)["action"], fields["action"][rows])


def test_real_sources_split_together_and_normalize_train_only(segments):
    path, fields = segments
    group = zarr.open_group(str(path), mode="a")
    group.attrs.update({
        "format": "dexmani.real.canonical",
        "source_episodes": ["a", "b", "a", "c"],
        "source_row_ranges": [[0, 1], [0, 2], [3, 8], [0, 4]],
    })
    dataset = DemonstrationDataset(
        path, sensor_modalities=["joint_state"], obs_horizon=1, horizon=3,
        pad_before=1, pad_after=2, val_episodes=1, max_train_episodes=1,
    )
    train, val = dataset.train_mask, dataset.val_mask
    assert train[0] == train[2] and val[0] == val[2]
    assert not (train & val).any()
    assert dataset.episode_store.count_groups(train) == dataset.episode_store.count_groups(val) == 1
    selected_rows = [row for start, end, selected in zip([0, 1, 3, 8], [1, 3, 8, 12], train)
                     if selected for row in range(start, end)]
    expected = fields["action"][selected_rows]
    chunks = list(dataset.iter_normalization_data("action"))
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    normalizer = LinearNormalizer()
    normalizer.fit_field("action", chunks)
    np.testing.assert_allclose(normalizer["action"].normalize(expected).amin(0), -1, atol=1e-6)
    np.testing.assert_allclose(normalizer["action"].normalize(expected).amax(0), 1, atol=1e-6)
    for index in range(len(dataset)):
        rows = dataset.sampler.sample_source_rows(index)
        episodes = np.searchsorted([1, 3, 8, 12], rows, side="right")
        assert np.all(episodes == episodes[0]) and train[episodes[0]]


def test_ode_tactile_and_afro_temporal_targets(segments):
    path, fields = segments
    group = zarr.open_group(str(path), mode="a")
    group["meta/episode_ends"][:] = [3, 6, 9, 12]
    options = dict(horizon=4, obs_horizon=2, pad_before=1, pad_after=3)
    ode = ODEDataset(path, sensor_modalities=["joint_state"], **options)
    tactile = AdapTacDataset(path, sensor_modalities=["finger_tactile_force"], **options)
    afro = AFROPretrainDataset(path, frame_stride=2, horizon=4, pad_before=1, pad_after=3)
    assert len(ode) == len(afro) == 16
    assert len(tactile) == 8
    for episode in range(4):
        start = episode * 3
        for offset, query in enumerate([start, start + 1, start + 2, start + 2]):
            expected = fields["joint_state"][query] if query == start else fields["action"][query - 1]
            np.testing.assert_array_equal(ode[episode * 4 + offset]["prev_action"], expected)
        for offset in range(2):
            sample = tactile[episode * 2 + offset]
            expected = [start + 1, start + 2, start + 2] if offset == 0 else [start + 2] * 3
            np.testing.assert_array_equal(sample["future_force"][:, 0, 0], expected)
            np.testing.assert_array_equal(sample["future_force_valid"], [True, offset == 0, False])
        for offset, rows in enumerate([[start, start + 1], [start, start + 2], [start + 1, start + 2], [start + 2] * 2]):
            np.testing.assert_array_equal(afro[episode * 4 + offset]["obs"]["point_cloud"], fields["point_cloud"][rows, :, :3])
