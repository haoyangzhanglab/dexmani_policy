import numpy as np
import pytest
import zarr

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler


def test_calibrated_dataset_rejects_empty_xyz_in_observation_window(make_demo):
    path = make_demo()
    data = zarr.open_group(str(path), mode="r+")["data"]
    frames = len(data["action"])
    data["rgb"] = np.full((frames, 2, 2, 3), 128, dtype=np.uint8)
    data["camera_intrinsic"] = np.tile(np.eye(3, dtype=np.float32).ravel(), (frames, 1))
    data["camera_extrinsic"] = np.tile(np.eye(4, dtype=np.float32)[:3].ravel(), (frames, 1))
    data["point_cloud"][2, :, :3] = 0
    data["point_cloud"][2, :, 3:] = 1

    dataset = DemonstrationDataset(
        path, sensor_modalities=["point_cloud", "rgb", "camera_intrinsic", "camera_extrinsic"],
        horizon=2, obs_horizon=2,
    )
    assert dataset[0]["obs"]["point_cloud"][..., :3].any()
    with pytest.raises(ValueError, match=r"sample 1.*empty frame at index 1") as error:
        dataset.__getitems__([0, 1])
    assert str(path) in str(error.value)

    points_only = DemonstrationDataset(path, sensor_modalities=["point_cloud"], horizon=2, obs_horizon=2)
    assert not points_only[1]["obs"]["point_cloud"][-1, :, :3].any()


@pytest.mark.parametrize("horizon,before,after", [(1, 0, 0), (4, 0, 0), (4, 3, 3), (6, 2, 5)])
@pytest.mark.parametrize("obs_horizon", [1, 2])
def test_window_boundaries_and_short_observations(segments, horizon, before, after, obs_horizon):
    path, fields = segments
    obs_horizon = min(obs_horizon, horizon)
    store = EpisodeStore(path, keys=("action", "point_cloud", "rgb"))
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
    indices = [*reversed(range(len(expected_rows))), 0, len(expected_rows) - 1]
    for index, sample in zip(indices, sampler.sample_batch(indices), strict=True):
        rows = expected_rows[index]
        np.testing.assert_array_equal(sample["action"], fields["action"][rows])
        np.testing.assert_array_equal(sample["point_cloud"], fields["point_cloud"][rows[:obs_horizon], :, :3])
        np.testing.assert_array_equal(sample["rgb"], fields["rgb"][rows[:obs_horizon]])
        assert sample["point_cloud"].dtype == np.float32
        assert sample["rgb"].dtype == np.uint8
        sample["action"][:] = -1


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
    states = fields["joint_state"][selected_rows]
    normalizer.fit_field("joint_state", dataset.iter_normalization_data("joint_state"), mode="mean_std")
    np.testing.assert_allclose(normalizer["joint_state"].normalize(states),
                               (states - states.mean(0)) / states.std(0), atol=2e-5)
    for index in range(len(dataset)):
        rows = dataset.sampler.source_rows(index, np.arange(dataset.horizon))
        episodes = np.searchsorted([1, 3, 8, 12], rows, side="right")
        assert np.all(episodes == episodes[0]) and train[episodes[0]]


@pytest.mark.parametrize("key,feature_shape", [("point_cloud", (3,)), ("fingertip_xyz", (5, 3))])
def test_streaming_mean_std(key, feature_shape):
    values = np.random.default_rng(5).normal(size=(11, 5, 3))
    values[..., 0] *= 1e-6
    values[..., 1] = 5
    points = values.reshape(-1, *feature_shape)
    mean, std = points.mean(axis=0), points.std(axis=0)
    divisor = np.where(std < 1e-4, 1, std)
    normalizer = LinearNormalizer()
    normalizer.fit_field(key, np.split(values, [1, 1, 4]), mode="mean_std")
    field = normalizer[key]
    np.testing.assert_allclose(field.scale, 1 / divisor, rtol=1e-6)
    np.testing.assert_allclose(field.offset, -mean / divisor, rtol=1e-6)
    np.testing.assert_allclose(field.normalize(values), (values - mean) / divisor, atol=1e-6)
