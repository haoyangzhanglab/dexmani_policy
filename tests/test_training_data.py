import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.aux.adaptac_dataset import AdapTacDataset
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.datasets.obs.afro_dataset import AFRODataset
from dexmani_policy.datasets.preprocess import contact_map


def write_zarr(path, fields, ends):
    group = zarr.open_group(str(path), mode='w')
    group.create_group('meta').create_dataset('episode_ends', data=np.array(ends))
    data = group.create_group('data')
    for key, values in fields.items():
        data.create_dataset(key, data=values, chunks=(3, *values.shape[1:]))
    return group


@pytest.mark.parametrize('stride', [1, 2, 4])
@pytest.mark.parametrize('padding', [(0, 2), (3, 3)])
def test_afro_reads_sampled_xyz_and_full_training_stats(tmp_path, stride, padding):
    points = np.arange(22 * 4 * 6, dtype=np.float32).reshape(22, 4, 6)
    write_zarr(tmp_path / 'data', {'point_cloud': points}, [10, 12, 22])
    manifest = {'source_id': None, 'episode_ids': None, 'episode_ends': [10, 12, 22],
                'train_indices': [0, 1], 'val_indices': [2]}
    dataset = AFRODataset(tmp_path / 'data', frame_stride=stride, horizon=5,
                          pad_before=padding[0], pad_after=padding[1], split_manifest=manifest)
    for view in (dataset, dataset.get_validation_dataset()):
        assert len(view) == len(view.sampler)
        for index in range(len(view)):
            old = view.sampler.sample_sequence(index)['point_cloud'][::stride, ..., :3]
            np.testing.assert_array_equal(view[index]['obs']['point_cloud'], old)
    expected = LinearNormalizer()
    expected.fit_field('point_cloud', [points[:12, ..., :3]])
    actual = dataset.get_normalizer({'point_cloud': 'limits'})
    torch.testing.assert_close(actual.state_dict(), expected.state_dict(), rtol=0, atol=0)
    assert dataset.get_validation_dataset().get_normalizer({'point_cloud': 'limits'}) is actual
    with pytest.raises(ValueError, match='at least two'):
        AFRODataset(tmp_path / 'data', frame_stride=5, horizon=5)


@pytest.mark.parametrize('mode', ['limits', 'gaussian'])
@pytest.mark.parametrize('feature_dims', [1, 2])
def test_normalizer_reference_and_strict_state(mode, feature_dims):
    values = torch.randn(31, 5, 3, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    values[..., 0] = 7
    values[..., 1] = values[..., 1] * 1e-6 + 2
    normalizer = LinearNormalizer()
    normalizer.fit_field('x', [values[:0], *values.split(4)], mode=mode, last_n_dims=feature_dims)
    shape = values.shape[-feature_dims:]
    flattened = values.reshape(-1, shape.numel())
    if mode == 'limits':
        low, high = flattened.amin(0).float(), flattened.amax(0).float()
        span = high - low
        constant = span < 1e-4
        scale = 2 / span.masked_fill(constant, 2)
        offset = -1 - scale * low
        offset[constant] = -low[constant]
    else:
        std, mean = flattened.std(0).float(), flattened.mean(0).float()
        scale = std.masked_fill(std < 1e-4, 1).reciprocal()
        offset = -mean * scale
    old_state = {'fields.x.scale': scale.reshape(shape), 'fields.x.offset': offset.reshape(shape)}
    torch.testing.assert_close(normalizer.state_dict(), old_state)
    restored = LinearNormalizer({'x': shape})
    restored.load_state_dict(old_state, strict=True)
    torch.testing.assert_close(normalizer['x'].normalize(values), restored['x'].normalize(values))
    torch.testing.assert_close(normalizer['x'].unnormalize(normalizer['x'].normalize(values)), values.float())
    assert not hasattr(normalizer['x'], 'input_stats')


def test_dataset_affine_special_cases(tmp_path):
    rng = np.random.default_rng(42)
    action = rng.normal(size=(8, 21)).astype(np.float32)
    tactile = rng.normal(size=(8, 5, 3)).astype(np.float32)
    write_zarr(tmp_path / 'data', {'action_ee': action, 'finger_tactile_force': tactile}, [8])
    robot = RobotDataset(tmp_path / 'data', sensor_modalities=[], action_key='action_ee')
    field = robot.get_normalizer({'action': 'auto'})['action']
    torch.testing.assert_close(field.scale[3:9], torch.ones(6))
    torch.testing.assert_close(field.offset[3:9], torch.zeros(6))
    adaptac = AdapTacDataset(tmp_path / 'data', sensor_modalities=['finger_tactile_force'], action_key='action_ee')
    normalizer = adaptac.get_normalizer({'action': 'identity', 'finger_tactile_force': 'limits'})
    expected = LinearNormalizer()
    expected.fit_field('force', [tactile])
    assert 'action' not in normalizer.fields
    torch.testing.assert_close(normalizer['finger_tactile_force'].scale, expected['force'].scale.expand(5, 3))
    torch.testing.assert_close(normalizer['finger_tactile_force'].offset, expected['force'].offset.expand(5, 3))


@pytest.mark.parametrize('scale', [1e-6, 0.01, 1, 1e3])
@pytest.mark.parametrize('chunk', [1, 128, 129, 1024])
def test_contact_formula(scale, chunk):
    rng = np.random.default_rng(42)
    objects, hand = rng.normal(size=(257, 3)) * scale, rng.normal(size=(31, 3)) * scale
    hand[0] = objects[0]
    normals = rng.normal(size=objects.shape)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    difference = hand[None] - objects[:, None]
    distance = np.linalg.norm(difference, axis=-1)
    direction = difference / (distance[..., None] + 1e-8)
    alignment = np.abs(np.einsum('ijk,ik->ij', direction, normals))
    minimum = (distance * np.exp(1 - alignment)).min(axis=1)
    old = 1 - 2 * (1 / (1 + np.exp(-10 * minimum)) - 0.5)
    np.testing.assert_allclose(contact_map.compute_contact_map(objects, normals, hand, point_chunk_size=chunk), old, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('key,shape', [('object_pointcloud', (2, 32, 3)), ('joint_state', (2, 18)), ('hand_pointcloud', (2, 0, 3))])
def test_contact_shapes_fail_before_output(tmp_path, monkeypatch, key, shape):
    fields = {'object_pointcloud': np.zeros((2, 1024, 3)), 'hand_pointcloud': np.zeros((2, 5, 3)), 'joint_state': np.zeros((2, 19))}
    fields[key] = np.zeros(shape)
    # Empty point dimensions need nonzero chunk dimensions.
    group = zarr.open_group(str(tmp_path / 'source'), mode='w')
    group.create_group('meta').create_dataset('episode_ends', data=np.array([2]))
    data = group.create_group('data')
    for name, values in fields.items():
        data.create_dataset(name, data=values, chunks=tuple(max(1, d) for d in values.shape))
    monkeypatch.setattr(contact_map, 'estimate_normals', lambda *a: pytest.fail('expensive computation started'))
    output = tmp_path / 'new_parent/output'
    with pytest.raises(ValueError):
        contact_map.generate(tmp_path / 'source', output)
    assert not output.parent.exists()
