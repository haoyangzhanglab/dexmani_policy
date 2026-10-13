import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
from dexmani_policy.datasets.resumable_sampler import multitask_indices, ResumableDistributedSampler
from dexmani_policy.evaluation.protocol import select_eval_seeds, validate_heldout


def write_data(path, n=24, *, channels=6):
    root = zarr.open_group(str(path), mode='w')
    rng = np.random.default_rng(14)
    for key, shape in [('action', (n, 19)), ('joint_state', (n, 19)), ('point_cloud', (n, 8, channels))]:
        root.create_dataset('data/' + key, data=rng.normal(size=shape).astype('f4'), chunks=(min(8, n), *shape[1:]))
    root.create_dataset('meta/episode_ends', data=np.array([n], dtype=np.int64))
    root.attrs['data_revision'] = 'synthetic-test-v1'
    return root


@pytest.mark.parametrize('lengths', [[3, 2], [1, 1]])
def test_weighted_unbiased_reproducible_and_bounded(lengths):
    counts = []
    for epoch in range(10000):
        seq = multitask_indices(lengths, [.1, .9], 42, epoch, sampling_strategy='weighted')
        assert len(seq) == sum(lengths)
        assert all(0 <= i < sum(lengths) for i in seq)
        counts.append(sum(i < lengths[0] for i in seq))
    # 10,000 independent deterministic epoch seeds; tolerance > 5 standard errors.
    assert abs(sum(counts) / (10000 * sum(lengths)) - .1) < .01
    assert len(set(counts)) == 2
    assert multitask_indices(lengths, [.1, .9], 42, 123, sampling_strategy='weighted') == multitask_indices(
        lengths, [.1, .9], 42, 123, sampling_strategy='weighted')
    for epoch in range(50):
        seq = multitask_indices([1, 2, 1], [0, 1, 0], 42, epoch, sampling_strategy='weighted')
        assert set(seq) <= {1, 2}


def test_sampler_ddp_padding_cursor_and_normalization_source(tmp_path):
    children = []
    for i, length in enumerate([3, 2]):
        path = tmp_path / str(i)
        write_data(path, length)
        children.append(BaseDataset(str(path), horizon=1, sensor_modalities=['joint_state']))
    ds = MultiTaskDataset(children, ['a', 'b'], sampling_strategy='weighted', task_weights=[.1, .9])
    seqs = []
    for rank in range(2):
        sampler = ResumableDistributedSampler(ds, batch_size=1, num_replicas=2, rank=rank, seed=42)
        sampler.set_epoch(3)
        seq = list(sampler)
        assert len(seq) == 3  # ceil(5 / 2), including DistributedSampler padding.
        sampler.set_epoch(3, 1)
        assert list(sampler) == seq[1:]
        seqs.append(seq)
    all_rows = np.concatenate(list(ds.iter_normalization_data('action')))
    zero = MultiTaskDataset(children, ['a', 'b'], sampling_strategy='weighted', task_weights=[0, 1])
    np.testing.assert_array_equal(all_rows, np.concatenate(list(zero.iter_normalization_data('action'))))
    assert len(all_rows) == 5
    from omegaconf import OmegaConf
    from dexmani_policy.training.resume import build_train_loader
    cfg = OmegaConf.create({'dataloader': {'batch_size': 8, 'drop_last': True, 'num_workers': 0},
                            'training': {'seed': 42}})
    with pytest.raises(ValueError, match='DataLoader is empty'):
        build_train_loader(cfg, ds)


def pools(long=200, short=50):
    return SimpleNamespace(runners={
        'long': SimpleNamespace(task_name='long', get_seed_list=lambda: list(range(long)) + [0, 1]),
        'short': SimpleNamespace(task_name='short', get_seed_list=lambda: list(range(short))),
    })


def test_seed_pool_tail_exclusions_and_budget():
    runner = pools()
    seen = set()
    for seed in range(20):
        plan = select_eval_seeds(runner, seed, 100)
        assert len(plan['long']) == len(plan['short']) == 50
        assert plan == select_eval_seeds(runner, seed, 100)
        seen.update(plan['long'])
    assert len(seen) > 190 and max(seen) == 199
    excluded = {'long': list(range(180)), 'short': list(range(10))}
    plan = select_eval_seeds(runner, 42, 100, excluded)
    assert len(plan['long']) == len(plan['short']) == 20
    assert not set(plan['long']) & set(excluded['long'])
    assert not set(plan['short']) & set(excluded['short'])
    with pytest.raises(ValueError, match='tasks'):
        select_eval_seeds(runner, 42, 10, {'wrong': [1]})
    with pytest.raises(RuntimeError, match='No evaluation seeds'):
        select_eval_seeds(runner, 42, 10, {'long': list(range(200)), 'short': []})
    selection = select_eval_seeds(runner, 42, 20)
    test = select_eval_seeds(runner, 43, 100, selection)
    validate_heldout(runner, {'selection': {'task_seeds': selection}}, test)


def test_xyz_selection_precedes_finite_fit_and_model_input(tmp_path):
    from dexmani_policy.training.build_utils import build_normalizer
    path = tmp_path / 'data'
    root = write_data(path)
    root['data/point_cloud'][0, :, 3:] = np.nan
    kwargs = dict(zarr_path=str(path), horizon=4, obs_horizon=2,
                  sensor_modalities=['joint_state', 'point_cloud'], pad_before=1, pad_after=1)
    xyz, color = BaseDataset(**kwargs, pc_dim=3), BaseDataset(**kwargs, pc_dim=6)
    assert xyz._obs_valid[0] and not color._obs_valid[0]
    assert xyz[0]['obs']['point_cloud'].shape[-1] == 3
    stats = build_normalizer(xyz, {'point_cloud': 'limits'}, 'action')
    assert stats['point_cloud'].params_dict['scale'].shape == (3,)
    assert torch.isfinite(stats['point_cloud'].params_dict['scale']).all()
    assert xyz.data_recipe['window_validity'] != color.data_recipe['window_validity']
    # Sim's producer can still send six channels; select before affine stats.
    from dexmani_policy.agents.core.base import BaseAgent
    agent = BaseAgent(torch.nn.Identity(), torch.nn.Identity(), 4, 2, 2, 19)
    agent.normalizer = stats
    agent.normalization_dims = {'point_cloud': 3}
    cloud = torch.as_tensor(root['data/point_cloud'][:3]).unsqueeze(0)
    preprocessed = agent.preprocess({'point_cloud': cloud})['point_cloud']
    assert preprocessed.shape == (2, 8, 3)
    torch.testing.assert_close(preprocessed, stats['point_cloud'].normalize(cloud[:, :2, :, :3]).flatten(0, 1))
    root['data/point_cloud'][1, 0, 0] = np.inf
    assert not BaseDataset(**kwargs, pc_dim=3)._obs_valid[1]
    with pytest.raises(ValueError, match='color augmentation'):
        BaseDataset(**kwargs, pc_dim=3, augmentation_cfg={'pc': {'color_noise': {'prob': 1}}})
    incomplete = BaseDataset(str(path), sensor_modalities=['joint_state'])
    with pytest.raises(ValueError, match='complete observation'):
        MultiTaskDataset([xyz, incomplete], ['x', 'y'])


def test_ejc_construct_and_loaded_mapping():
    from dexmani_policy.agents.action_decoders.backbone.sat import EmbodiedJointCodebook
    with pytest.raises(ValueError, match='num_functions'):
        EmbodiedJointCodebook(19, 16, num_functions=2)
    model = EmbodiedJointCodebook(19, 16)
    assert model().shape == (19, 16)
    for value in (torch.zeros(18, dtype=torch.long), torch.zeros(19), torch.full((19,), 19),
                  torch.full((19,), -1), torch.zeros(19, dtype=torch.bool)):
        state = copy.deepcopy(model.state_dict())
        state['joint_function'] = value
        with pytest.raises(ValueError, match='EJC mapping'):
            model.load_state_dict(state)


@pytest.mark.parametrize('norm', ['batch_norm', 'frozen_bn', 'group_norm'])
@pytest.mark.parametrize('tune', ['freeze', 'full'])
def test_rgb_modes_and_full_state_restore(norm, tune):
    from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNet
    from dexmani_policy.agents.obs_encoder.rgb.r3m import R3M
    from dexmani_policy.training.build_utils import _validate_ema_batchnorm_compatibility
    assert ResNet().tune_mode == 'full'
    for cls in (ResNet, R3M):
        model = cls(norm_mode=norm, tune_mode=tune, load_pretrained=False)
        restored = cls(norm_mode=norm, tune_mode=tune, load_pretrained=False)
        restored.load_state_dict(model.state_dict())
        model.eval(); restored.eval()
        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(x)['global_token'], restored(x)['global_token'])
        running = {name: value.clone() for name, value in model.named_buffers() if 'running_' in name}
        parent = torch.nn.Sequential(model)
        parent.train()
        if tune == 'freeze':
            for mode in (True, False, True):
                parent.train(mode)
                assert not model.backbone.training
                with torch.no_grad():
                    torch.testing.assert_close(parent(x)['global_token'], restored(x)['global_token'])
                for name, value in model.named_buffers():
                    if name in running:
                        torch.testing.assert_close(value, running[name], rtol=0, atol=0)
        if tune == 'full' and norm == 'batch_norm':
            with pytest.raises(ValueError, match='BatchNorm'):
                _validate_ema_batchnorm_compatibility(model)
        else:
            _validate_ema_batchnorm_compatibility(model)
    with pytest.raises(ValueError, match='Frozen pretrained R3M'):
        R3M(norm_mode='group_norm', tune_mode='freeze')
