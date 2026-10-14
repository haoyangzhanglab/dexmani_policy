import copy
import io
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


@pytest.mark.parametrize('key', ['color', 'color_noise'])
@pytest.mark.parametrize('color_cfg', [None, {}, {'prob': 0}, {'prob': .5}, {'prob': 1}, {'hue': 0}])
def test_xyz_color_config_matches_dataset(tmp_path, key, color_cfg):
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import validate_config

    # A nonempty configuration without prob defaults to enabled.
    if color_cfg == {'hue': 0} and key == 'color_noise':
        color_cfg = {'noise_std': .01}
    cfg = load_config('dp3')
    cfg.agent.pc_dim = cfg.dataset.pc_dim = 3
    cfg.dataset.augmentation_cfg = {'pc': {key: color_cfg}}
    path = tmp_path / 'data'
    write_data(path)

    def dataset():
        return BaseDataset(str(path), pc_dim=3, sensor_modalities=['joint_state', 'point_cloud'],
                           augmentation_cfg=cfg.dataset.augmentation_cfg)

    if color_cfg and color_cfg.get('prob', 1) > 0:
        with pytest.raises(ValueError, match='color augmentation'):
            validate_config(cfg)
        with pytest.raises(ValueError, match='color augmentation'):
            dataset()
    else:
        validate_config(cfg)
        ds = dataset()
        assert 'point_cloud' not in ds.augmentors
        assert ds[0]['obs']['point_cloud'].shape[-1] == 3


@pytest.mark.parametrize('augmentation_cfg', [None, {}, {'pc': None}, {'pc': {}}])
def test_xyz_empty_augmentation_config(tmp_path, augmentation_cfg):
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import validate_config
    cfg = load_config('dp3')
    cfg.agent.pc_dim = cfg.dataset.pc_dim = 3
    cfg.dataset.augmentation_cfg = augmentation_cfg
    validate_config(cfg)
    path = tmp_path / 'data'
    write_data(path)
    ds = BaseDataset(str(path), pc_dim=3, sensor_modalities=['point_cloud'],
                     augmentation_cfg=augmentation_cfg)
    assert ds.augmentors == {}
    assert ds[0]['obs']['point_cloud'].shape[-1] == 3


def test_multitask_color_config_and_unspecified_channels(tmp_path):
    from omegaconf import OmegaConf
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import validate_config
    cfg = load_config('dp3')
    cfg.agent.pc_dim = cfg.dataset.pc_dim = 3
    child = OmegaConf.to_container(cfg.dataset, resolve=True)
    child['augmentation_cfg'] = {'pc': {'color_noise': {'prob': 0}}}
    cfg.dataset = {'datasets': [copy.deepcopy(child), copy.deepcopy(child)], 'augmentation_cfg': {}}
    validate_config(cfg)
    cfg.dataset.datasets[1].augmentation_cfg.pc.color_noise.prob = 1
    with pytest.raises(ValueError, match='color augmentation'):
        validate_config(cfg)
    cfg.dataset.datasets[1].pc_dim = None
    with pytest.raises(ValueError, match='color augmentation'):
        validate_config(cfg)  # The explicit Agent XYZ width still applies.
    cfg.dataset.datasets[1].augmentation_cfg.pc.color_noise.prob = 0
    cfg.dataset.datasets[1].pc_dim = 6
    with pytest.raises(ValueError, match='Dataset pc_dim must match'):
        validate_config(cfg)
    path = tmp_path / 'data'
    write_data(path)
    ds = BaseDataset(str(path), pc_dim=None, sensor_modalities=['point_cloud'])
    assert ds.pc_dim is None and ds[0]['obs']['point_cloud'].shape[-1] == 6


@pytest.mark.parametrize('augmentation_cfg', [
    {}, {'pc': None}, {'pc': {'color_noise': {'prob': 0}}}, {'pc': {'color_noise': {'prob': 1}}},
])
def test_multitask_shared_color_config_matches_construction(tmp_path, augmentation_cfg):
    import hydra
    from omegaconf import OmegaConf
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import validate_config

    path = tmp_path / 'data'
    write_data(path)
    cfg = load_config('dp3')
    cfg.agent.pc_dim = cfg.dataset.pc_dim = 3
    cfg.dataset.zarr_path = str(path)
    cfg.dataset.augmentation_cfg = None
    child = OmegaConf.to_container(cfg.dataset, resolve=True)
    cfg.dataset = {'_target_': 'dexmani_policy.datasets.multi_task_dataset.MultiTaskDataset',
                   'datasets': [child, copy.deepcopy(child)], 'task_names': ['a', 'b'],
                   'augmentation_cfg': augmentation_cfg}
    if augmentation_cfg == {'pc': {'color_noise': {'prob': 1}}}:
        with pytest.raises(ValueError, match='color augmentation'):
            validate_config(cfg)
        with pytest.raises(hydra.errors.InstantiationException, match='color augmentation'):
            hydra.utils.instantiate(cfg.dataset)
    else:
        validate_config(cfg)
        ds = hydra.utils.instantiate(cfg.dataset)
        assert all(not child.augmentors for child in ds.datasets)
        assert ds[0]['obs']['point_cloud'].shape[-1] == 3


def test_xyzrgb_color_order_and_rng_unchanged(tmp_path):
    from dexmani_policy.datasets.augmentation import (
        PointCoordNoiseAug, PointColorJitter, PointColorNoiseAug, PointDropout,
    )
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import validate_config
    cfg = load_config('dp3')
    cfg.agent.pc_dim = cfg.dataset.pc_dim = 6
    aug_cfg = {'pc': {'coord_noise': {'prob': 0}, 'color': {'prob': 1},
                      'color_noise': {'prob': 1}, 'dropout': {'prob': 0}}}
    cfg.dataset.augmentation_cfg = aug_cfg
    validate_config(cfg)
    path = tmp_path / 'data'
    write_data(path)
    kwargs = dict(zarr_path=str(path), pc_dim=6, sensor_modalities=['point_cloud'])
    ds = BaseDataset(**kwargs, augmentation_cfg=aug_cfg)
    plain = BaseDataset(**kwargs)[0]['obs']['point_cloud'].numpy()
    classes = [PointCoordNoiseAug, PointColorJitter, PointColorNoiseAug, PointDropout]
    assert [type(aug) for aug in ds.augmentors['point_cloud']] == classes
    expected = plain.copy()
    np.random.seed(12)
    for cls, key in zip(classes, ['coord_noise', 'color', 'color_noise', 'dropout']):
        aug = cls(**aug_cfg['pc'][key])
        if np.random.random() <= aug.prob:
            aug._augment(expected)
    expected_rng = np.random.random()
    np.random.seed(12)
    actual = ds[0]['obs']['point_cloud'].numpy()
    np.testing.assert_array_equal(actual, expected)
    assert np.random.random() == expected_rng
    np.testing.assert_array_equal(actual[..., :3], plain[..., :3])
    assert not np.array_equal(actual[..., 3:], plain[..., 3:])


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
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
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


@pytest.mark.parametrize('norm', ['identity', 'batch_nrom'])
@pytest.mark.parametrize('backbone', ['resnet', 'r3m'])
def test_rgb_invalid_norm_rejected_before_backbone(monkeypatch, norm, backbone):
    from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNet
    from dexmani_policy.agents.obs_encoder.rgb.r3m import R3M
    from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid norm must fail before backbone construction or weight loading')

    monkeypatch.setattr('torchvision.models.resnet18', forbidden)
    monkeypatch.setattr('dexmani_policy.agents.obs_encoder.rgb.r3m._load_r3m_convnet_state_dict', forbidden)
    cls, kwargs = (ResNet, {'weights': 'DEFAULT'}) if backbone == 'resnet' else (R3M, {})
    with pytest.raises(ValueError, match='Unsupported norm_mode'):
        cls(norm_mode=norm, **kwargs)
    with pytest.raises(ValueError, match='Unsupported norm_mode'):
        build_backbone(backbone, dict(norm_mode=norm, **kwargs))


@pytest.mark.parametrize('backbone', ['resnet', 'r3m'])
def test_rgb_pretrained_frozen_gn_rejected_before_loading(monkeypatch, backbone):
    from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNet
    from dexmani_policy.agents.obs_encoder.rgb.r3m import R3M
    from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone

    def forbidden(*args, **kwargs):
        pytest.fail('Frozen pretrained GN must fail before backbone construction or weight loading')

    monkeypatch.setattr('torchvision.models.resnet18', forbidden)
    cls, kwargs = (ResNet, {'weights': 'DEFAULT'}) if backbone == 'resnet' else (R3M, {})
    with pytest.raises(ValueError, match='Frozen pretrained .* requires batch_norm/frozen_bn'):
        cls(norm_mode='group_norm', tune_mode='freeze', **kwargs)
    with pytest.raises(ValueError, match='Frozen pretrained .* requires batch_norm/frozen_bn'):
        build_backbone(backbone, dict(norm_mode='group_norm', tune_mode='freeze', **kwargs))


@pytest.mark.parametrize('weights,load_pretrained,tune', [
    (None, True, 'freeze'), ('DEFAULT', False, 'freeze'), ('DEFAULT', True, 'full'),
])
def test_resnet_gn_allowed_recipes_and_restore(monkeypatch, weights, load_pretrained, tune):
    from torchvision.models import ResNet18_Weights, resnet18
    from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone

    # Substitute local weights only at the download boundary; construct actual modules.
    calls = []

    def local_weights(*args, **kwargs):
        calls.append(True)
        return resnet18(weights=None).state_dict()

    monkeypatch.setattr(ResNet18_Weights, 'get_state_dict', local_weights)
    model, _ = build_backbone('resnet', dict(norm_mode='group_norm', tune_mode=tune, weights=weights,
                                           load_pretrained=load_pretrained))
    assert len(calls) == int(load_pretrained and weights is not None)
    assert model.effective_config['weights'] == weights
    restored, _ = build_backbone('resnet', dict(norm_mode='group_norm', tune_mode=tune,
                                              weights='DEFAULT', load_pretrained=False))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert len(calls) == int(load_pretrained and weights is not None)
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.backbone.modules())
    assert all(param.requires_grad == (tune == 'full') for param in model.backbone.parameters())
    model.eval(); restored.eval()
    with torch.no_grad():
        x = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(model(x)['global_token'], restored(x)['global_token'])
