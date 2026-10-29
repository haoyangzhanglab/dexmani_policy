"""Shared component, action, EMA, point-cloud, dataset and experiment regression checks."""

import json
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import policy_state_dict
from dexmani_policy.common.config import POINT_CLOUD_FIELDS, load_policy_config
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.test_tools.smoke_checks import expect_value_error, validate_target_imports


def validate_config_targets():
    from unittest import TestCase

    cfg = OmegaConf.create({
        'agent': {'_target_': 'torch.nn.Identity'},
        'components': [{'_target_': 'torch.relu'}],
    })
    with patch.object(torch.nn.Identity, '__init__', side_effect=AssertionError('config checks must not instantiate')):
        assert validate_target_imports(cfg) == 2
    for target in ('torch.nn.DoesNotExist', 'torch.float32'):
        cfg.components[0]._target_ = target
        with TestCase().assertRaises((ImportError, ValueError)):
            validate_target_imports(cfg)
    return {'status': 'passed', 'checks': 'callable targets, nested lists, no construction'}


def validate_action_recipes():
    import hydra
    from dexmani_policy.agents.policies.base import BasePolicy
    from dexmani_policy.common.config import validate_normalization_config

    names = ('dp', 'dp3', 'maniflow', 'dqrise', 'lamp', 'latent_flow', 'r3d', 'sat', 'afro', 'cordvip', 'adaptac')
    for name in names:
        cfg = load_policy_config(name)
        cfg.policy_name = 'display_name_is_not_a_recipe'
        validate_normalization_config(cfg)
        cls: type[BasePolicy] = hydra.utils.get_class(cfg.agent._target_)
        if cfg.normalization.action != 'identity':
            cfg.normalization.action = 'limits'
            validate_normalization_config(cfg)
        for mode in ('rms', 'max_abs', 'gaussian'):
            cfg.normalization.action = mode
            if 'clip_sample' in cfg.agent:
                cfg.agent.clip_sample = False
            expect_value_error(lambda: validate_normalization_config(cfg))
            expect_value_error(lambda: cls.validate_normalization_recipe(cfg.normalization, action_key=cfg.action_key))
        if 'action_ee' in cls.supported_action_keys:
            cfg.action_key = 'action_ee'
            cfg.normalization.action = 'auto'
            validate_normalization_config(cfg)
            cfg.normalization.action = 'limits'
            expect_value_error(lambda: validate_normalization_config(cfg))
    model = BasePolicy(torch.nn.Identity(), torch.nn.Identity(), 4, 1, 2, 19)
    for mode in ('rms', 'max_abs', 'gaussian', 'identity'):
        expect_value_error(lambda: model.set_normalization_spec({'action': mode}, {'action': 19}, action_key='action'))
    model.set_normalization_spec({'action': 'auto'}, {'action': 19}, action_key='action')
    expect_value_error(lambda: model.set_normalization_spec({'action': 'limits'}, {'action': 19}, action_key='action_ee'))
    # The mathematical normalizer remains independent of policy recipes.
    normalizer = LinearNormalizer()
    values = torch.tensor([[1., -2.], [3., 4.]])
    normalizer.fit_field('action', [values], mode='rms')
    torch.testing.assert_close(normalizer['action'].scale, values.double().square().mean().sqrt().reciprocal().float().expand(2))
    return {'status': 'passed', 'policies': list(names)}


def validate_ema_updates(device):
    import copy
    from dexmani_policy.training.common.optimization import EMA

    class Model(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.first = torch.nn.Linear(3, 2, device=device, dtype=dtype)
            self.frozen = torch.nn.Parameter(torch.arange(3., device=device), requires_grad=False)
            self.last = torch.nn.Linear(2, 1, device=device, dtype=torch.float64)
            self.register_buffer('normalizer', torch.tensor([2., 3.], device=device))
            self.register_buffer('counter', torch.tensor(5, device=device))

    cases = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for fixed_decay in (None, 0., .85):
            raw = Model(dtype)
            ema = EMA(copy.deepcopy(raw), fixed_decay=fixed_decay)
            expected = copy.deepcopy(ema.averaged_model)
            raw.first = cast(torch.nn.Linear, torch.compile(raw.first, backend="eager"))
            ema.averaged_model.first = torch.compile(ema.averaged_model.first, backend='eager')
            for step in range(5):
                with torch.no_grad():
                    for index, parameter in enumerate(raw.parameters()):
                        if parameter.requires_grad:
                            parameter.fill_((step - 2) * .17 + index * .3)
                    raw.counter.add_(1)
                    for source, target in zip(raw.parameters(), expected.parameters()):
                        if source.requires_grad:
                            decay = ema.get_decay(step)
                            target.mul_(decay).add_(source.to(target), alpha=1 - decay)
                ema.step(raw)
                assert ema.optimization_step == step + 1
                torch.testing.assert_close(policy_state_dict(ema.averaged_model), expected.state_dict(), rtol=1e-5, atol=1e-6)
                assert all(p.grad is None and not p.requires_grad for p in ema.averaged_model.parameters())
                assert ema.averaged_model.first.weight.dtype == torch.float32
                torch.testing.assert_close(ema.averaged_model.normalizer, raw.normalizer, rtol=0, atol=0)
                if step == 2:
                    # Check the next update after restoring the existing state/count fields.
                    restored = EMA(Model(dtype), fixed_decay=fixed_decay)
                    restored.averaged_model.load_state_dict(policy_state_dict(ema.averaged_model))
                    restored.optimization_step = ema.optimization_step
                    ema = restored
            cases += 1
    return {'status': 'passed', 'device': str(device), 'cases': cases,
            'checks': 'warmup/fixed decay, frozen/buffers, mixed precision, compile pairing, resume'}


def validate_fps(device):
    from itertools import product
    from dexmani_policy.agents.obs_encoder.pointcloud.ops import farthest_point_sample, load_pointcloud_ops

    try:
        backend = load_pointcloud_ops()
    except ImportError as error:
        return {'status': 'not_run', 'reason': str(error)}
    def rng_state():
        return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.empty(0))

    count = 0
    for channels, duplicate in product((3, 6), (False, True)):
        cloud = torch.arange(2 * 16 * channels, device=device, dtype=torch.float32).reshape(2, 16, channels) / 13
        if duplicate:
            cloud[:, 4:8, :3] = cloud[:, :1, :3]
        for random, start, shuffle, noise in product((False, True), (False, True), (False, True), (0., .01)):
            options: dict[str, Any] = dict(use_random=random, use_random_start=start, use_shuffle_output=shuffle, random_noise_scale=noise)
            set_seed(123)
            before = rng_state()
            points, indices = farthest_point_sample(cloud, 7, **options)
            after = rng_state()
            assert points.shape == (2, 7, channels) and indices.shape == (2, 7)
            assert indices.dtype == torch.int64 and ((indices >= 0) & (indices < 16)).all()
            torch.testing.assert_close(points, cloud.gather(1, indices[..., None].expand(-1, -1, channels)), rtol=0, atol=0)
            if not random:
                torch.testing.assert_close(after, before, rtol=0, atol=0)
                _, expected = backend.sample_farthest_points(cloud[..., :3], K=7, random_start_point=False)
                torch.testing.assert_close(indices, expected, rtol=0, atol=0)
            set_seed(123)
            again = farthest_point_sample(cloud, 7, **options)
            torch.testing.assert_close((points, indices), again, rtol=0, atol=0)
            torch.testing.assert_close(after, rng_state(), rtol=0, atol=0)
            count += 1
    return {'status': 'passed', 'backend': 'native PyTorch3D', 'device': str(device), 'cases': count}


def validate_dataset_layouts():
    import zarr
    from dexmani_policy.common.hand_semantics import HAND_ACTION_SEMANTICS
    from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
    from dexmani_policy.datasets.common.robot_dataset import RobotDataset
    from dexmani_policy.datasets.aux.r3d_dataset import R3DAuxDataset
    from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
    from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
    from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
    from dexmani_policy.datasets.obs.cordvip_dataset import CordViPDataset
    from dexmani_policy.training.pretrain.hand_prior.lamp import prepare_prior_data, prior_batch
    from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_policy_data

    split = dict(episode_ends=[8, 16, 24], train_indices=[0], val_indices=[1])
    rng = np.random.default_rng(3)
    arrays = {name: rng.normal(size=shape).astype(np.float32) for name, shape in {
        'action': (24, 19), 'action_ee': (24, 21), 'joint_state': (24, 19), 'point_cloud': (24, 5, 6),
        'hand_pointcloud': (24, 5, 6), 'object_pointcloud': (24, 1024, 6), 'contact_map': (24, 1024),
    }.items()}
    windows = 0
    with tempfile.TemporaryDirectory(prefix='dataset-contract-') as directory:
        root = zarr.open_group(directory, mode='w')
        root.attrs.update(dt=.0625)
        root.create_dataset('meta/episode_ends', data=np.array(split['episode_ends']))
        for key, values in arrays.items():
            root.create_dataset(f'data/{key}', data=values, chunks=(3, *values.shape[1:]))
        cordvip = CordViPDataset(directory, horizon=4, obs_horizon=2, pad_before=1,
                                pad_after=2, split_manifest=split)
        for view in (cordvip, cordvip.get_validation_dataset()):
            for index in range(len(view)):
                rows = view.sampler.sample_source_rows(index)[:2]
                batch = view[index]
                for key in view.sensor_modalities:
                    expected = arrays[key][rows]
                    if key in POINT_CLOUD_FIELDS:
                        expected = expected[..., :3]
                    np.testing.assert_array_equal(batch['obs'][key].numpy(), expected)
                    assert batch['obs'][key].dtype == torch.float32
                np.testing.assert_array_equal(batch['contact_map'].numpy(), arrays['contact_map'][rows])
                windows += 1
        for cls in (RobotDataset, R3DAuxDataset, LatentFlowDataset):
            for pc_dim in (3, 6):
                options = dict(data_semantics=HAND_ACTION_SEMANTICS) if cls is LatentFlowDataset else dict(pad_after=2)
                dataset = cls(directory, sensor_modalities=['joint_state', 'point_cloud'], horizon=4,
                              obs_horizon=2, pad_before=1, split_manifest=split, pc_dim=pc_dim, **options)
                for view in (dataset, dataset.get_validation_dataset()):
                    for index in range(len(view)):
                        rows = view.sampler.sample_source_rows(index)
                        batch = view[index]
                        np.testing.assert_array_equal(batch['obs']['point_cloud'], arrays['point_cloud'][rows[:2], :, :pc_dim])
                        np.testing.assert_array_equal(batch['obs']['joint_state'], arrays['joint_state'][rows[:2]])
                        if cls is LatentFlowDataset:
                            expected = arrays['action'][rows[1:]]
                            np.testing.assert_array_equal(batch['action_context']['reference_state'], arrays['joint_state'][rows[1]])
                        elif cls is R3DAuxDataset:
                            expected = np.concatenate((arrays['action'][rows], arrays['action_ee'][rows, :9]), axis=-1)
                        else:
                            expected = arrays['action'][rows]
                        np.testing.assert_array_equal(batch['action'], expected)
                        windows += 1
                stats = np.concatenate(list(dataset.iter_normalization_data('action')))
                if cls is LatentFlowDataset:
                    reference = np.concatenate([
                        item['action'].numpy() - item['action_context']['reference_state'].numpy()
                        for item in (dataset[index] for index in range(len(dataset)))
                    ])
                elif cls is R3DAuxDataset:
                    reference = np.concatenate((arrays['action'][:8], arrays['action_ee'][:8, :9]), axis=-1)
                else:
                    reference = arrays['action'][:8]
                np.testing.assert_array_equal(stats, reference)
        # Joint auto/limits equivalence and EEF's unscaled rot6d segment.
        for key in ('action', 'action_ee'):
            dataset = RobotDataset(directory, sensor_modalities=['joint_state'], action_key=key, split_manifest=split)
            normalizer = dataset.get_normalizer({'action': 'auto', 'joint_state': 'limits'})
            expected = LinearNormalizer()
            expected.fit_field('action', [arrays[key][:8]], mode='limits')
            scale, offset = expected['action'].scale, expected['action'].offset
            if key == 'action_ee':
                scale[3:9] = 1
                offset[3:9] = 0
            torch.testing.assert_close(normalizer['action'].scale, scale)
            torch.testing.assert_close(normalizer['action'].offset, offset)
        cfg = load_policy_config('dqrise')
        cfg.dataset.zarr_path = directory
        cfg.dataset.split_manifest = split
        reads = {'action': 0, 'joint_state': 0}
        original = ReplayBuffer.read

        def read(buffer, key, rows, *, columns=None):
            reads[key] += rows.stop - rows.start
            return original(buffer, key, rows, columns=columns)

        with patch.object(ReplayBuffer, 'read', read):
            hand = HandChunkDataset(directory, horizon=4, data_semantics=HAND_ACTION_SEMANTICS, split_manifest=split)
        assert reads == {'action': 16, 'joint_state': 16}
        np.testing.assert_array_equal(hand.hand_actions[:16], arrays['action'][:16, 7:19])
        assert not hand.hand_actions.flags.writeable and not hand.hand_states.flags.writeable
        for view in (hand, hand.get_validation_dataset()):
            indices = np.array([len(view) - 1, 0, 1])
            actions, states = view.sample_chunks(indices)
            np.testing.assert_array_equal(actions, np.stack([view[i]['action'] for i in indices]))
            np.testing.assert_array_equal(states, np.stack([view[i]['reference_state'] for i in indices]))
        reads = dict.fromkeys(reads, 0)
        with patch.object(ReplayBuffer, 'read', read):
            train, val, normalizer, _ = prepare_policy_data(cfg)
        assert reads == {'action': 16, 'joint_state': 0}
        torch.testing.assert_close(normalizer['hand'].unnormalize(train), torch.from_numpy(arrays['action'][:8, 7:19]))
        torch.testing.assert_close(normalizer['hand'].unnormalize(val), torch.from_numpy(arrays['action'][8:16, 7:19]))

        lamp_cfg = load_policy_config('lamp')
        lamp_cfg.dataset.zarr_path = directory
        lamp_cfg.dataset.split_manifest = split
        lamp_cfg.lamp.history_steps = 3
        reads = dict.fromkeys(reads, 0)
        with patch.object(ReplayBuffer, 'read', read):
            prior = prepare_prior_data(lamp_cfg)
        assert reads == {'action': 16, 'joint_state': 2}
        lamp = LAMPDataset(directory, sensor_modalities=['joint_state'], history_steps=3, split_manifest=split)
        assert prior['hand_data'] == lamp.hand_data
        for view, row_key in ((lamp, 'train_rows'), (lamp.get_validation_dataset(), 'val_rows')):
            for index, row in enumerate(prior[row_key]):
                start = 0 if row < 8 else 8
                expected = np.stack([
                    arrays['action'][t, 7:19] if t >= start else arrays['joint_state'][start, 7:19]
                    for t in range(row - 3, row)
                ])
                history, target = prior_batch(prior, np.array([row]), 'cpu')
                np.testing.assert_array_equal(history[0], expected)
                np.testing.assert_array_equal(target[0], arrays['action'][row, 7:19])
                np.testing.assert_array_equal(view[index]['action_context']['hand_history'], expected)

        for bad in (np.nan, np.inf):
            root['data/action'][9, 10] = bad
            with patch.object(LinearNormalizer, 'fit_field', side_effect=AssertionError('validate before fitting')):
                for build in (
                    lambda: HandChunkDataset(directory, horizon=4, data_semantics=HAND_ACTION_SEMANTICS, split_manifest=split),
                    lambda: prepare_policy_data(cfg),
                    lambda: prepare_prior_data(lamp_cfg),
                ):
                    try:
                        build()
                    except ValueError as error:
                        assert 'row=9, channel=(10,)' in str(error), error
                    else:
                        raise AssertionError('Invalid validation frame was accepted')

    with tempfile.TemporaryDirectory(prefix='relative-windows-') as directory:
        split = dict(episode_ends=[1030, 1038, 1046], train_indices=[0, 2], val_indices=[1])
        root = zarr.open_group(directory, mode='w')
        root.attrs.update(dt=.0625)
        root.create_dataset('meta/episode_ends', data=np.array(split['episode_ends']))
        values = {key: rng.normal(size=(1046, 19)).astype(np.float32) for key in ('action', 'joint_state')}
        for key, array in values.items():
            root.create_dataset(f'data/{key}', data=array, chunks=(64, 19))
        dataset = LatentFlowDataset(directory, sensor_modalities=['joint_state'], horizon=4, obs_horizon=3,
                                    data_semantics=HAND_ACTION_SEMANTICS, split_manifest=split)
        batches = list(dataset.iter_normalization_data('action'))
        assert [len(batch) for batch in batches] == [1024 * 4, 3 * 4, 5 * 4]
        queries = np.r_[np.arange(1027), np.arange(1038, 1043)]
        expected = values['action'][queries[:, None] + np.arange(4)] - values['joint_state'][queries, None]
        np.testing.assert_array_equal(np.concatenate(batches), expected.reshape(-1, 19))
    return {'status': 'passed', 'windows': windows, 'hand_reads': 'one pass per selected field',
            'hand_priors': 'chunk batching, LAMP bootstrap/history, relative windows across batches and episodes'}


def validate_hand_prior_contracts():
    import copy
    import zarr

    from dexmani_policy.common.hand_semantics import (
        HAND_ACTION_SEMANTICS, validate_codec_data, validate_pretraining_split,
    )
    from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
    from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
    from dexmani_policy.training.pretrain.hand_prior.hand_vae import fit_relative_statistics
    from dexmani_policy.training.pretrain.hand_prior.lamp import prepare_prior_data, prior_batch
    from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_policy_data

    rng = np.random.default_rng(31)
    lengths = np.array([7, 49, 5000, 53, 3000])
    ends, starts = lengths.cumsum(), np.r_[0, lengths.cumsum()[:-1]]
    split = dict(source_id='hand-contract', episode_ids=[f'ep-{i}' for i in range(5)],
                 episode_ends=ends.tolist(), train_indices=[0, 3], val_indices=[1])
    selected_rows = np.concatenate([np.arange(starts[i], ends[i]) for i in (0, 1, 3)])
    arrays = {key: rng.normal(size=(ends[-1], 19)).astype(np.float32) for key in ('action', 'joint_state')}
    for values in arrays.values():
        values[starts[2] + 1, 10] = np.nan

    with tempfile.TemporaryDirectory(prefix='hand-priors-') as directory:
        root = zarr.open_group(directory, mode='w')
        root.attrs.update(dt=.0625, source_id=split['source_id'], episode_ids=split['episode_ids'])
        root.create_dataset('meta/episode_ends', data=ends.astype(np.uint64))
        for key, values in arrays.items():
            root.create_dataset(f'data/{key}', data=values, chunks=(128, 19))

        for horizon in (3, 8, 32):
            hand = HandChunkDataset(directory, data_semantics=HAND_ACTION_SEMANTICS,
                                    horizon=horizon, split_manifest=split)
            assert hand.replay_buffer.episode_ends.dtype == np.int64
            assert hand.hand_row_offsets[3] == -lengths[2]
            assert hand.hand_actions.shape == hand.hand_states.shape == (len(selected_rows), 12)
            assert not hand.hand_actions.flags.writeable and not hand.hand_states.flags.writeable
            np.testing.assert_array_equal(hand.hand_actions, arrays['action'][selected_rows, 7:19])
            np.testing.assert_array_equal(hand.hand_states, arrays['joint_state'][selected_rows, 7:19])
            for view in (hand, hand.get_validation_dataset()):
                assert view.hand_actions is hand.hand_actions and view.hand_states is hand.hand_states
                rows = view.sampler.source_rows(view.sampler.indices)
                actions, q0 = view.sample_chunks(np.arange(len(view)))
                np.testing.assert_array_equal(actions, arrays['action'][rows, 7:19])
                np.testing.assert_array_equal(q0, arrays['joint_state'][rows[:, 0], 7:19])
            actions, q0 = hand.sample_chunks(np.arange(len(hand)))
            relative = actions.astype(np.float64) - q0[:, None].astype(np.float64)
            for batch_size in (1, 13, 128):
                mean, std, statistics = fit_relative_statistics(hand, batch_size)
                np.testing.assert_allclose(mean, relative.mean((0, 1)), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(std, relative.std((0, 1)).clip(min=1e-6), rtol=1e-6, atol=1e-7)
                assert statistics['frames'] == len(hand) * horizon and type(statistics['frames']) is int
            expect_value_error(lambda: fit_relative_statistics(hand.get_validation_dataset()))

        lamp_split = {**split, 'train_indices': [3]}
        lamp = LAMPDataset(directory, sensor_modalities=['joint_state'], history_steps=8,
                           split_manifest=lamp_split)
        cfg = load_policy_config('lamp')
        cfg.dataset.zarr_path, cfg.dataset.split_manifest = directory, lamp_split
        prior = prepare_prior_data(cfg)
        assert prior['hand_actions'].shape == lamp.hand_actions.shape == (lengths[[1, 3]].sum(), 12)
        assert prior['bootstrap'].shape == lamp.hand_state_bootstrap.shape == (2, 12)
        assert not lamp.hand_actions.flags.writeable and not lamp.hand_state_bootstrap.flags.writeable
        for view, key in ((lamp, 'train_rows'), (lamp.get_validation_dataset(), 'val_rows')):
            assert view.hand_actions is lamp.hand_actions
            for index, local_row in enumerate(prior[key]):
                row = view.sampler.sample_source_rows(index)[0]
                episode = np.searchsorted(ends, row, side='right')
                start = starts[episode]
                expected = np.stack([
                    arrays['action'][t, 7:19] if t >= start else arrays['joint_state'][start, 7:19]
                    for t in range(row - 8, row)
                ])
                history, target = prior_batch(prior, np.array([local_row]), 'cpu')
                np.testing.assert_array_equal(history[0], expected)
                np.testing.assert_array_equal(target[0], arrays['action'][row, 7:19])
                np.testing.assert_array_equal(view[index]['action_context']['hand_history'], expected)
        cfg.dataset.split_manifest = {**lamp_split, 'val_indices': []}
        no_val = prepare_prior_data(cfg)
        assert not len(no_val['val_rows']) and no_val['hand_actions'].shape == (lengths[3], 12)

        cfg = load_policy_config('dqrise')
        cfg.dataset.zarr_path = directory
        for validation in ([1], []):
            cfg.dataset.split_manifest = {**split, 'val_indices': validation}
            train, val, normalizer, _ = prepare_policy_data(cfg)
            train_rows = np.r_[np.arange(ends[0]), np.arange(starts[3], ends[3])]
            expected_val = arrays['action'][starts[1]:ends[1], 7:19] if validation else np.empty((0, 12))
            np.testing.assert_allclose(normalizer['hand'].unnormalize(train),
                                       arrays['action'][train_rows, 7:19], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(normalizer['hand'].unnormalize(val), expected_val, rtol=1e-5, atol=1e-6)

        data = dict(hand_data=hand.hand_data, split_manifest=hand.split_manifest, zarr_path=directory)
        assert set(hand.hand_data) == {'contract', 'semantics_evidence'}
        validate_codec_data(data, data)
        validate_pretraining_split(data['split_manifest'], data['split_manifest'])
        reordered = copy.deepcopy(data)
        reordered['split_manifest'].update(episode_ids=split['episode_ids'][::-1],
                                            episode_ends=lengths[::-1].cumsum().tolist(),
                                            train_indices=[1, 4], val_indices=[3])
        validate_codec_data(reordered, data)
        independent = copy.deepcopy(data)
        independent['split_manifest'].update(source_id='other', train_indices=[1], val_indices=[0])
        validate_codec_data(independent, data)
        for key, value in (('source_id', 'other'), ('train_indices', [3]), ('val_indices', [3])):
            changed = {**data['split_manifest'], key: value}
            expect_value_error(lambda: validate_pretraining_split(data['split_manifest'], changed))
        conflict = copy.deepcopy(data)
        conflict['split_manifest'].update(train_indices=[0, 1], val_indices=[3])
        expect_value_error(lambda: validate_codec_data(conflict, data))
        conflict = copy.deepcopy(independent)
        conflict['hand_data']['contract']['dt'] = .1
        expect_value_error(lambda: validate_codec_data(conflict, data))
        fallback = copy.deepcopy(data)
        fallback['split_manifest'].update(source_id=None, episode_ids=None)
        validate_codec_data(fallback, fallback)

        for noise in (0., 1e-4):
            for key in arrays:
                root[f'data/{key}'][:] = (2 + noise * rng.normal(size=arrays[key].shape)).astype(np.float32)
            hand = HandChunkDataset(directory, data_semantics=HAND_ACTION_SEMANTICS,
                                    horizon=8, split_manifest=split)
            actions, q0 = hand.sample_chunks(np.arange(len(hand)))
            relative = actions.astype(np.float64) - q0[:, None].astype(np.float64)
            mean, std, _ = fit_relative_statistics(hand, batch_size=13)
            np.testing.assert_allclose(mean, relative.mean((0, 1)), rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(std, relative.std((0, 1)).clip(min=1e-6), rtol=1e-6, atol=1e-10)
    return {'status': 'passed', 'selected_frames': len(selected_rows), 'source_frames': int(ends[-1]),
            'checks': 'compact arrays, unsigned source boundaries, rolling moments, LAMP/VQ data, canonical splits'}


def validate_experiment_paths():
    import os
    from unittest import TestCase

    from dexmani_policy.common.experiments import discover_milestone_checkpoints, resolve_policy_checkpoint
    from dexmani_policy.training.common.resume import archive_resume_milestones

    check = TestCase()
    with tempfile.TemporaryDirectory(prefix='experiment-contract-') as directory:
        root = Path(directory) / 'source'
        checkpoints = root / 'checkpoints'
        checkpoints.mkdir(parents=True)
        files = [checkpoints / f'epoch=1-step={step}-milestone={pct}pct.pt' for step, pct in ((2, 20), (4, 40), (6, 60))]
        for path in files:
            path.write_bytes(path.name.encode())
        (checkpoints / 'latest.pt').symlink_to(files[-1].name)
        selection = dict(ckpt_relpath=str(files[0].relative_to(root)), global_step=2,
                         inference=dict(use_ema=False, inference_steps=3, action_steps=2),
                         selection_result='selection.json')
        (root / 'best_ckpt.json').write_text(json.dumps(selection))
        for experiment in (root, Path(os.path.relpath(root))):
            assert [item.path for item in discover_milestone_checkpoints(experiment)] == files
            for selector, expected in (('best', files[0]), ('latest', files[-1]), ('40pct', files[1]),
                                       (files[0], files[0]), (os.path.relpath(files[0]), files[0])):
                path, record = resolve_policy_checkpoint(experiment, selector)
                assert path == expected
                assert record == (selection if selector == 'best' else None)
        assert resolve_policy_checkpoint(root, files[1].name, relative_to=checkpoints)[0] == files[1]
        for selector in ('80pct', 'missing.pt'):
            with check.assertRaises(FileNotFoundError):
                resolve_policy_checkpoint(root, selector, relative_to=checkpoints)
        outside = Path(directory) / 'outside.pt'
        outside.write_bytes(b'outside')
        (checkpoints / 'outside.pt').symlink_to(outside)
        for selector in (outside, checkpoints / 'outside.pt', checkpoints):
            expect_value_error(lambda: resolve_policy_checkpoint(root, selector))
        with patch('torch.load', side_effect=AssertionError('archive must not deserialize')):
            for max_step, expected in ((0, []), (4, files[:2])):
                destination = Path(directory) / f'resume-{max_step}'
                archive_resume_milestones(files[-1], destination, max_step=max_step)
                actual = sorted((destination / 'checkpoints').iterdir())
                assert [path.name for path in actual] == [path.name for path in expected]
                assert [path.read_bytes() for path in actual] == [path.read_bytes() for path in expected]
            with check.assertRaises(FileExistsError):
                archive_resume_milestones(files[-1], destination, max_step=4)
    return {'status': 'passed', 'checks': 'selectors, relative roots, experiment boundary, file-only archive'}


def validate_resume_facts():
    import copy
    from types import SimpleNamespace

    from dexmani_policy.training.common.config import assemble_effective_config

    cfg = OmegaConf.create(dict(agent={}, dataset={'zarr_path': '/old/data.zarr'}, training=dict(lr_scheduler='cosine')))
    model = SimpleNamespace(inference_metadata={'control_action_dim': 19}, effective_config={})
    description = dict(
        normalization_dims={'action': 19}, real_runtime={'dt': 0.05},
        hand_data=dict(contract={'dt': 0.05}), zarr_path='/old/data.zarr',
        split_manifest=dict(source_id='same', episode_ends=[4], train_indices=[0], val_indices=[]),
    )
    saved = OmegaConf.to_container(assemble_effective_config(cfg, description, model), resolve=True)
    checkpoint = SimpleNamespace(effective_config=saved)
    moved = copy.deepcopy(description)
    moved['zarr_path'] = '/new/data.zarr'
    moved_cfg = OmegaConf.create(saved)
    moved_cfg.dataset.zarr_path = moved['zarr_path']
    resumed = assemble_effective_config(moved_cfg, moved, model, checkpoint=checkpoint)
    assert resumed.dataset.zarr_path == '/new/data.zarr'
    assert saved['dataset']['zarr_path'] == '/old/data.zarr'
    assert resumed.hand_data == saved['hand_data'] == moved['hand_data']
    assert OmegaConf.is_readonly(resumed)
    for key in ('normalization_dims', 'real_runtime', 'hand_data', 'policy_metadata', 'split_manifest'):
        changed, actual_model = copy.deepcopy(moved), copy.deepcopy(model)
        if key == 'policy_metadata':
            actual_model.inference_metadata['control_action_dim'] = 18
        elif key == 'normalization_dims':
            changed[key]['action'] = 18
        elif key == 'real_runtime':
            changed[key]['dt'] = 0.1
        elif key == 'hand_data':
            changed[key]['contract']['dt'] = 0.1
        else:
            changed[key]['source_id'] = 'different'
        expect_value_error(lambda: assemble_effective_config(OmegaConf.create(saved), changed, actual_model, checkpoint=checkpoint))
    for key in ('real_runtime', 'hand_data'):
        changed = {name: value for name, value in moved.items() if name != key}
        expect_value_error(lambda: assemble_effective_config(OmegaConf.create(saved), changed, model, checkpoint=checkpoint))
    return {'status': 'passed', 'checks': 'relocation, actual data/model changes and missing metadata'}


def _check_finite_gradients(module):
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def _hand_observations(device):
    rotation = torch.tensor([
        [[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]],
        [[1., 0., 0.], [0., .8, -.6], [0., .6, .8]],
    ], device=device)
    origin = torch.tensor([[0., 0., 0.], [.25, -.1, .35]], device=device)
    local_xyz = torch.tensor([
        [-.04, .03, .02], [.025, .09, .01], [0., .105, .02],
        [-.025, .09, .01], [-.045, .065, 0.],
    ], device=device).expand(2, -1, -1)
    observations = {
        'joint_state': torch.linspace(-.6, .7, 38, device=device).reshape(2, 19),
        'eef_pose': torch.cat((origin, rotation[..., :, 0], rotation[..., :, 1]), -1),
        'fingertip_xyz': local_xyz @ rotation.transpose(-1, -2) + origin[:, None],
    }
    return observations, rotation, local_xyz


def validate_hand_state_encoder(device):
    from dexmani_policy.agents.obs_encoder.proprio import HandStateEncoder

    set_seed(31)
    observations, rotation, local_xyz = _hand_observations(device)
    joints, pose, fingertips = (observations[key] for key in ('joint_state', 'eef_pose', 'fingertip_xyz'))
    for frequencies in (2, 0):
        model = HandStateEncoder(num_frequencies=frequencies).to(device)
        assert model.KEYPOINTS == ('wrist', 'thumb', 'index', 'middle', 'ring', 'pinky')
        assert model.out_dim == 192 and model.out_shape == (6, 192)
        position_dim = 3 + 6 * frequencies
        assert model.position_mlp[0].in_features == position_dim
        assert model.arm_mlp[0].in_features == 13
        assert model.finger_mlp[0].in_features == 3 + position_dim
        captured = {}
        with (
            model.position_mlp.register_forward_pre_hook(lambda _, args: captured.update(position=args[0].detach())),
            model.arm_mlp.register_forward_pre_hook(lambda _, args: captured.update(arm=args[0].detach())),
            model.finger_mlp.register_forward_pre_hook(lambda _, args: captured.update(finger=args[0].detach())),
        ):
            output = model(**observations)
        shapes = {'hand_tokens': (2, 6, 192), 'anchor_xyz': (2, 6, 3), 'rotation_world_from_eef': (2, 3, 3)}
        assert output.keys() == shapes.keys()
        for key, shape in shapes.items():
            assert output[key].shape == shape and torch.isfinite(output[key]).all(), key
        for key in ('anchor_xyz', 'rotation_world_from_eef'):
            assert output[key].dtype == torch.float32
        torch.testing.assert_close(output['rotation_world_from_eef'], rotation)
        torch.testing.assert_close(output['anchor_xyz'], torch.cat((pose[:, None, :3], fingertips), 1), rtol=0, atol=0)
        torch.testing.assert_close(captured['position'][..., :3], output['anchor_xyz'] / model.position_scale)
        torch.testing.assert_close(captured['arm'], torch.cat((joints[:, :7], pose[:, 3:]), -1))
        torch.testing.assert_close(captured['finger'][..., 3:6], local_xyz / model.hand_scale)
        padded = joints.new_zeros(2, 5, 3)
        padded[:, :2] = joints[:, 7:13].reshape(2, 2, 3)
        padded[:, 2:, :2] = joints[:, 13:].reshape(2, 3, 2)
        torch.testing.assert_close(captured['finger'][..., :3], padded, rtol=0, atol=0)

        with torch.no_grad():
            baseline = output['hand_tokens']
            for slot, (start, end) in enumerate(((7, 10), (10, 13), (13, 15), (15, 17), (17, 19)), 1):
                changed = joints.clone()
                changed[:, start:end] += .4
                delta = (model(changed, pose, fingertips)['hand_tokens'] - baseline).abs().sum(-1)
                assert (delta[:, slot] > 1e-5).all()
                assert not delta[:, torch.arange(6, device=device) != slot].any()
            changed = joints.clone()
            changed[:, :7] += .4
            delta = (model(changed, pose, fingertips)['hand_tokens'] - baseline).abs().sum(-1)
            assert (delta > 1e-5).all()
            for row in range(2):
                single = model(**{key: value[row:row + 1] for key, value in observations.items()})
                torch.testing.assert_close(single, {key: value[row:row + 1] for key, value in output.items()})

        # LayerNorm makes an unweighted squared norm almost constant.
        (output['hand_tokens'] * torch.randn_like(output['hand_tokens'])).mean().backward()
        _check_finite_gradients(model)
        for name in ('position_mlp', 'arm_mlp', 'finger_mlp'):
            assert all(parameter.grad.abs().sum() > 0 for parameter in getattr(model, name).parameters()), name

    expect_value_error(lambda: HandStateEncoder(num_frequencies=-1))
    expect_value_error(lambda: HandStateEncoder(hand_scale=0))
    expect_value_error(lambda: model(joints[:, :18], pose, fingertips))
    expect_value_error(lambda: model(joints, pose[:1], fingertips))
    expect_value_error(lambda: model(joints, pose, fingertips[:, :4]))
    return {'status': 'passed', 'device': str(device),
            'checks': 'column rotations, batched frames, interfaces, joint groups, MLP gradients, Fourier/none'}


def validate_hand_scene_encoder(device):
    from dexmani_policy.agents.obs_encoder.interaction.hand_scene_encoder import (
        HandSceneEncoder, query_surface_neighbors,
    )

    set_seed(37)
    observations, rotation, local_xyz = _hand_observations(device)
    anchors = torch.cat((observations['eef_pose'][:, None, :3], observations['fingertip_xyz']), 1).requires_grad_()
    rotation = rotation.requires_grad_()
    xyz = torch.cat((anchors.detach() + .002, anchors.detach() + torch.tensor([-.005, .004, .002], device=device)), 1)
    points = torch.cat((xyz, torch.rand_like(xyz)), -1)
    points[:, -1, :3] = torch.nan
    points[1, -3, :3] = torch.inf
    points.requires_grad_()
    point_mask = torch.ones(2, 12, dtype=torch.bool, device=device)
    point_mask[:, -2:] = False
    centers = anchors[:, [0, 2, 4]].detach() + .008
    centers[0, 2] = torch.nan
    centers.requires_grad_()
    memory = torch.randn(2, 3, 192, device=device)
    memory[1, 2] = torch.nan
    memory.requires_grad_()
    patches = {'tokens': memory, 'xyz': centers,
               'mask': torch.tensor([[True, False, True], [True, True, True]], device=device)}
    patch_valid = torch.tensor([[True, False, False], [True, True, False]], device=device)
    hand = torch.randn(2, 6, 192, device=device, requires_grad=True)
    query_context = torch.randn(2, 192, device=device, requires_grad=True)
    model = HandSceneEncoder(surface_k=4).to(device)
    captured = {}
    with (
        model.context.edge.register_forward_pre_hook(lambda _, args: captured.update(patch=args[0].detach())),
        model.hand_shape.register_forward_pre_hook(lambda _, args: captured.update(shape=args[0].detach())),
        model.surface.register_forward_pre_hook(lambda _, args: captured.update(surface=args[0].detach())),
    ):
        output = model(points, patches, anchors, hand, rotation_world_from_eef=rotation,
                       point_mask=point_mask, query_context=query_context, return_intermediate=True)
        first_inputs = dict(captured)
        assert output['hand_query'] is hand
        assert output['geometry_descriptor'].shape == (2, 6, 7)
        torch.testing.assert_close(output['interaction_center'], anchors, rtol=0, atol=0)
        torch.testing.assert_close(first_inputs['shape'], (local_xyz / model.metric_scale).flatten(1))
        torch.testing.assert_close(first_inputs['surface'], output['surface_relative'][:, 1:], rtol=0, atol=0)
        for key in ('geometry_descriptor', 'surface_relative'):
            assert output[key].dtype == torch.float32 and not output[key].requires_grad
        assert not output['surface_relative'][~output['surface_valid']].any()
        assert not output['context_attention'][..., :-1].masked_select(~patch_valid[:, None, None]).any()

        support = query_surface_neighbors(points[..., :3], anchors, 4, model.surface_radius, model.metric_scale, point_mask)
        torch.testing.assert_close(output['surface_relative'], support['relative'] @ rotation[:, None])
        torch.testing.assert_close(output['geometry_descriptor'][..., :3], support['descriptor'][..., :3] @ rotation)
        torch.testing.assert_close(output['geometry_descriptor'][..., 3:], support['descriptor'][..., 3:], rtol=0, atol=0)
        patch_relative = (torch.nan_to_num(centers)[:, None] - anchors[:, :, None]) @ rotation[:, None]
        patch_relative = torch.where(patch_valid[:, None, :, None], patch_relative, 0.)
        torch.testing.assert_close(first_inputs['patch'][..., :3], patch_relative / model.metric_scale)

        with torch.no_grad():
            transform = torch.tensor([
                [[1., 0., 0.], [0., .6, -.8], [0., .8, .6]],
                [[.8, 0., .6], [0., 1., 0.], [-.6, 0., .8]],
            ], device=device)
            translation = torch.tensor([[.1, -.2, .05], [-.12, .08, -.04]], device=device)[:, None]
            transformed_points = torch.cat((points[..., :3] @ transform.transpose(-1, -2) + translation, points[..., 3:]), -1)
            transformed_patches = {**patches, 'xyz': centers @ transform.transpose(-1, -2) + translation}
            transformed = model(
                transformed_points, transformed_patches, anchors @ transform.transpose(-1, -2) + translation,
                hand, rotation_world_from_eef=transform @ rotation, point_mask=point_mask,
                query_context=query_context[:, None].expand(-1, 6, -1), return_intermediate=True,
            )
            for key in first_inputs:
                torch.testing.assert_close(captured[key], first_inputs[key], rtol=1e-4, atol=1e-5)
            for key in ('geometry_descriptor', 'surface_relative', 'relation_update', 'context_update', 'finger_local',
                        'context_attention', 'min_observed_distance'):
                torch.testing.assert_close(transformed[key], output[key], rtol=1e-4, atol=1e-5)
            for key in ('surface_valid', 'has_reference', 'near_has_support'):
                torch.testing.assert_close(transformed[key], output[key], rtol=0, atol=0)
            shared_context = model(points, patches, anchors, hand, rotation_world_from_eef=rotation,
                                   point_mask=point_mask, query_context=query_context[:, None])
            torch.testing.assert_close(shared_context['relation_update'], output['relation_update'], rtol=0, atol=0)

    (output['relation_update'] * torch.randn_like(output['relation_update'])).mean().backward()
    _check_finite_gradients(model)
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    for value in (hand, query_context, memory):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert not memory.grad[~patch_valid].any()
    assert all(value.grad is None for value in (anchors, rotation, centers, points))

    with torch.no_grad():
        for empty in (False, True):
            cloud = points[:, :0] if empty else points
            missing_patches = {key: value[:, :0] for key, value in patches.items()} if empty else {
                **patches, 'mask': torch.zeros_like(patches['mask']),
            }
            missing = model(cloud, missing_patches, anchors, hand, rotation_world_from_eef=rotation,
                            point_mask=torch.zeros(cloud.shape[:2], dtype=torch.bool, device=device),
                            return_intermediate=True)
            for key in ('relation_update', 'geometry_descriptor', 'surface_relative', 'surface_valid',
                        'finger_local', 'has_reference', 'near_has_support'):
                assert not missing[key].any(), key
            assert torch.isinf(missing['min_observed_distance']).all()
            torch.testing.assert_close(missing['context_null_mass'], torch.ones_like(missing['context_null_mass']), rtol=0, atol=0)
    expect_value_error(lambda: model(points, patches, anchors, hand, rotation_world_from_eef=rotation[:1]))
    return {'status': 'passed', 'device': str(device),
            'checks': 'four EEF directions, rigid transforms, query context, coordinate/parameter gradients, empty/masked scenes'}


def validate_gated_tactile_fusion(device):
    from dexmani_policy.agents.obs_encoder.interaction.gated_tactile_fusion import GatedTactileFusion
    from dexmani_policy.agents.obs_encoder.xhand_tactile import (
        XHandForceEncoder, XHandTaxelEncoder, XHandPatchEncoder,
    )

    def check_backward(model, output, active):
        model.zero_grad(set_to_none=True)
        output['interaction_tokens'].square().mean().backward()
        _check_finite_gradients(model)
        if active:
            for name in ('tactile_encoder', 'self_attn', 'ffn', 'q', 'kv', 'proj', 'gate_mlp'):
                assert any(p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
            assert model.finger_embedding.grad.abs().sum() > 0
            assert model.summary_token.grad.abs().sum() > 0
        else:
            assert all(not p.grad.any() for p in model.parameters())

    set_seed(31)
    valid = torch.tensor([[True] * 5, [False, True, False, True, False], [False] * 5], device=device)
    hand_scene = {
        'hand_query': torch.randn(3, 6, 192, device=device, requires_grad=True),
        'relation_update': torch.randn(3, 6, 192, device=device, requires_grad=True),
        'geometry_descriptor': torch.rand(3, 6, 7, device=device),
    }
    baseline = hand_scene['hand_query'].detach() + hand_scene['relation_update'].detach()
    bf16 = device.type == 'cpu' or (device.type == 'cuda' and torch.cuda.is_bf16_supported())
    encoders = (XHandForceEncoder, XHandTaxelEncoder, XHandPatchEncoder)
    for encoder_type in encoders:
        model = GatedTactileFusion(tactile_encoder=encoder_type()).to(device)
        assert model.out_dim == 192 and model.out_shape == (6, 192)
        tactile = torch.randn(3, *model.input_shape, device=device, requires_grad=True)
        output = model(hand_scene, tactile=tactile, tactile_valid=valid, return_intermediate=True)
        shapes = {'interaction_tokens': (3, 6, 192), 'tactile_tokens': (3, 5, 64),
                  'tactile_summary': (3, 64), 'tactile_valid': (3, 5), 'gate': (3, 6),
                  'tactile_update': (3, 6, 192)}
        assert output.keys() == shapes.keys()
        for key, shape in shapes.items():
            assert output[key].shape == shape and torch.isfinite(output[key]).all(), key
        torch.testing.assert_close(output['tactile_valid'], valid)
        torch.testing.assert_close(output['gate'][:2], torch.full_like(output['gate'][:2], .1), rtol=0, atol=.02)
        torch.testing.assert_close(output['interaction_tokens'], baseline + output['tactile_update'], rtol=0, atol=0)
        torch.testing.assert_close(output['interaction_tokens'][2], baseline[2], rtol=0, atol=0)
        assert not output['gate'][2].any() and not output['tactile_update'][2].any()
        check_backward(model, output, active=True)
        assert torch.isfinite(tactile.grad).all() and not tactile.grad[~valid].any()
        assert tactile.grad[valid].abs().sum() > 0
        for key in ('hand_query', 'relation_update'):
            assert torch.isfinite(hand_scene[key].grad).all()

        with torch.no_grad():
            default = model(hand_scene, tactile=tactile, tactile_valid=valid)
            assert set(default) == {'interaction_tokens'}
            torch.testing.assert_close(default['interaction_tokens'], output['interaction_tokens'], rtol=0, atol=0)
            # A missing frame cannot change other frames, and masked finite placeholders are irrelevant.
            for row in range(3):
                single = model({k: v[row:row + 1] for k, v in hand_scene.items()},
                               tactile=tactile[row:row + 1], tactile_valid=valid[row:row + 1],
                               return_intermediate=True)
                torch.testing.assert_close(single, {k: v[row:row + 1] for k, v in output.items()})
            changed = tactile.detach().clone()
            changed[~valid] = 17.0
            torch.testing.assert_close(
                model(hand_scene, tactile=changed, tactile_valid=valid, return_intermediate=True),
                output, rtol=0, atol=0,
            )
            zero = model(hand_scene, tactile=torch.zeros_like(tactile), return_intermediate=True)
            assert zero['tactile_valid'].all() and (zero['gate'] > 0).all()
            assert (zero['tactile_update'].abs().sum(-1) > 0).all()
            torch.testing.assert_close(
                zero, model(hand_scene, tactile=torch.zeros_like(tactile),
                            tactile_valid=torch.ones_like(valid), return_intermediate=True), rtol=0, atol=0,
            )
            no_geometry = {**hand_scene, 'relation_update': torch.zeros_like(baseline),
                           'geometry_descriptor': torch.zeros_like(hand_scene['geometry_descriptor'])}
            touch = model(no_geometry, tactile=tactile, tactile_valid=valid, return_intermediate=True)
            # All six queries can update even when their own finger or all geometric evidence is absent.
            assert (touch['gate'][:2] > 0).all()
            assert (touch['tactile_update'][:2].abs().sum(-1) > 0).all()
            zero_touch = model(no_geometry, tactile=torch.zeros_like(tactile), tactile_valid=valid)
            assert (touch['interaction_tokens'][:2] - zero_touch['interaction_tokens'][:2]).abs().sum() > 0

        for values, mask in ((tactile, torch.zeros_like(valid)), (None, None), (None, torch.zeros_like(valid))):
            missing = model(hand_scene, tactile=values, tactile_valid=mask, return_intermediate=True)
            torch.testing.assert_close(missing['interaction_tokens'], baseline, rtol=0, atol=0)
            assert not missing['gate'].any() and not missing['tactile_update'].any()
            check_backward(model, missing, active=False)
        expect_value_error(lambda: model(hand_scene, tactile_valid=valid))
        if bf16:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                mixed = model(hand_scene, tactile=tactile, tactile_valid=valid, return_intermediate=True)
            assert all(torch.isfinite(value).all() for value in mixed.values())
            torch.testing.assert_close(mixed['interaction_tokens'][2], baseline[2], rtol=0, atol=0)
            assert not mixed['gate'][2].any() and not mixed['tactile_update'][2].any()
            check_backward(model, mixed, active=True)

    return {'status': 'passed', 'device': str(device), 'encoders': [cls.__name__ for cls in encoders],
            'bf16': 'passed' if bf16 else 'not_run: unsupported device',
            'checks': 'gradients, masking, valid zeros, missing graph, geometric independence'}


def validate_hand_interaction_chain(device):
    from dexmani_policy.agents.obs_encoder.interaction.gated_tactile_fusion import GatedTactileFusion
    from dexmani_policy.agents.obs_encoder.interaction.hand_scene_encoder import HandSceneEncoder
    from dexmani_policy.agents.obs_encoder.proprio import HandStateEncoder

    set_seed(41)
    state = HandStateEncoder().to(device)
    scene = HandSceneEncoder(surface_k=4).to(device)
    fusion = GatedTactileFusion().to(device)
    observations, rotation, _ = _hand_observations(device)
    anchors = torch.cat((observations['eef_pose'][:, None, :3], observations['fingertip_xyz']), 1)
    points = torch.cat((anchors + .002, anchors + .004), 1)
    patches = {'tokens': torch.randn(2, 3, 192, device=device, requires_grad=True),
               'xyz': points[:, :3], 'mask': torch.tensor([[True] * 3, [False] * 3], device=device)}
    point_mask = patches['mask'][:, :1].expand(-1, 12)
    tactile = torch.randn(2, *fusion.input_shape, device=device, requires_grad=True)
    tactile_valid = torch.tensor([[True, False, True, False, True], [True] * 5], device=device)
    loss_weights = torch.randn(2, 6, 192, device=device)
    bf16 = device.type == 'cpu' or (device.type == 'cuda' and torch.cuda.is_bf16_supported())
    for mixed in ((False, True) if bf16 else (False,)):
        for module in (state, scene, fusion):
            module.zero_grad(set_to_none=True)
        patches['tokens'].grad = tactile.grad = None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mixed):
            hand = state(**observations)
            encoded = scene(points, patches, hand['anchor_xyz'], hand['hand_tokens'],
                            rotation_world_from_eef=hand['rotation_world_from_eef'],
                            point_mask=point_mask, return_intermediate=True)
            fused = fusion(encoded, tactile=tactile, tactile_valid=tactile_valid, return_intermediate=True)
            missing = fusion(encoded, return_intermediate=True)
        assert encoded['hand_query'] is hand['hand_tokens']
        for key in ('anchor_xyz', 'rotation_world_from_eef'):
            assert hand[key].dtype == torch.float32
        for key in ('interaction_center', 'geometry_descriptor', 'surface_relative', 'min_observed_distance'):
            assert encoded[key].dtype == torch.float32
        torch.testing.assert_close(hand['rotation_world_from_eef'], rotation)
        torch.testing.assert_close(encoded['interaction_center'], anchors, rtol=0, atol=0)
        assert not encoded['relation_update'][1].any() and not encoded['geometry_descriptor'][1].any()
        assert torch.isinf(encoded['min_observed_distance'][1]).all()
        assert fused['interaction_tokens'].shape == (2, 6, 192)
        assert torch.isfinite(fused['interaction_tokens']).all()
        assert (fused['tactile_update'][1].abs().sum(-1) > 0).all()
        baseline = hand['hand_tokens'].float() + encoded['relation_update'].float()
        torch.testing.assert_close(fused['interaction_tokens'], baseline + fused['tactile_update'], rtol=0, atol=0)
        torch.testing.assert_close(missing['interaction_tokens'], baseline, rtol=0, atol=0)
        assert not missing['gate'].any() and not missing['tactile_update'].any()

        (fused['interaction_tokens'] * loss_weights).mean().backward()
        for module in (state, scene, fusion):
            _check_finite_gradients(module)
            assert any(parameter.grad.abs().sum() > 0 for parameter in module.parameters())
        for name in ('position_mlp', 'arm_mlp', 'finger_mlp'):
            assert all(parameter.grad.abs().sum() > 0 for parameter in getattr(state, name).parameters()), name
        assert torch.isfinite(patches['tokens'].grad).all() and patches['tokens'].grad[0].abs().sum() > 0
        assert not patches['tokens'].grad[1].any()
        assert torch.isfinite(tactile.grad).all() and not tactile.grad[~tactile_valid].any()
        assert tactile.grad[tactile_valid].abs().sum() > 0
    return {'status': 'passed', 'device': str(device), 'bf16': 'passed' if bf16 else 'not_run: unsupported device',
            'checks': 'measured hand-state/scene/tactile chain, single residuals, gradients, empty geometry, FP32 metadata'}


def validate_common_contracts(device):
    return {
        'hand_state_encoder': validate_hand_state_encoder(device),
        'hand_scene_encoder': validate_hand_scene_encoder(device),
        'gated_tactile_fusion': validate_gated_tactile_fusion(device),
        'hand_interaction_chain': validate_hand_interaction_chain(device),
        'config_targets': validate_config_targets(),
        'experiment_paths': validate_experiment_paths(),
        'resume_facts': validate_resume_facts(),
        'action_recipes': validate_action_recipes(),
        'ema': validate_ema_updates(device),
        'fps': validate_fps(device),
        'datasets': validate_dataset_layouts(),
        'hand_priors': validate_hand_prior_contracts(),
    }
