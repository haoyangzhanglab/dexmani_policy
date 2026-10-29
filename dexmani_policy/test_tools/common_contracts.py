"""Shared action, EMA, point-cloud, dataset and experiment regression checks."""

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
    from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS
    from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
    from dexmani_policy.datasets.common.robot_dataset import RobotDataset
    from dexmani_policy.datasets.aux.r3d_dataset import R3DAuxDataset
    from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
    from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
    from dexmani_policy.datasets.obs.cordvip_dataset import CordViPDataset
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
        reads = dict.fromkeys(reads, 0)
        with patch.object(ReplayBuffer, 'read', read):
            train, val, normalizer, _ = prepare_policy_data(cfg)
        assert reads == {'action': 16, 'joint_state': 0}
        torch.testing.assert_close(normalizer['hand'].unnormalize(train), torch.from_numpy(arrays['action'][:8, 7:19]))
        torch.testing.assert_close(normalizer['hand'].unnormalize(val), torch.from_numpy(arrays['action'][8:16, 7:19]))
        for bad in (np.nan, np.inf):
            root['data/action'][9, 10] = bad
            with patch.object(LinearNormalizer, 'fit_field', side_effect=AssertionError('validate before fitting')):
                for build in (
                    lambda: HandChunkDataset(directory, horizon=4, data_semantics=HAND_ACTION_SEMANTICS, split_manifest=split),
                    lambda: prepare_policy_data(cfg),
                ):
                    try:
                        build()
                    except ValueError as error:
                        assert 'row=9, channel=(10,)' in str(error), error
                    else:
                        raise AssertionError('Invalid validation frame was accepted')
    return {'status': 'passed', 'windows': windows, 'hand_reads': 'one pass per selected field'}


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

    cfg = OmegaConf.create(dict(agent={}, dataset={}, training=dict(lr_scheduler='cosine')))
    model = SimpleNamespace(inference_metadata={'control_action_dim': 19}, effective_config={})
    description = dict(
        normalization_dims={'action': 19}, real_runtime={'dt': 0.05},
        hand_data=dict(contract={'dt': 0.05}, provenance={'source_id': 'same', 'zarr_path': '/old/data.zarr'}),
        split_manifest=dict(episode_ends=[4], train_indices=[0], val_indices=[]),
    )
    saved = OmegaConf.to_container(assemble_effective_config(cfg, description, model), resolve=True)
    checkpoint = SimpleNamespace(effective_config=saved)
    moved = copy.deepcopy(description)
    moved['hand_data']['provenance']['zarr_path'] = '/new/data.zarr'
    resumed = assemble_effective_config(OmegaConf.create(saved), moved, model, checkpoint=checkpoint)
    assert resumed.hand_data.provenance.zarr_path == '/new/data.zarr'
    assert saved['hand_data']['provenance']['zarr_path'] == '/old/data.zarr'
    assert moved['hand_data']['provenance']['zarr_path'] == '/new/data.zarr'
    assert OmegaConf.is_readonly(resumed)
    for key in ('normalization_dims', 'real_runtime', 'hand_data', 'policy_metadata'):
        changed, actual_model = copy.deepcopy(moved), copy.deepcopy(model)
        if key == 'policy_metadata':
            actual_model.inference_metadata['control_action_dim'] = 18
        elif key == 'normalization_dims':
            changed[key]['action'] = 18
        elif key == 'real_runtime':
            changed[key]['dt'] = 0.1
        else:
            changed[key]['provenance']['source_id'] = 'different'
        expect_value_error(lambda: assemble_effective_config(OmegaConf.create(saved), changed, actual_model, checkpoint=checkpoint))
    for key in ('real_runtime', 'hand_data'):
        changed = {name: value for name, value in moved.items() if name != key}
        expect_value_error(lambda: assemble_effective_config(OmegaConf.create(saved), changed, model, checkpoint=checkpoint))
    return {'status': 'passed', 'checks': 'relocation, actual data/model changes and missing metadata'}


def validate_common_contracts(device):
    return {
        'config_targets': validate_config_targets(),
        'experiment_paths': validate_experiment_paths(),
        'resume_facts': validate_resume_facts(),
        'action_recipes': validate_action_recipes(),
        'ema': validate_ema_updates(device),
        'fps': validate_fps(device),
        'datasets': validate_dataset_layouts(),
    }
