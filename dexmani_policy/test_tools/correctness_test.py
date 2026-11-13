"""Cheap CPU regressions for resume, optimizer and publication boundaries."""
import copy
import io
import json
import tempfile
import unittest
import weakref
from contextlib import closing, nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from dexmani_policy.agents.hand_prior.codebook import load_codebook
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.deployment.runtime import PolicyRuntime
from dexmani_policy.evaluation.selection import run_pipeline
from dexmani_policy.training.common.config import load_policy_config, validate_config
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.pretrain.obs.common import export_encoder, save_run
from dexmani_policy.training.trainer import PolicyTrainer


class CorrectnessTest(unittest.TestCase):
    def test_dp_resnet_recipe(self):
        from dexmani_policy.agents.obs_encoder.rgb.build import build_rgb_encoder
        cfg = load_policy_config('dp')
        self.assertIsNone(cfg.initialization.rgb)
        self.assertEqual(list(cfg.dataset.rgb_preprocess_size), [84, 84])
        self.assertEqual(list(cfg.dataset.rgb_random_crop_size), [76, 76])
        self.assertEqual((cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps), (16, 2, 8))
        self.assertEqual((cfg.agent.prediction_type, cfg.agent.num_inference_steps), ('sample', 10))
        self.assertEqual(list(cfg.agent.down_dims), [256, 512, 1024])
        self.assertEqual(cfg.agent.rgb_backbone_name, 'resnet')
        self.assertEqual(dict(cfg.agent.rgb_backbone_config), dict(
            model_name='resnet18', tune_mode='full', norm_mode='group_norm', image_size=[76, 76]))
        encoder, processor = build_rgb_encoder(cfg.agent.rgb_backbone_name, dict(cfg.agent.rgb_backbone_config))
        encoder.initialize_weights(None)
        self.assertTrue(all(p.requires_grad for p in encoder.parameters()))
        self.assertTrue(any(isinstance(m, torch.nn.GroupNorm) for m in encoder.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in encoder.modules()))
        self.assertEqual(processor.process_images(torch.zeros(1, 3, 76, 76)).shape[-2:], (76, 76))

    def test_dp_dino_restoration(self):
        import hydra
        from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
        from dexmani_policy.common.checkpoint import restore_policy

        cfg = load_policy_config('dp')
        cfg.agent.rgb_backbone_name = 'dino'
        # A tiny saved DINO recipe exercises restoration without training or external weights.
        cfg.agent.rgb_backbone_config = dict(tune_mode='lora', image_size=[224, 224],
            architecture=dict(model_type='dinov2', hidden_size=24, num_hidden_layers=1,
                              num_attention_heads=4, image_size=224, patch_size=16))
        cfg.agent.down_dims = [16, 32]
        cfg.agent.diffusion_step_embed_dim = 16
        model = hydra.utils.instantiate(cfg.agent, _convert_='all')
        dims = dict(action=19, joint_state=19, rgb=3)
        model.configure_normalizer(dims, dict(action=(19,), joint_state=(19,)))
        saved_agent = OmegaConf.to_container(cfg.agent, resolve=True)
        saved_agent.update(model.effective_config)
        saved = dict(agent=saved_agent, normalization_dims=dims,
                     normalization=OmegaConf.to_container(cfg.normalization))
        with patch.object(ViTEncoder, 'initialize_weights', side_effect=AssertionError('external weights')):
            restored = restore_policy(saved, copy.deepcopy(model.state_dict()), 'cpu')
        self.assertIsInstance(restored.obs_encoder.rgb_encoder, ViTEncoder)
        self.assertEqual(restored.obs_encoder.image_processor.image_size, (224, 224))
        torch.testing.assert_close(restored.obs_encoder.rgb_encoder.backbone.embeddings.cls_token,
                                   model.obs_encoder.rgb_encoder.backbone.get_base_model().embeddings.cls_token)

    def test_evaluation_seed_split(self):
        from dexmani_policy.evaluation.seeds import split_evaluation_seeds
        pool = list(reversed(range(20)))
        small, selection = split_evaluation_seeds(pool, 42, 3, 4)
        large, same_selection = split_evaluation_seeds(pool, 42, 16, 4)
        standalone, empty = split_evaluation_seeds(pool, 42, 16)
        self.assertEqual(small, large[:3])
        self.assertEqual(selection, same_selection)
        self.assertFalse(set(large) & set(selection))
        self.assertEqual(standalone, large)
        self.assertEqual(empty, [])
        self.assertEqual(split_evaluation_seeds(sorted(pool), 42, 16, 4), (large, selection))
        self.assertEqual(pool, list(reversed(range(20))))
        with self.assertRaisesRegex(ValueError, 'Seed pool has'):
            split_evaluation_seeds(pool, 42, 17, 4)

    def test_afro_initialization_summary(self):
        from dexmani_policy.train import build_trainer
        for kind, frozen in (('random', True), ('encoder', True), ('policy', False), ('resume', True), ('obs_lr', True)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                cfg = load_policy_config('afro', [f'workspace.output_dir={directory}', 'workspace.wandb_cfg=null',
                                                'training.device=cpu', 'training.use_compile=false'])
                with open_dict(cfg):
                    cfg.control_dt_s = .05
                    cfg.normalization_dims = {'action': 19, 'joint_state': 19, 'point_cloud': 3}
                cfg.agent.freeze_visual = kind not in ('policy', 'obs_lr')
                cfg.initialization.encoder = 'pretrained/encoder.pt' if kind == 'encoder' else None
                cfg.initialization.policy = 'source/policy.pt' if kind == 'policy' else None
                if kind == 'obs_lr':
                    cfg.optimizer.obs_lr = 0
                model = SimpleNamespace(obs_encoder=SimpleNamespace(pc_encoder=torch.nn.Linear(1, 1)),
                                        normalization_dims=dict(cfg.normalization_dims), effective_config={})
                model.obs_encoder.pc_encoder.requires_grad_(not cfg.agent.freeze_visual)
                saved = SimpleNamespace(effective_config=OmegaConf.to_container(cfg, resolve=True), global_step=9)
                dataset = SimpleNamespace(episode_store=SimpleNamespace(path='synthetic.zarr'))
                def trainer(config, model, dataset, **kwargs):
                    if config.optimizer.obs_lr == 0:
                        model.obs_encoder.pc_encoder.requires_grad_(False)
                    return SimpleNamespace(model=model)
                output = io.StringIO()
                with patch('dexmani_policy.train.build_dataset', return_value=(dataset, dict(
                        control_dt_s=.05, field_dims=dict(cfg.normalization_dims)))), \
                        patch('dexmani_policy.train._prepare_policy', return_value=model) as prepare, \
                        patch('dexmani_policy.train.load_checkpoint', return_value=saved), \
                        patch('dexmani_policy.train.PolicyTrainer', side_effect=trainer), redirect_stdout(output):
                    build_trainer(cfg, checkpoint=saved if kind == 'resume' else None)
                expected = {'random': 'random initialization', 'obs_lr': 'random initialization',
                            'encoder': 'encoder: pretrained/encoder.pt', 'policy': 'policy checkpoint: source/policy.pt',
                            'resume': 'resume checkpoint (step=9)'}[kind]
                self.assertEqual(output.getvalue().strip(), f'[AFRO] weights={expected}, visual_frozen={frozen}')
                self.assertIs(prepare.call_args.kwargs['checkpoint'], saved if kind in ('policy', 'resume') else None)

    def test_pretrain_finite_boundary_before_teacher(self):
        from dexmani_policy.training.pretrain.obs import common
        # Exercise the real boundary with one scalar and mocked data/publication, not an AFRO training run.
        class Scalar(torch.nn.Module):
            def __init__(self, offset, gradient):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.weight.register_hook(lambda grad: torch.full_like(grad, gradient))
                self.register_buffer('teacher', torch.tensor(0.))
                self.normalizer = SimpleNamespace(fields={})
                self.offset = offset
                self.update_teacher = Mock(side_effect=lambda epoch: self.teacher.add_(1))

            def forward(self, batch):
                loss = self.weight * 3 + self.offset
                return loss, {'loss': loss.detach()}

        for offset, gradient in ((0., 3.), (float('nan'), 3.), (float('inf'), 3.),
                                 (0., float('nan')), (0., float('inf'))):
            valid = offset == 0 and gradient == 3
            with self.subTest(offset=offset, gradient=gradient), tempfile.TemporaryDirectory() as directory:
                cfg = load_policy_config('pretrain/obs/afro', [f'workspace.output_dir={directory}',
                    'training.device=cpu', 'training.num_epochs=1', 'training.lr_warmup_steps=0'])
                model = Scalar(offset, gradient)
                optimizer = torch.optim.AdamW([model.weight], lr=.1)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
                dataset = MagicMock()
                dataset.episode_store.path = 'synthetic.zarr'
                dataset.get_validation_dataset.return_value = None
                dataset.__len__.return_value = 1
                with patch.object(common, 'load_run', return_value=(cfg, None, None)), \
                        patch.object(common.hydra.utils, 'instantiate', return_value=dataset), \
                        patch.object(common, 'build_train_loader', return_value=[{}]), \
                        patch.object(common, 'fit_normalizer'), patch.object(common, 'close_data_loader'), \
                        patch.object(common.torch.optim, 'AdamW', return_value=optimizer), \
                        patch.object(common, 'build_scheduler', return_value=scheduler), \
                        patch.object(common, 'ExperimentLogger'), patch.object(common, 'save_run') as save, \
                        patch.object(common, 'export_encoder') as export, redirect_stdout(io.StringIO()):
                    with nullcontext() if valid else self.assertRaisesRegex(RuntimeError, 'Non-finite loss or gradient norm at step 0'):
                        common.train('afro', lambda normalization: model)
                self.assertEqual(scheduler.last_epoch, int(valid))
                self.assertEqual(model.teacher.item(), int(valid))
                if valid:
                    self.assertLess(model.weight.item(), 1.)
                    self.assertEqual(save.call_args.args[-1], 1)
                    model.update_teacher.assert_called_once_with(0)
                    export.assert_called_once()
                else:
                    self.assertEqual(model.weight.item(), 1.)
                    self.assertFalse(optimizer.state)
                    model.update_teacher.assert_not_called()
                    save.assert_not_called()
                    export.assert_not_called()

    def test_evaluation_incremental_publication(self):
        from dexmani_policy.evaluation.metrics import summarize_episodes
        from dexmani_policy.evaluation.rollout import evaluate_policy

        agent = SimpleNamespace(n_obs_steps=1, n_action_steps=1, future_steps=1, requires_prev_action=False,
                                force_predictor=SimpleNamespace(num_inference_steps=7))
        cfg = dict(env_runner=dict(task_name='test'), dataset={}, control_dt_s=.05,
                   normalization_dims={'joint_state': 19}, eval=dict(use_ema=False, inference_steps=2))
        loaded = SimpleNamespace(effective_config=cfg, model_state={}, global_step=10)
        outcomes = [(11, True, 2, 3), (22, False, None, 9), (33, True, 4, 6)]
        seeds = [row[0] for row in outcomes]
        statistics = [None,
            dict(n_episodes=1, n_success=1, success_rate=1., avg_steps=2., avg_steps_all=3.),
            dict(n_episodes=2, n_success=1, success_rate=.5, avg_steps=2., avg_steps_all=6.),
            dict(n_episodes=3, n_success=2, success_rate=2 / 3, avg_steps=3., avg_steps_all=6.)]
        frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        for video, failure, count, cleanup_failure in (
            (False, None, 3, False), (False, 'episode', 0, False), (False, 'episode', 2, False),
            (False, None, 3, True), (True, None, 3, False),
            (True, 'read', 2, False), (True, 'save', 2, False),
            (True, 'save', 2, True), (True, None, 3, True),
        ):
            with self.subTest(video=video, failure=failure, count=count, cleanup=cleanup_failure), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'eval/result_details.json'
                primary = KeyboardInterrupt('interrupted') if failure == 'episode' else RuntimeError('video failed')
                cleanup = RuntimeError('close failed')
                expected = [dict(task_name='test', seed=seed, success=success, steps=steps,
                                 total_steps=total, video=str(root / f'eval/videos/test/episode_{seed}.mp4')
                                 if video and not (failure and seed == 22) else None)
                            for seed, success, steps, total in outcomes[:count]]
                env = SimpleNamespace(action_cnt=0, video_fps=16)

                def check_snapshot(index):
                    record = json.loads(path.read_text())
                    self.assertFalse(record['completed'])
                    self.assertEqual(record['episodes'], expected[:index])
                    self.assertEqual(record['statistics'], summarize_episodes(expected[:index]) if index else None)
                    if index:
                        for key, value in statistics[index].items():
                            self.assertEqual(record['statistics'][key], value)

                def run_episode(agent, current_env, seed, inference_steps, *, action_steps):
                    index = seeds.index(seed)
                    check_snapshot(index)
                    if failure == 'episode' and index == count:
                        raise primary
                    current_env.seed = seed
                    _, success, steps, current_env.action_cnt = outcomes[index]
                    return success, steps

                def get_video():
                    if failure == 'read' and env.seed == 22:
                        raise primary
                    return frames

                def save_video(data, fps, destination):
                    self.assertIs(data, frames)
                    self.assertEqual(fps, 16)
                    if failure == 'save' and env.seed == 22:
                        raise primary
                    return str(destination)

                def close():
                    check_snapshot(count)
                    if cleanup_failure:
                        raise cleanup

                env.get_video, env.close = Mock(side_effect=get_video), Mock(side_effect=close)
                with patch('dexmani_policy.evaluation.rollout.load_checkpoint', return_value=loaded), \
                        patch('dexmani_policy.evaluation.rollout.restore_policy', return_value=agent), \
                        patch.object(SimRunner, 'make_env', return_value=env) as make_env, \
                        patch.object(SimRunner, '_run_episode', side_effect=run_episode) as episode, \
                        patch('dexmani_policy.env_runner.video.save_episode_video', side_effect=save_video) as save:
                    if failure or cleanup_failure:
                        error = primary if failure else cleanup
                        with self.assertLogs(level='ERROR') if failure and cleanup_failure else nullcontext() as logs:
                            with self.assertRaises(type(error)) as caught:
                                evaluate_policy(root / 'candidate.pt', seeds=seeds, device='cpu',
                                                video=video, output_dir=path.parent)
                        self.assertIs(caught.exception, error)
                        if logs:
                            self.assertIn('close failed', '\n'.join(logs.output))
                    else:
                        result, returned_path = evaluate_policy(root / 'candidate.pt', seeds=seeds,
                                                               device='cpu', video=video, output_dir=path.parent)
                        self.assertEqual(returned_path, path)
                        self.assertEqual(result, json.loads(path.read_text()))
                record = json.loads(path.read_text())
                self.assertEqual(record['episodes'], expected)
                self.assertEqual(record['statistics'], summarize_episodes(expected) if count else None)
                self.assertEqual(record['completed'], not (failure or cleanup_failure))
                self.assertEqual((record['global_step'], record['weights'], record['force_inference_steps']), (10, 'raw', 7))
                self.assertEqual(record['seeds'], seeds)
                self.assertEqual([call.args[2] for call in episode.call_args_list], seeds[:count + int(failure == 'episode')])
                self.assertEqual(env.get_video.call_count, count if video else 0)
                self.assertEqual(save.call_count, (count - int(failure == 'read')) if video else 0)
                make_env.assert_called_once_with(record_video=video)
                env.close.assert_called_once()

    def test_tactile_metadata_does_not_change_samples(self):
        from dexmani_policy.datasets.variants.tactile import InterActDexDataset
        from dexmani_policy.test_tools.smoke_test import make_source
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.zarr'
            source = make_source(path)
            force = np.arange(12 * 5 * 3, dtype=np.float32).reshape(12, 5, 3) / 255
            source.create_dataset('data/finger_tactile_force', data=force)
            source.attrs.update(format='dexmani.real.canonical', tactile_scale_divisor=255., dt=.05)
            options = dict(sensor_modalities=['finger_tactile_force'], horizon=4, obs_horizon=2,
                           pad_before=1, pad_after=1, future_force_horizon=4, include_future_force=True)
            baseline = InterActDexDataset(path, **options)
            samples = [baseline[i] for i in range(len(baseline))]
            for divisor in (None, 1.):
                with self.subTest(divisor=divisor):
                    if divisor is None:
                        del source.attrs['tactile_scale_divisor']
                    else:
                        source.attrs['tactile_scale_divisor'] = divisor
                    dataset = InterActDexDataset(path, **options)
                    self.assertEqual(dataset.episode_store.attrs, dict(source.attrs))
                    np.testing.assert_array_equal(dataset.episode_store.read('finger_tactile_force', slice(None)), force)
                    np.testing.assert_array_equal(dataset.sampler.indices, baseline.sampler.indices)
                    for i, expected in enumerate(samples):
                        torch.testing.assert_close(dataset[i], expected, rtol=0, atol=0)
                        rows = dataset.sampler.sample_source_rows(i)[:2]
                        np.testing.assert_array_equal(dataset[i]['obs']['finger_tactile_force'], force[rows])

    def test_evaluation_write_error_survives_cleanup_failure(self):
        import os
        from dexmani_policy.evaluation.rollout import evaluate_policy
        agent = SimpleNamespace(n_obs_steps=1, n_action_steps=1, future_steps=1, requires_prev_action=False)
        cfg = dict(env_runner=dict(task_name='test'), dataset={}, control_dt_s=.05,
                   normalization_dims={'joint_state': 19}, eval=dict(use_ema=False, inference_steps=2))
        loaded = SimpleNamespace(effective_config=cfg, model_state={}, global_step=10)
        primary = OSError('disk full')
        replace = os.replace
        def publish(source, target):
            if publication.call_count == 3:
                raise primary
            replace(source, target)
        for video_failure in (False, True):
            with self.subTest(video_failure=video_failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = SimpleNamespace(action_cnt=3, close=Mock(side_effect=RuntimeError('close failed')),
                                      get_video=Mock(return_value=np.zeros((2, 2, 2, 3), dtype=np.uint8)), video_fps=16)
                with patch('dexmani_policy.evaluation.rollout.load_checkpoint', return_value=loaded), \
                        patch('dexmani_policy.evaluation.rollout.restore_policy', return_value=agent), \
                        patch.object(SimRunner, 'make_env', return_value=env), \
                        patch.object(SimRunner, '_run_episode', return_value=(True, 2)) as episode, \
                        patch('dexmani_policy.env_runner.video.save_episode_video',
                              side_effect=['first.mp4', RuntimeError('video failed')]), \
                        patch('dexmani_policy.common.io_util.os.replace', side_effect=publish) as publication:
                    with self.assertLogs(level='ERROR') as logs, self.assertRaises(OSError) as caught:
                        evaluate_policy(root / 'candidate.pt', seeds=[11, 22, 33], device='cpu',
                                        video=video_failure, output_dir=root / 'eval')
                self.assertIs(caught.exception, primary)
                self.assertIn('close failed', '\n'.join(logs.output))
                env.close.assert_called_once()
                self.assertEqual(episode.call_count, 2)
                record = json.loads((root / 'eval/result_details.json').read_text())
                self.assertFalse(record['completed'])
                self.assertEqual([row['seed'] for row in record['episodes']], [11])
                self.assertEqual(record['statistics']['n_episodes'], 1)
                self.assertFalse(list((root / 'eval').glob('.dexmani-publish-*')))

    def test_dataset_semantics(self):
        from dexmani_policy.test_tools.smoke_test import make_source
        from dexmani_policy.training.common.data import build_dataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.zarr'
            source = make_source(path)
            cfg = load_policy_config('dp3', [f'zarr_path={path}', 'horizon=4',
                                            'n_obs_steps=2', 'n_action_steps=2'])
            for field, values, message in (
                ('task_name', ['other'], 'Dataset task mismatch'),
                ('action_space', ['ee'], 'Simulation dataset must use joint actions'),
                ('dt', [float('nan'), float('inf'), -float('inf'), 0, -.05], 'Invalid dataset dt'),
            ):
                original = source.attrs[field]
                for value in values:
                    with self.subTest(field=field, value=value):
                        source.attrs[field] = value
                        with self.assertRaisesRegex(ValueError, message):
                            build_dataset(cfg)
                source.attrs[field] = original
            del source.attrs['dt']
            with self.assertRaisesRegex(KeyError, 'dt'):
                build_dataset(cfg)

    def test_point_counts_and_contact_head_boundary(self):
        import zarr
        from dexmani_policy.training.common.data import build_dataset
        from dexmani_policy.datasets.preprocess.contact_map import generate
        from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.zarr'
            source = zarr.open_group(str(path), mode='w')
            source.attrs.update(task_name='pick_apple_messy', action_space='joint', dt=.05)
            source.create_dataset('meta/episode_ends', data=np.array([4]))
            for key, shape in (('joint_state', (4, 19)), ('action', (4, 19)),
                               ('hand_pointcloud', (4, 7, 6)), ('object_pointcloud', (4, 11, 6)),
                               ('contact_map', (4, 11))):
                source.create_dataset('data/' + key, data=np.ones(shape, np.float16))
            cfg = load_policy_config('cordvip', [f'zarr_path={path}', 'horizon=4', 'n_obs_steps=2', 'n_action_steps=2'])
            dataset, description = build_dataset(cfg)
            self.assertEqual(set(description), {'field_dims', 'control_dt_s'})
            self.assertEqual(description['control_dt_s'], .05)
            self.assertIs(type(description['control_dt_s']), float)
            self.assertEqual(description['field_dims']['object_pointcloud'], 3)
            self.assertEqual(dataset[0]['obs']['object_pointcloud'].shape, (2, 11, 3))
            self.assertEqual(dataset[0]['obs']['object_pointcloud'].dtype, torch.float32)
            with self.assertRaisesRegex(ValueError, '1024'):
                generate(path, Path(directory) / 'contact.zarr')
            with self.assertRaisesRegex(ValueError, '1024'):
                CordViPPretrainDataset(path)

    def test_cordvip_pretrain_windows(self):
        import hydra
        import zarr
        from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
        with tempfile.TemporaryDirectory() as directory:
            source = zarr.open_group(directory, mode='w')
            source.attrs.update(task_name='pick_apple_messy', action_space='joint', dt=.05)
            source.create_dataset('meta/episode_ends', data=np.array([6, 12]))
            shapes = {'joint_state': (19,), 'hand_pointcloud': (7, 3),
                      'object_pointcloud': (1024, 3), 'contact_map': (1024,)}
            for key, shape in shapes.items():
                values = np.broadcast_to(np.arange(12, dtype=np.float32).reshape(
                    (12,) + (1,) * len(shape)), (12, *shape))
                source.create_dataset('data/' + key, data=values)
            for n in range(1, 6):
                with self.subTest(n=n):
                    cfg = load_policy_config('pretrain/obs/cordvip', [
                        f'zarr_path={directory}', f'n_obs_steps={n}'])
                    dataset = hydra.utils.instantiate(cfg.dataset, _convert_='all')
                    self.assertEqual(len(dataset), 12)
                    self.assertEqual(dataset.pad_before, n - 1)
                    for i in range(12):
                        rows = np.maximum(np.arange(i - n + 1, i + 1), (i // 6) * 6)
                        np.testing.assert_array_equal(dataset.sampler.sample_source_rows(i), rows)
                        sample = dataset[i]
                        for key, value in {**sample['obs'], 'contact_map': sample['contact_map']}.items():
                            self.assertEqual(tuple(value.shape), (n, *shapes[key]))
                            np.testing.assert_array_equal(value, source['data/' + key][:][rows])
            default = CordViPPretrainDataset(directory)
            explicit = CordViPPretrainDataset(directory, horizon=4, obs_horizon=4, pad_before=3, pad_after=0)
            np.testing.assert_array_equal(default.sampler.indices, explicit.sampler.indices)
            for i in range(len(default)):
                np.testing.assert_array_equal(default.sampler.sample_source_rows(i), explicit.sampler.sample_source_rows(i))
            for padding in (0, 1):
                cfg = load_policy_config('pretrain/obs/cordvip', [
                    f'zarr_path={directory}', 'dataset.horizon=6',
                    f'+dataset.pad_before={padding}', '+dataset.pad_after=1'])
                explicit = hydra.utils.instantiate(cfg.dataset, _convert_='all')
                self.assertEqual((explicit.pad_before, explicit.pad_after), (padding, 1))
                np.testing.assert_array_equal(explicit.sampler.sample_source_rows(0),
                                              np.maximum(np.arange(6) - padding, 0))
            independent = CordViPPretrainDataset(directory, horizon=6, obs_horizon=2)
            self.assertEqual(independent.pad_before, 1)
            self.assertEqual(independent[0]['contact_map'].shape, (2, 1024))
            with self.assertRaisesRegex(ValueError, 'padding'):
                CordViPPretrainDataset(directory, horizon=2, obs_horizon=2, pad_before=3)

    def test_evaluation_without_environment_report(self):
        from dexmani_policy.evaluation.rollout import evaluate_policy
        observation = {'joint_state': np.zeros(19, np.float32)}
        envs = [SimpleNamespace(action_cnt=1, reset=Mock(return_value=(observation, {})),
                    step=Mock(return_value=(observation, 0, True, False,
                              {'success': success, 'success_condition': success})), close=Mock())
                for success in (True, False)]
        agent = torch.nn.Linear(1, 1)
        agent.n_obs_steps = agent.n_action_steps = agent.future_steps = 1
        agent.requires_prev_action = False
        agent.normalization_dims = {'joint_state': 19}
        agent.force_predictor = SimpleNamespace(num_inference_steps=7)
        cfg = dict(env_runner=dict(task_name='test', env_rebuild_interval=1),
                   eval=dict(use_ema=False, inference_steps=2), dataset={},
                   normalization_dims=agent.normalization_dims,
                   control_dt_s=.05)
        pending_envs = iter(envs)
        def make_env(runner, **kwargs):
            self.assertEqual(runner.control_dt_s, .05)
            return next(pending_envs)
        loaded = SimpleNamespace(effective_config=cfg, model_state={}, global_step=10)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'candidate.pt'
            checkpoint.touch()
            with patch('dexmani_policy.evaluation.rollout.load_checkpoint', return_value=loaded), \
                    patch('dexmani_policy.evaluation.rollout.restore_policy', return_value=agent), \
                    patch.object(SimRunner, 'make_env', autospec=True, side_effect=make_env), \
                    patch.object(SimRunner, 'predict_action_chunk', return_value=np.zeros((1, 19))):
                result, path = evaluate_policy(checkpoint, seeds=[11, 22], device='cpu',
                                               output_dir=Path(directory) / 'eval')
            self.assertEqual(result, json.loads(path.read_text()))
            self.assertTrue(result['completed'])
            self.assertTrue({'runner_options', 'environment', 'runtime'}.isdisjoint(result))
            self.assertEqual(result['seeds'], [11, 22])
            self.assertEqual(result['statistics']['success_rate'], .5)
            self.assertEqual(result['statistics']['avg_steps'], 1)
            self.assertEqual(result['force_inference_steps'], 7)
            self.assertEqual((result['global_step'], result['weights'], result['inference_steps']), (10, 'raw', 2))
            for env, seed in zip(envs, result['seeds']):
                env.reset.assert_called_once_with(seed=seed)
                env.close.assert_called_once()

    def test_optimizer_boundary(self):
        from dexmani_policy.training.common.optimization import EMA
        for limit in (0, 1.):
            trainer = PolicyTrainer.__new__(PolicyTrainer)
            model = torch.nn.Linear(1, 1, bias=False)
            parameter = model.weight
            parameter.data.fill_(1.)
            averaged = copy.deepcopy(model)
            trainer.trainable_parameters = [parameter]
            trainer.global_step = 0
            trainer.max_grad_norm = limit
            trainer.last_grad_norm = None
            trainer.optimizer = torch.optim.Adam([parameter], lr=.1)
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1, gamma=.9)
            trainer.ema_updater = EMA(model, averaged)
            parameter.grad = torch.full_like(parameter, 3.)
            trainer.optimizer_step(torch.tensor(True))
            self.assertEqual(trainer.last_grad_norm.item(), 3.)
            self.assertLess(parameter.item(), 1.)
            self.assertEqual(trainer.scheduler.last_epoch, 1)
            self.assertIsNone(parameter.grad)
            # Adam's first moment records the gradient actually passed to the optimizer.
            expected_grad = 3. if limit == 0 else 1.
            self.assertAlmostEqual(trainer.optimizer.state[parameter]['exp_avg'].item(), expected_grad * .1, places=6)
            torch.testing.assert_close(averaged.weight, parameter)
            for loss_value, gradient in ((float('nan'), 3.), (float('inf'), 3.),
                                         (1., float('nan')), (1., float('inf'))):
                with self.subTest(limit=limit, loss=loss_value, gradient=gradient):
                    parameter.grad = torch.full_like(parameter, gradient)
                    before = (parameter.detach().clone(), copy.deepcopy(trainer.optimizer.state_dict()),
                              copy.deepcopy(trainer.scheduler.state_dict()), averaged.weight.detach().clone())
                    norm = trainer.last_grad_norm
                    with patch.object(trainer.optimizer, 'zero_grad') as clear, \
                            patch.object(trainer.ema_updater, 'step', wraps=trainer.ema_updater.step) as ema:
                        with self.assertRaisesRegex(RuntimeError, 'Non-finite loss or gradient norm at step 0'):
                            trainer.optimizer_step(torch.isfinite(torch.tensor(loss_value)))
                        clear.assert_not_called()
                        ema.assert_not_called()
                    torch.testing.assert_close(parameter, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(trainer.optimizer.state_dict(), before[1], rtol=0, atol=0)
                    self.assertEqual(trainer.scheduler.state_dict(), before[2])
                    torch.testing.assert_close(averaged.weight, before[3], rtol=0, atol=0)
                    self.assertIs(trainer.last_grad_norm, norm)
                    self.assertEqual(trainer.global_step, 0)
        for value in (-1, float('nan'), float('inf')):
            cfg = load_policy_config('dp3')
            cfg.training.max_grad_norm = value
            with self.assertRaisesRegex(ValueError, 'max_grad_norm'):
                validate_config(cfg)

    def test_rtc(self):
        runtime = PolicyRuntime.__new__(PolicyRuntime)
        runtime.info = SimpleNamespace(control_action_dim=19, future_steps=3)
        runtime.generator = torch.Generator()
        prefix = np.zeros((3, 19))
        for value in (0.0, 0.5, False, True, -1, 4):
            with self.assertRaises(ValueError):
                runtime._prepare_rtc(prefix, value, 0)
        for value in (0, 1, 3):
            self.assertEqual(runtime._prepare_rtc(prefix, value, 0)['delay_steps'], value)
        def predict(observation, **kwargs):
            runtime._prepare_rtc(kwargs.get('rtc_prefix'), kwargs.get('delay_steps', 0), 0)
            torch.rand(1, generator=runtime.generator)
            return prefix
        runtime.predict = predict
        state = runtime.generator.get_state()
        with self.assertRaises(ValueError):
            runtime.warmup({}, rtc_delay=0.5, rtc_guidance_cap=1)
        self.assertTrue(torch.equal(state, runtime.generator.get_state()))

    def test_policy_accumulation_boundary(self):
        # A nonfinite early microbatch must remain visible even when the final loss is finite.
        for first_loss in (0., float('nan'), float('inf')):
            with self.subTest(first_loss=first_loss):
                trainer = PolicyTrainer.__new__(PolicyTrainer)
                trainer.device = torch.device('cpu')
                trainer.model = torch.nn.Linear(1, 1, bias=False)
                parameter = trainer.model.weight
                before = parameter.detach().clone()
                trainer.ema_model = None
                trainer.ema_updater = Mock()
                trainer.train_loader = [{}, {}]
                trainer.trainable_parameters = [parameter]
                trainer.optimizer = torch.optim.Adam([parameter], lr=.1)
                trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1)
                trainer.max_grad_norm = trainer.global_step = 0
                trainer.total_train_steps = trainer.log_interval_steps = 2
                trainer.batches_per_epoch = trainer.gradient_accumulation_steps = 2
                trainer.updates_per_epoch = 1
                trainer.milestone_steps = set()
                trainer.last_grad_norm = None
                trainer.save_checkpoint = Mock()
                trainer.log_step = Mock()
                offsets = iter((first_loss, 0.))
                def microbatch(batch, *, loss_divisor, log_metrics):
                    loss = parameter.sum() * 3 + next(offsets)
                    (loss / loss_divisor).backward()
                    return loss.detach(), {'loss': loss.detach()}
                trainer.train_microbatch = Mock(side_effect=microbatch)
                valid = first_loss == 0
                with patch('dexmani_policy.training.trainer.tqdm'), \
                        patch('dexmani_policy.training.trainer.close_data_loader'), \
                        patch('dexmani_policy.training.trainer.training_signal_handlers', return_value=nullcontext()):
                    with nullcontext() if valid else self.assertRaisesRegex(RuntimeError, 'Non-finite loss or gradient norm at step 0'):
                        trainer.train(max_updates=1)
                self.assertEqual(trainer.train_microbatch.call_count, 2)
                self.assertEqual(trainer.global_step, int(valid))
                self.assertEqual(trainer.scheduler.last_epoch, int(valid))
                if valid:
                    self.assertFalse(torch.equal(parameter, before))
                    trainer.ema_updater.step.assert_called_once_with(0)
                    trainer.save_checkpoint.assert_called_once()
                else:
                    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                    self.assertFalse(trainer.optimizer.state)
                    self.assertIsNone(trainer.last_grad_norm)
                    trainer.ema_updater.step.assert_not_called()
                    trainer.save_checkpoint.assert_not_called()
                    trainer.log_step.assert_not_called()

    def test_selection_survives_test_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'candidate.pt'
            checkpoint.touch()
            result = dict(statistics=dict(success_rate=1, avg_steps=2), global_step=1,
                          weights='raw', inference_steps=2, action_steps=2)
            with patch('dexmani_policy.evaluation.selection.load_checkpoint', return_value=SimpleNamespace(
                    effective_config={'env_runner': {'task_name': 'test', 'env_kwargs': {}},
                                      'control_dt_s': .05})), patch(
                    'dexmani_policy.evaluation.selection.load_seed_pool', return_value=list(range(5))), patch(
                    'dexmani_policy.evaluation.selection.evaluate_policy') as evaluate:
                for fail in (True, False):
                    output = root / str(fail)
                    evaluate.side_effect = [(result, output / 'candidate.json'),
                                            RuntimeError('test failed') if fail else (result, output / 'test.json')]
                    kwargs = dict(selection_episodes=1, test_episodes=1, output_dir=output)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'test failed'):
                            run_pipeline([checkpoint], **kwargs)
                    else:
                        run_pipeline([checkpoint], **kwargs)
                    record = json.loads((output / 'selection_result.json').read_text())
                    self.assertEqual(record['global_step'], 1)
                    self.assertEqual('test_result' in record, not fail)
                evaluate.reset_mock()
                evaluate.side_effect = RuntimeError('candidate failed')
                with patch('dexmani_policy.evaluation.selection.candidate_score') as score:
                    with self.assertRaisesRegex(RuntimeError, 'candidate failed'):
                        run_pipeline([checkpoint], selection_episodes=1, test_episodes=1,
                                     output_dir=root / 'candidate_failure')
                    score.assert_not_called()
                evaluate.assert_called_once()
                self.assertFalse((root / 'candidate_failure/selection_result.json').exists())

    def test_encoder_export(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Module()
            model.encoder = torch.nn.Linear(3, 2)
            model.normalizer = LinearNormalizer({'point_cloud': (3,)})
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            cfg = OmegaConf.create(dict(workspace=dict(output_dir=directory), training=dict(device='cpu'),
                                        normalization=dict(point_cloud='limits')))
            checkpoint = save_run(model, optimizer, scheduler, cfg, 7)
            output = Path(directory) / 'encoder.pt'
            export_encoder(checkpoint, output)
            artifact = torch.load(output, weights_only=True)
            self.assertEqual(set(artifact), {'encoder_state', 'normalizer_state', 'normalization', 'global_step'})
            model.encoder.load_state_dict(artifact['encoder_state'], strict=True)
            model.normalizer.load_state_dict(artifact['normalizer_state'], strict=True)
            self.assertEqual(artifact['global_step'], 7)
            loaded = torch.load(Path(directory) / 'checkpoints/latest.pt', weights_only=False)
            export_encoder(loaded, output)
            repeated = torch.load(output, weights_only=True)
            torch.testing.assert_close(repeated["encoder_state"], artifact["encoder_state"])
            self.assertEqual(repeated["global_step"], artifact["global_step"])

    def test_codebook_export(self):
        from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
        from dexmani_policy.training.pretrain.hand_prior.codebook import export_codebook
        model = HandVQVAE(hand_dim=12, loss_weight=[1.0] * 12, latent_dim=4, hidden_dim=8, num_groups=2, codebook_size=2, num_layers=1)
        normalizer = LinearNormalizer({'hand': (12,)})
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'best.pt', Path(directory) / 'codebook.npz'
            torch.save(dict(model_config=model.model_config, model_state_dict=model.state_dict(),
                            normalizer_state_dict=normalizer.state_dict()), source)
            export_codebook(source, output)
            with np.load(output, allow_pickle=False) as artifact:
                self.assertEqual(set(artifact.files), {
                    'sorted_hand_poses', 'hand_normalizer_scale', 'hand_normalizer_offset',
                })
            codebook, affine = load_codebook(output)
            self.assertEqual(codebook.num_codes, 4)
            torch.testing.assert_close(affine['scale'], torch.ones(12))

    def test_bounded_training_closes_active_iterator(self):
        from dexmani_policy.test_tools.smoke_test import make_source
        from dexmani_policy.train import build_trainer
        from dexmani_policy.training.common.data import close_data_loader
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'source.zarr')
            cfg = load_policy_config('dp3', [
                f'zarr_path={root / "source.zarr"}', f'workspace.output_dir={root / "run"}',
                'workspace.wandb_cfg=null', 'training.device=cpu',
                'horizon=4', 'n_obs_steps=2', 'n_action_steps=2', 'agent.num_points=8',
                'agent.down_dims=[16,32]', 'agent.diffusion_step_embed_dim=16',
                'training.use_compile=false', 'training.use_bfloat16=false',
                'training.loop.total_train_steps=2', 'training.lr_warmup_steps=0',
                'dataloader.batch_size=2', 'dataloader.num_workers=0',
                'dataloader.pin_memory=false', 'dataset.augmentation_cfg=null',
            ])
            with closing(build_trainer(cfg)) as trainer:
                with patch('dexmani_policy.training.trainer.close_data_loader', wraps=close_data_loader) as close:
                    trainer.train(max_updates=1)
                close.assert_called_once()
                loader, iterator = close.call_args.args
                self.assertIs(loader, trainer.train_loader)
                self.assertIsNotNone(iterator)
                self.assertFalse(trainer.model.training)

    def test_cleanup_attempts_all_resources(self):
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        trainer.train_loader = Mock()
        trainer.logger = ExperimentLogger.__new__(ExperimentLogger)
        trainer.logger.metrics_file = Mock()
        trainer.logger.metrics_file.close.side_effect = RuntimeError('metrics')
        trainer.logger.wandb_run = Mock()
        with patch('dexmani_policy.training.trainer.close_data_loader',
                   side_effect=RuntimeError('workers')) as close_loader:
            with self.assertRaises(RuntimeError):
                trainer.close()
            close_loader.assert_called_once_with(trainer.train_loader)
        trainer.logger.metrics_file.close.assert_called_once()
        trainer.logger.wandb_run.finish.assert_called_once()

    def test_atomic_publication_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'old')
            with self.assertRaisesRegex(ValueError, 'writer'):
                with atomic_path(path) as temporary:
                    temporary.write_bytes(b'partial')
                    raise ValueError('writer')
            self.assertEqual(path.read_bytes(), b'old')
            self.assertFalse(temporary.exists())
            with patch('dexmani_policy.common.io_util.os.replace', side_effect=OSError('publish')):
                with self.assertRaisesRegex(OSError, 'publish'):
                    with atomic_path(path) as temporary:
                        temporary.write_bytes(b'new')
            self.assertFalse(temporary.exists())
            self.assertEqual(path.read_bytes(), b'old')
            with atomic_path(path) as temporary:
                temporary.write_bytes(b'new')
            self.assertEqual(path.read_bytes(), b'new')

    def test_policy_info_requires_control_period(self):
        from dexmani_policy.deployment.metadata import policy_info
        cfg = dict(task_name='test', action_key='action', dataset={},
                   control_dt_s=.05)
        agent = SimpleNamespace(normalization_dims={'joint_state': 19}, n_obs_steps=1,
                                n_action_steps=1, future_steps=1, control_action_dim=19,
                                supports_rtc=False, requires_prev_action=False)
        info = policy_info('test.pt', cfg, agent, weights='raw', inference_steps=1)
        self.assertEqual(info.control_dt_s, .05)
        del cfg['control_dt_s']
        with self.assertRaisesRegex(KeyError, 'control_dt_s'):
            policy_info('test.pt', cfg, agent, weights='raw', inference_steps=1)

    def test_default_randomization(self):
        cfg = load_policy_config('dp3')
        self.assertFalse(cfg.env_runner.env_kwargs.randomize_model_id)
        self.assertTrue(cfg.env_runner.env_kwargs.randomize_table_height)
        explicit = load_policy_config('dp3', ['env_runner.env_kwargs.randomize_model_id=true'])
        self.assertTrue(explicit.env_runner.env_kwargs.randomize_model_id)

    def test_simulation_control_period(self):
        runner = SimRunner('test', 1, ['joint_state'], {}, control_dt_s=.05,
                           env_rebuild_interval=1)
        instances = []
        periods = iter([.05, .05 + 1e-8, .1])
        class Env:
            def __init__(self, **kwargs):
                self.dt = next(periods)
                self.action_cnt = 1
                self.close = Mock()
                instances.append(self)
        with patch.dict('sys.modules', {'dexmani_sim': SimpleNamespace(resolve_env_class=lambda _: Env)}), \
                patch.object(runner, '_run_episode', return_value=(True, 1)) as episode:
            results = runner.run(Mock(requires_prev_action=False), seeds=[1, 2, 3],
                                 inference_steps=1, action_steps=1)
            self.assertTrue(next(results)['success'])
            self.assertTrue(next(results)['success'])
            with self.assertRaisesRegex(ValueError, r'Sim dt=0.1.*Policy control_dt_s=0.05'):
                next(results)
            self.assertEqual(episode.call_count, 2)
        self.assertEqual(len(instances), 3)
        for env in instances:
            env.close.assert_called_once()

    def test_simulation_rebuild_releases_previous_environment(self):
        for failure in (None, 'close', 'make'):
            with self.subTest(failure=failure):
                runner = SimRunner('test', 1, ['joint_state'], {}, control_dt_s=.05,
                                   env_rebuild_interval=1)
                references, made, closed = [], [], []
                error = RuntimeError(failure)

                class Scene:
                    pass

                class Env:
                    def __init__(self, index):
                        self.index = index
                        self.action_cnt = 1
                        self.task = SimpleNamespace(scene=Scene())

                    def close(self):
                        closed.append(self.index)
                        if failure == 'close':
                            raise error

                def make_env(*, record_video):
                    made.append(record_video)
                    if references:
                        for reference in references[-1]:
                            self.assertIsNone(reference())
                        if failure == 'make':
                            raise error
                    env = Env(len(references))
                    references.append((weakref.ref(env), weakref.ref(env.task.scene)))
                    return env

                # Mock call histories would themselves keep the old env alive.
                runner.make_env = make_env
                runner._run_episode = lambda *args, **kwargs: (True, 1)
                results = runner.run(SimpleNamespace(requires_prev_action=False), seeds=[0, 1],
                                     inference_steps=1, action_steps=1)
                self.assertTrue(next(results)['success'])
                if failure is None:
                    self.assertTrue(next(results)['success'])
                    with self.assertRaises(StopIteration):
                        next(results)
                    self.assertEqual(closed, [0, 1])
                    self.assertTrue(all(ref() is None for pair in references for ref in pair))
                else:
                    with self.assertRaises(RuntimeError) as caught:
                        next(results)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(closed, [0])
                self.assertEqual(made, [False] if failure == 'close' else [False, False])

    def test_simulation_shutdown(self):
        runner = SimRunner('test', 1, ['joint_state'], {}, control_dt_s=.05)
        for early, fail in ((False, False), (True, False), (False, True), (True, True)):
            env = Mock()
            env.close.side_effect = RuntimeError('close') if fail else None
            with self.subTest(early=early, fail=fail), patch.object(runner, 'make_env', return_value=env), \
                    patch.object(runner, '_run_episode', return_value=(True, 1)):
                results = runner.run(Mock(requires_prev_action=False), seeds=[0],
                                     inference_steps=1, action_steps=1)
                with self.assertRaisesRegex(RuntimeError, 'close') if fail else nullcontext():
                    if early:
                        next(results)
                        results.close()
                    else:
                        self.assertEqual(len(list(results)), 1)
                env.close.assert_called_once()

    def test_simulation_error_survives_close_failure(self):
        runner = SimRunner('test', 1, ['joint_state'], {}, control_dt_s=.05)
        env = Mock()
        primary, cleanup = ValueError('simulation'), RuntimeError('close')
        env.close.side_effect = cleanup
        with patch.object(runner, 'make_env', return_value=env), patch.object(
            runner, '_run_episode', side_effect=primary,
        ):
            with self.assertLogs(level='ERROR') as logs, self.assertRaisesRegex(ValueError, 'simulation'):
                list(runner.run(Mock(requires_prev_action=False), seeds=[0],
                                inference_steps=1, action_steps=1))
        self.assertIn('RuntimeError: close', '\n'.join(logs.output))
        env.close.assert_called_once()

        class BrokenEnv:
            close = Mock(side_effect=cleanup)

            def __init__(self, **kwargs):
                raise primary

        with patch.dict('sys.modules', {'dexmani_sim': SimpleNamespace(resolve_env_class=lambda _: BrokenEnv)}):
            with self.assertLogs(level='ERROR') as logs, self.assertRaisesRegex(ValueError, 'simulation'):
                runner.make_env(record_video=False)
        self.assertIn('RuntimeError: close', '\n'.join(logs.output))
        BrokenEnv.close.assert_called_once()

    def test_contact_temporary_directory_cleanup(self):
        import zarr
        from dexmani_policy.datasets.preprocess.contact_map import generate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = zarr.open_group(str(root / 'source.zarr'), mode='w')
            source.attrs.update(task_name='test', dt=.05, action_space='joint',
                                format='obsolete', action_semantics='obsolete')
            source.create_dataset('meta/episode_ends', data=np.array([1]))
            for key, shape in (('joint_state', (1, 19)), ('hand_pointcloud', (1, 2, 3)),
                               ('object_pointcloud', (1, 1024, 3))):
                source.create_dataset('data/' + key, data=np.ones(shape, np.float32))
            with patch('dexmani_policy.datasets.preprocess.contact_map.frame_contact_map',
                       side_effect=ValueError('contact')):
                with self.assertRaisesRegex(ValueError, 'contact'):
                    generate(root / 'source.zarr', root / 'output.zarr')
            self.assertFalse((root / 'output.zarr').exists())
            self.assertFalse(list(root.glob('.contact-*')))
            with patch('dexmani_policy.datasets.preprocess.contact_map.frame_contact_map',
                       return_value=np.ones(1024, np.float32)):
                generate(root / 'source.zarr', root / 'output.zarr')
            output = zarr.open_group(str(root / 'output.zarr'))
            self.assertEqual(output['data/contact_map'].shape, (1, 1024))
            self.assertEqual(dict(output.attrs), dict(task_name='test', dt=.05, action_space='joint'))
            self.assertFalse(list(root.glob('.contact-*')))

    def test_resume_logger_preserves_history_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = b'{"step": 999}\ninvalid\n{"step": 1}\n{"partial":'
            (root / 'metrics.jsonl').write_bytes(history)
            (root / 'config.yaml').write_text('original: true\n')
            with closing(ExperimentLogger(directory, resuming=True, wandb_enabled=False)) as logger:
                logger.save_config(OmegaConf.create({'original': False}))
                self.assertEqual((root / 'metrics.jsonl').read_bytes(), history)
                logger.log({'loss': 2}, 1)
            self.assertEqual((root / 'metrics.jsonl').read_bytes(), history + b'{"step": 1, "loss": 2}\n')
            self.assertEqual((root / 'config.yaml').read_text(), 'original: true\n')
            with closing(ExperimentLogger(directory, resuming=True, wandb_enabled=False)) as logger:
                with patch.object(logger.metrics_file, 'write', side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):
                        logger.log({'loss': 1}, 2)

    def test_selection_environment_contract(self):
        from dexmani_policy.evaluation.seeds import split_evaluation_seeds
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = [root / 'a.pt', root / 'b.pt']
            for checkpoint in checkpoints:
                checkpoint.touch()
            configs = [SimpleNamespace(effective_config={'env_runner': {
                'task_name': 'test', 'env_kwargs': {'friction': .1}},
                'control_dt_s': .05 + value * 1e-8,
                'agent': {'architecture': str(value)},
                'normalization_dims': {'joint_state': 19} if value == 0 else {'point_cloud': 6}})
                for value in (0, 1)]
            calls = []
            def evaluate(checkpoint, **kwargs):
                calls.append((Path(checkpoint).resolve(), kwargs))
                return dict(statistics=dict(success_rate=1, avg_steps=2),
                            global_step=1 if Path(checkpoint).resolve() == checkpoints[0] else 2,
                            weights='raw', inference_steps=3, action_steps=2), kwargs['output_dir'] / 'result.json'
            with patch('dexmani_policy.evaluation.selection.load_checkpoint', side_effect=configs), \
                    patch('dexmani_policy.evaluation.selection.load_seed_pool', return_value=list(range(10))), \
                    patch('dexmani_policy.evaluation.selection.evaluate_policy', side_effect=evaluate):
                result = run_pipeline(checkpoints, selection_episodes=2, test_episodes=3,
                                      seed=42, output_dir=root / 'eval')
            test, selection = split_evaluation_seeds(list(range(10)), 42, 3, 2)
            self.assertFalse(set(test) & set(selection))
            self.assertEqual([item[1]['seeds'] for item in calls], [selection, selection, test])
            self.assertTrue(all(not item[1]['video'] for item in calls))
            self.assertEqual(calls[2][1]['weights'], 'raw')
            self.assertEqual(calls[2][1]['action_steps'], 2)
            self.assertEqual([item[0] for item in calls], [*checkpoints, checkpoints[1]])
            self.assertIsNone(calls[0][1]['weights'])
            self.assertIsNone(calls[1][1]['inference_steps'])
            self.assertEqual(calls[2][1]['inference_steps'], 3)
            self.assertEqual(json.loads(result.read_text())['global_step'], 2)
            for field, value in (('task_name', 'other'), ('env_kwargs', {'friction': .9}),
                                 ('control_dt_s', .1)):
                config = configs[1].effective_config
                section = config if field == 'control_dt_s' else config['env_runner']
                original = section[field]
                section[field] = value
                with self.subTest(field=field), patch(
                        'dexmani_policy.evaluation.selection.load_checkpoint', side_effect=configs), patch(
                        'dexmani_policy.evaluation.selection.evaluate_policy') as evaluate:
                    with self.assertRaisesRegex(ValueError, field):
                        run_pipeline(checkpoints)
                    evaluate.assert_not_called()
                section[field] = original

    def test_logger_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            with closing(ExperimentLogger(directory, wandb_enabled=False)) as logger:
                with self.assertRaises(ValueError):
                    logger.log({'loss': float('nan')}, 1)
            self.assertEqual((Path(directory) / 'metrics.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
