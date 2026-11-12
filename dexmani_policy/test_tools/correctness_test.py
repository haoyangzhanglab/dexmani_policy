"""Cheap CPU regressions for resume, optimizer and publication boundaries."""
import json
import tempfile
import unittest
from contextlib import closing, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

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
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        trainer.trainable_parameters = [parameter]
        trainer.global_step = 0
        trainer.max_grad_norm = 0
        trainer.optimizer = torch.optim.Adam([parameter], lr=0.1)
        trainer.scheduler = Mock()
        trainer.ema_updater = Mock()
        parameter.grad = torch.tensor([3.0])
        with patch.object(trainer.optimizer, 'step') as step:
            trainer.optimizer_step()
            step.assert_called_once()
            self.assertEqual(trainer.last_grad_norm.item(), 3)
        for value in (float('nan'), float('inf')):
            parameter.grad = torch.tensor([value])
            before = parameter.detach().clone()
            trainer.scheduler.reset_mock()
            trainer.ema_updater.reset_mock()
            with self.assertRaises(RuntimeError):
                trainer.optimizer_step()
            torch.testing.assert_close(parameter, before)
            self.assertFalse(trainer.optimizer.state)
            trainer.scheduler.step.assert_not_called()
            trainer.ema_updater.step.assert_not_called()
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
        self.assertIsNone(info.pointcloud_config)
        self.assertIsNone(info.fingertip_link_names)
        self.assertIsNone(info.depth_config)
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
