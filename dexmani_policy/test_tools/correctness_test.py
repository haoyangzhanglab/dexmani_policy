"""Cheap CPU regressions for resume, optimizer and publication boundaries."""
import json
import tempfile
import unittest
from contextlib import closing
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
from dexmani_policy.training.common.data import EpochBatchSampler, epoch_position
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.pretrain.obs.common import export_encoder, save_run
from dexmani_policy.training.trainer import PolicyTrainer


class CorrectnessTest(unittest.TestCase):
    def test_point_counts_and_contact_head_boundary(self):
        import zarr
        from dexmani_policy.training.common.data import build_dataset
        from dexmani_policy.datasets.preprocess.contact_map import generate
        from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.zarr'
            source = zarr.open_group(str(path), mode='w')
            source.create_dataset('meta/episode_ends', data=np.array([4]))
            for key, shape in (('joint_state', (4, 19)), ('action', (4, 19)),
                               ('hand_pointcloud', (4, 7, 6)), ('object_pointcloud', (4, 11, 6)),
                               ('contact_map', (4, 11))):
                source.create_dataset('data/' + key, data=np.ones(shape, np.float16))
            cfg = load_policy_config('cordvip', [f'zarr_path={path}', 'horizon=4', 'n_obs_steps=2', 'n_action_steps=2'])
            dataset, description = build_dataset(cfg)
            self.assertEqual(set(description), {'field_dims', 'deployment_metadata'})
            self.assertEqual(description['deployment_metadata'], {})
            self.assertEqual(description['field_dims']['object_pointcloud'], 3)
            self.assertEqual(dataset[0]['obs']['object_pointcloud'].shape, (2, 11, 3))
            self.assertEqual(dataset[0]['obs']['object_pointcloud'].dtype, torch.float32)
            with self.assertRaisesRegex(ValueError, '1024'):
                generate(path, Path(directory) / 'contact.zarr')
            with self.assertRaisesRegex(ValueError, '1024'):
                CordViPPretrainDataset(path)

    def test_sampling_position(self):
        sampler = EpochBatchSampler(range(10), 2, False, shuffle=True, seed=42)
        for step in range(7):
            epoch, start = epoch_position(step, sampler.batches_per_epoch, 2)
            sampler.set_epoch(epoch)
            full = list(sampler)
            sampler.set_epoch(epoch, start)
            self.assertEqual(list(sampler), full[start:])
        self.assertEqual(epoch_position(3, 5, 2), (1, 0))

    def test_optimizer_boundary(self):
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        trainer.trainable_parameters = [parameter]
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
        gradient = torch.tensor([3.0, 4.0])
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_([parameter], float('inf'), error_if_nonfinite=True)
        self.assertTrue(torch.equal(parameter.grad, gradient))
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
                    effective_config={'env_runner': {'task_name': 'test'}})), patch(
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

    def test_cleanup_attempts_all_resources_and_preserves_training_error(self):
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        trainer.train_loader = Mock()
        trainer.logger = ExperimentLogger.__new__(ExperimentLogger)
        errors = [RuntimeError(name) for name in ('workers', 'metrics', 'wandb')]
        trainer.logger.metrics_file = Mock()
        trainer.logger.metrics_file.close.side_effect = errors[1]
        trainer.logger.wandb_run = Mock()
        trainer.logger.wandb_run.finish.side_effect = errors[2]
        for primary in (None, ValueError('training failed')):
            with self.subTest(primary=primary), patch(
                'dexmani_policy.training.trainer.close_data_loader', side_effect=errors[0],
            ) as close_loader:
                try:
                    with closing(trainer):
                        if primary is not None:
                            raise primary
                except RuntimeError as error:
                    self.assertIs(error, errors[2])
                    if primary is not None:
                        self.assertIs(error.__context__, errors[1])
                        self.assertIs(errors[1].__context__, errors[0])
                        self.assertIs(errors[0].__context__, primary)
                else:
                    self.fail('Cleanup errors must propagate')
                close_loader.assert_called_once_with(trainer.train_loader)
            trainer.logger.metrics_file.close.assert_called_once()
            trainer.logger.wandb_run.finish.assert_called_once()
            trainer.logger.metrics_file.close.reset_mock()
            trainer.logger.wandb_run.finish.reset_mock()

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

    def test_simulation_error_survives_close_failure(self):
        runner = SimRunner('test', 1, ['joint_state'], {})
        env = Mock()
        primary, cleanup = ValueError('simulation'), RuntimeError('close')
        env.close.side_effect = cleanup
        with patch.object(runner, 'make_env', return_value=env), patch.object(
            runner, '_run_episode', side_effect=primary,
        ), patch('dexmani_policy.env_runner.sim_runner.asdict', return_value={}):
            try:
                list(runner.run(Mock(requires_prev_action=False), seeds=[0],
                                inference_steps=1, action_steps=1))
            except RuntimeError as error:
                self.assertIs(error, cleanup)
                self.assertIs(error.__context__, primary)
            else:
                self.fail('Simulation must fail')
        env.close.assert_called_once()

        class BrokenEnv:
            close = Mock(side_effect=cleanup)

            def __init__(self, **kwargs):
                raise primary

        with patch.dict('sys.modules', {'dexmani_sim': SimpleNamespace(resolve_env_class=lambda _: BrokenEnv)}):
            with self.assertRaises(RuntimeError) as caught:
                runner.make_env(record_video=False)
        self.assertIs(caught.exception.__context__, primary)
        BrokenEnv.close.assert_called_once()

    def test_contact_temporary_directory_cleanup(self):
        import zarr
        from dexmani_policy.datasets.preprocess.contact_map import generate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = zarr.open_group(str(root / 'source.zarr'), mode='w')
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
            self.assertEqual(zarr.open_group(str(root / 'output.zarr'))['data/contact_map'].shape, (1, 1024))
            self.assertFalse(list(root.glob('.contact-*')))

    def test_logger_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            with closing(ExperimentLogger(directory, wandb_enabled=False)) as logger:
                with self.assertRaises(ValueError):
                    logger.log({'loss': float('nan')}, 1)
            self.assertEqual((Path(directory) / 'metrics.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
