"""Cheap CPU regressions for resume, optimizer and publication boundaries."""
import copy
import json
import shutil
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.codebook import validate_hand_coordinates, load_codebook
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.deployment.runtime import PolicyRuntime
from dexmani_policy.evaluation.selection import run_pipeline
from dexmani_policy.test_tools.smoke_test import make_source
from dexmani_policy.training.common.config import load_policy_config, validate_config
from dexmani_policy.training.common.data import build_dataset, prepare_data_config, EpochBatchSampler, epoch_position
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.pretrain.obs.common import export_encoder, save_run
from dexmani_policy.training.trainer import PolicyTrainer


class CorrectnessTest(unittest.TestCase):
    def test_resume_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root / 'source.zarr')
            source.attrs['dt'] = 0.05
            cfg = load_policy_config('dp3', [f'zarr_path={root / "source.zarr"}',
                                            'horizon=4', 'n_obs_steps=2', 'n_action_steps=2',
                                            'dataset.augmentation_cfg=null'])
            cfg.workspace.output_dir = str(root / "run")
            dataset, description = build_dataset(cfg)
            metadata = description['deployment_metadata']
            prepare_data_config(cfg, dataset, metadata)
            saved = OmegaConf.to_container(cfg, resolve=True)
            prepare_data_config(cfg, dataset, metadata, saved=saved)
            source['meta/episode_ends'][:] = [4, 12]
            changed, _ = build_dataset(cfg)
            self.assertEqual(len(changed), len(dataset))
            with self.assertRaisesRegex(ValueError, 'layout'):
                prepare_data_config(cfg, changed, metadata, saved=saved, confirm_data_relocation=True)
            source['meta/episode_ends'][:] = [6, 12]
            for changes in ({'episode_ends': [4, 12]}, {'train_windows': 9}):
                old = copy.deepcopy(saved)
                old['data_info'].update(changes)
                with self.assertRaisesRegex(ValueError, 'layout'):
                    prepare_data_config(cfg, dataset, metadata, saved=old)
            with self.assertRaisesRegex(ValueError, 'deployment_metadata'):
                prepare_data_config(cfg, dataset, {'control_dt_s': 0.1}, saved=saved)
            self.assertEqual(cfg.deployment_metadata.control_dt_s, 0.05)
            shutil.copytree(root / 'source.zarr', root / 'copy.zarr')
            cfg.dataset.zarr_path = str(root / 'copy.zarr')
            moved, _ = build_dataset(cfg)
            with self.assertRaisesRegex(ValueError, 'path changed'):
                prepare_data_config(cfg, moved, metadata, saved=saved)
            prepare_data_config(cfg, moved, metadata, saved=saved, confirm_data_relocation=True)
            for key in ('data_info', 'deployment_metadata'):
                incomplete = copy.deepcopy(saved)
                del incomplete[key]
                with self.assertRaises(KeyError):
                    prepare_data_config(cfg, dataset, metadata, saved=incomplete)

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
                            normalizer_state_dict=normalizer.state_dict(), epoch=3), source)
            export_codebook(source, output)
            codebook, affine = load_codebook(output)
            self.assertEqual(codebook.num_codes, 4)
            torch.testing.assert_close(affine['scale'], torch.ones(12))
            with np.load(output, allow_pickle=False) as artifact:
                self.assertEqual(artifact['source_epoch'].item(), 3)
                self.assertEqual(artifact['source_checkpoint'].item(), str(source))

    def test_contact_metadata(self):
        import zarr
        from dexmani_policy.datasets.preprocess.contact_map import generate
        with tempfile.TemporaryDirectory() as directory:
            source_path, output = Path(directory) / 'source.zarr', Path(directory) / 'contact.zarr'
            source = make_source(source_path)
            source.attrs.update(format='dexmani.real.canonical',
                                action_semantics='absolute_joint_goal_not_next_step_arrival')
            for key, shape in (('hand_pointcloud', (12, 8, 3)), ('object_pointcloud', (12, 1024, 3))):
                source.create_dataset('data/' + key, data=np.zeros(shape, dtype=np.float32))
            with patch('dexmani_policy.datasets.preprocess.contact_map.frame_contact_map',
                       return_value=np.zeros(1024, dtype=np.float32)):
                generate(source_path, output)
            result = zarr.open_group(str(output), mode='r')
            self.assertEqual(dict(result.attrs), dict(source.attrs))
            validate_hand_coordinates(dict(result.attrs))

    def test_coordinate_and_json_boundaries(self):
        validate_hand_coordinates({'format': 'dexmani.real.canonical'})
        validate_hand_coordinates({'hand_joint_order': ['thumb:3', 'index:3', 'middle:2', 'ring:2', 'pinky:2'],
                                   'hand_units': 'radians'})
        for attrs in ({}, {'hand_units': 'degrees'}, {'hand_joint_order': []}, {'format': 'unknown'},
                      {'format': 'dexmani.real.canonical', 'hand_units': 'degrees'}):
            with self.assertRaises(ValueError):
                validate_hand_coordinates(attrs)
        with tempfile.TemporaryDirectory() as directory:
            with closing(ExperimentLogger(directory, wandb_enabled=False)) as logger:
                with self.assertRaises(ValueError):
                    logger.log({'loss': float('nan')}, 1)
            self.assertEqual((Path(directory) / 'metrics.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
