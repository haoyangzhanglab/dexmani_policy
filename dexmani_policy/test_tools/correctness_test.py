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
            codebook, affine = load_codebook(output)
            self.assertEqual(codebook.num_codes, 4)
            torch.testing.assert_close(affine['scale'], torch.ones(12))

    def test_logger_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            with closing(ExperimentLogger(directory, wandb_enabled=False)) as logger:
                with self.assertRaises(ValueError):
                    logger.log({'loss': float('nan')}, 1)
            self.assertEqual((Path(directory) / 'metrics.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
