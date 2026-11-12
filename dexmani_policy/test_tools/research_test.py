"""Small real-model regressions for research training and initialization."""
import inspect
import os
import signal
import sys
import tempfile
import unittest
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.deployment import load_policy
from dexmani_policy.test_tools.smoke_test import make_source
from dexmani_policy.train import build_trainer
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.resume import prepare_resume
from dexmani_policy.training.common.optimization import EMA

DEVICE = os.environ.get('DEXMANI_TEST_DEVICE', 'cpu')


@contextmanager
def stop_before_break_check(function, boundary):
    lines, first = inspect.getsourcelines(function)
    target = [first + i - 1 for i, line in enumerate(lines) if line.strip() == 'break'][boundary]
    previous = sys.gettrace()
    fired = False

    def trace(frame, event, arg):
        nonlocal fired
        if frame.f_code is not function.__code__:
            return None
        if event == 'line' and frame.f_lineno == target and not fired:
            fired = True
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            # CPython trace synchronization must retain the handler's closure update.
            frame.f_locals
        return trace

    try:
        sys.settrace(trace)
        yield
    finally:
        sys.settrace(previous)
    assert fired, 'Stop injection did not reach the requested loop boundary'


class ResearchTest(unittest.TestCase):
    def config(self, root):
        cfg = load_policy_config('dp3', [
            f'zarr_path={root / "source.zarr"}', f'workspace.output_dir={root / "run"}',
            f'training.device={DEVICE}', 'workspace.wandb_cfg=null',
            'horizon=4', 'n_obs_steps=2', 'n_action_steps=2', 'agent.num_points=8',
            'agent.down_dims=[16,32]', 'agent.diffusion_step_embed_dim=16',
            'training.use_compile=false', 'training.lr_warmup_steps=0',
            'training.loop.total_train_steps=2', 'dataset.augmentation_cfg=null',
            'dataloader.batch_size=3', 'dataloader.drop_last=false',
            'dataloader.num_workers=0', 'training.loop.gradient_accumulation_steps=2',
        ])
        with open_dict(cfg.normalization):
            cfg.normalization.unused = 'not-a-mode'
        return cfg

    def test_training_extension_and_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'source.zarr')
            cfg = self.config(root)
            with closing(build_trainer(cfg)) as trainer:
                with patch.object(trainer, 'save_checkpoint', wraps=trainer.save_checkpoint) as save:
                    trainer.train()
                    self.assertEqual(save.call_count, 2)
                    trainer.train()
                    self.assertEqual(save.call_count, 2)
                self.assertEqual(trainer.global_step, 2)
            source = root / 'run/checkpoints/latest.pt'
            checkpoint = load_checkpoint(source)
            self.assertNotIn('rng_state', vars(checkpoint))
            with closing(prepare_resume(root / 'run', total_steps=4, log_interval=2)) as trainer:
                self.assertEqual(trainer.global_step, 2)
                self.assertEqual(trainer.log_interval_steps, 2)
                self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], cfg.optimizer.lr * .5)
                torch.testing.assert_close(trainer.optimizer.state_dict()['state'], checkpoint.optimizer_state['state'], check_device=False)
                with patch.object(trainer, 'save_checkpoint', wraps=trainer.save_checkpoint) as save:
                    trainer.train(max_updates=1)
                    save.assert_called_once()
                self.assertEqual(trainer.global_step, 3)
            checkpoint = load_checkpoint(source)
            cfg.workspace.output_dir = str(root / 'initialized')
            cfg.initialization.policy = str(source)
            cfg.optimizer.lr = 0.002
            cfg.optimizer.obs_lr = 0
            with patch('dexmani_policy.train.fit_normalizer', side_effect=AssertionError('must not fit')):
                with closing(build_trainer(cfg)) as trainer:
                    self.assertEqual(trainer.global_step, 0)
                    self.assertFalse(trainer.optimizer.state)
                    self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], .002)
                    self.assertFalse(any(p.requires_grad for p in trainer.model.obs_encoder.parameters()))
                    torch.testing.assert_close(trainer.model.state_dict(), checkpoint.model_state, check_device=False)
                    torch.testing.assert_close(trainer.ema_model.state_dict(), checkpoint.model_state, check_device=False)
                    trainer.train(max_updates=1)
            with closing(prepare_resume(root / 'initialized')) as trainer:
                self.assertEqual(trainer.global_step, 1)
                trainer.train(max_updates=1)

    def test_policy_stop_boundaries(self):
        from dexmani_policy.training.trainer import PolicyTrainer
        for boundary in (0, 1):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_source(root / 'source.zarr')
                cfg = self.config(root)
                cfg.training.loop.total_train_steps = 20
                cfg.training.loop.gradient_accumulation_steps = 1
                cfg.dataloader.batch_size = 10
                with closing(build_trainer(cfg)) as trainer:
                    with patch.object(trainer, 'save_checkpoint', wraps=trainer.save_checkpoint) as save, \
                            stop_before_break_check(PolicyTrainer.train, boundary):
                        trainer.train(max_updates=3)
                    self.assertEqual(trainer.global_step, 2)
                    save.assert_called_once()
                    checkpoint = load_checkpoint(root / 'run/checkpoints/latest.pt')
                    self.assertEqual(checkpoint.global_step, 2)
                    torch.testing.assert_close(checkpoint.model_state, trainer.model.state_dict(),
                                               check_device=False, rtol=0, atol=0)
                    torch.testing.assert_close(checkpoint.ema_model_state, trainer.ema_model.state_dict(),
                                               check_device=False, rtol=0, atol=0)

    def test_pretrain_stop_and_save_boundaries(self):
        from dexmani_policy.training.pretrain.obs import common
        from dexmani_policy.training.pretrain.obs.afro import AFRO
        for mode in ('inner', 'outer', 'validation', 'total', 'max_updates'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_source(root / 'source.zarr')
                bounded = mode in ('total', 'max_updates')
                args = ['--output-dir', str(root / 'run'), '--zarr-path', str(root / 'source.zarr'),
                        '--device', DEVICE, '--num-workers', '0', '--no-wandb']
                if mode != 'total':
                    args += ['--max-updates', '1' if mode == 'max_updates' else '3']
                args += ['dataset.horizon=3', 'dataset.frame_stride=1', 'dataset.pad_after=0',
                         'dataloader.batch_size=8', f'training.num_epochs={2 if bounded else 20}',
                         f'training.save_every_epochs={1 if bounded else 20}']
                if mode == 'validation':
                    args += ['dataset.val_episodes=1', 'training.val_every_epochs=1']
                evaluate = common.evaluate

                def stop_after_validation(*args, **kwargs):
                    metrics = evaluate(*args, **kwargs)
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    return metrics

                if mode in ('inner', 'outer'):
                    injection = stop_before_break_check(common.train, ('inner', 'outer').index(mode))
                elif mode == 'validation':
                    injection = patch.object(common, 'evaluate', side_effect=stop_after_validation)
                else:
                    injection = nullcontext()
                with patch.object(common, 'save_run', wraps=common.save_run) as save, injection:
                    common.train('afro', AFRO, args)
                expected = 1 if mode in ('validation', 'max_updates') else 2
                self.assertEqual([call.args[-1] for call in save.call_args_list],
                                 [1, 2] if mode == 'total' else [expected])
                checkpoint = torch.load(root / 'run/checkpoints/latest.pt', weights_only=False)
                encoder = torch.load(root / 'run/encoder.pt', weights_only=True)
                self.assertEqual(checkpoint['global_step'], expected)
                self.assertEqual(encoder['global_step'], expected)
                torch.testing.assert_close(checkpoint['model_state'], save.call_args.args[0].state_dict(), rtol=0, atol=0)
                for name in ('encoder', 'normalizer'):
                    state = {key.removeprefix(name + '.'): value for key, value in checkpoint['model_state'].items()
                             if key.startswith(name + '.')}
                    torch.testing.assert_close(encoder[name + '_state'], state, check_device=False, rtol=0, atol=0)
                self.assertEqual(encoder['normalization'], checkpoint['effective_config']['normalization'])
                if mode == 'total':
                    with patch.object(common, 'save_run') as save, patch.object(common, 'export_encoder') as export:
                        common.train('afro', AFRO, ['--resume-from', str(root / 'run'), '--no-wandb'])
                    save.assert_not_called()
                    export.assert_not_called()

    def test_initialization_without_original_sources(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root / 'source.zarr')
            source.create_dataset('data/rgb', data=np.zeros((12,32,32,3), np.uint8))
            source.create_dataset('data/camera_intrinsic', data=np.tile(np.eye(3).reshape(1,9), (12,1)))
            source.create_dataset('data/camera_extrinsic', data=np.tile(np.eye(3,4).reshape(1,12), (12,1)))
            np.savez(root / 'codebook.npz', sorted_hand_poses=np.zeros((4,12), np.float32),
                     hand_normalizer_scale=np.ones(12, np.float32), hand_normalizer_offset=np.zeros(12, np.float32))
            cfg = load_policy_config('dqrise', [f'zarr_path={root / "source.zarr"}',
                f'workspace.output_dir={root / "run"}', f'training.device={DEVICE}',
                f'initialization.codebook={root / "codebook.npz"}', 'initialization.rgb=null',
                'workspace.wandb_cfg=null', 'horizon=4', 'n_action_steps=2', 'agent.num_points=8',
                'agent.hidden_dim=24', 'agent.n_layers=1', 'agent.n_head=4',
                'training.use_compile=false', 'dataloader.num_workers=0', 'dataloader.batch_size=2',
                'training.loop.total_train_steps=2', 'training.lr_warmup_steps=0'])
            with open_dict(cfg.agent):
                cfg.agent.pc_encoder_config = dict(channels=24, num_patches=2, group_size=2, depth=1, num_heads=4)
                cfg.agent.rgb_backbone_name = 'dino'
                cfg.agent.rgb_backbone_config = dict(image_size=[32,32], tune_mode='full', architecture=dict(
                    model_type='dinov2', hidden_size=24, num_hidden_layers=1,
                    num_attention_heads=4, image_size=32, patch_size=16))
            with closing(build_trainer(cfg)) as trainer:
                trainer.train(max_updates=1)
            checkpoint = load_checkpoint(root / 'run/checkpoints/latest.pt')
            (root / 'codebook.npz').unlink()
            self.assertEqual(checkpoint.effective_config['agent']['num_codes'], 4)
            observation = {key: source[f'data/{key}'][:2] for key in cfg.dataset.sensor_modalities}
            with closing(load_policy(root / 'run', weights='raw', inference_steps=1, device=DEVICE)) as runtime:
                self.assertEqual(runtime.agent.hand_codebook.num_codes, 4)
                prediction = runtime.predict(observation)
                self.assertEqual(prediction.shape, (3, 19))
                self.assertTrue(np.isfinite(prediction).all())
            del cfg.agent.rgb_backbone_config.architecture
            cfg.initialization.rgb = '/missing/pretrained'
            cfg.initialization.policy = str(root / 'run/checkpoints/latest.pt')
            cfg.workspace.output_dir = str(root / 'initialized')
            cfg.agent.rgb_backbone_config.tune_mode = 'freeze'
            with closing(build_trainer(cfg)) as trainer:
                self.assertEqual(trainer.model.hand_codebook.num_codes, 4)
                torch.testing.assert_close(trainer.model.state_dict(), checkpoint.model_state, check_device=False)
                self.assertFalse(trainer.optimizer.state)
                trainer.train(max_updates=1)
            cfg.agent.rgb_backbone_config = None
            cfg.workspace.output_dir = str(root / 'null_rgb_config')
            with closing(build_trainer(cfg)) as trainer:
                self.assertEqual(trainer.model.obs_encoder.rgb_encoder.hidden_dim, 24)
                torch.testing.assert_close(trainer.model.state_dict(), checkpoint.model_state, check_device=False)

    def test_initialization_preserves_tactile_regions(self):
        import numpy as np
        from dexmani_policy.agents.obs_encoder.xhand_tactile import right_hand_region_ids
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_source(root / 'source.zarr')
            data.create_dataset('data/taxel_tactile_force', data=np.zeros((12,5,120,3), np.float32))
            cfg = load_policy_config('adaptac', [
                'tactile=patch', f'zarr_path={root / "source.zarr"}',
                f'workspace.output_dir={root / "run"}', f'training.device={DEVICE}',
                'workspace.wandb_cfg=null', 'horizon=4', 'n_action_steps=2',
                'agent.num_points=8', 'agent.pc_encoder_config.num_patches=2',
                'agent.pc_encoder_config.group_size=2', 'agent.pc_encoder_config.depth=1',
                'agent.feature_dim=32', 'agent.down_dims=[16,32]',
                'agent.force_predictor_config.depth=1', 'agent.force_predictor_config.embed_dim=32',
                'dataloader.num_workers=0', 'dataloader.batch_size=2',
            ])
            regions = (right_hand_region_ids() + 1) % 5
            with open_dict(cfg.agent.tactile_encoder):
                cfg.agent.tactile_encoder.region_ids = regions.tolist()
            with closing(build_trainer(cfg)) as trainer:
                trainer.save_checkpoint()
            del cfg.agent.tactile_encoder.region_ids
            cfg.initialization.policy = str(root / 'run/checkpoints/latest.pt')
            cfg.workspace.output_dir = str(root / 'initialized')
            cfg.agent.force_loss_weight = .5
            with closing(build_trainer(cfg)) as trainer:
                encoder = trainer.model.obs_encoder.tactile_encoder
                torch.testing.assert_close(encoder.region_ids.cpu(), regions)
                self.assertEqual(trainer.model.force_loss_weight, .5)
                self.assertFalse(trainer.optimizer.state)
                self.assertEqual(trainer.global_step, 0)

    def test_normalization_active_fields(self):
        self.assertEqual(normalization_feature_shapes({'action': 19}, {'action': 'limits', 'rgb': 'bad'}), {'action': (19,)})
        for spec in ({'actoin': 'limits'}, {'action': 'identity'}, {'action': 'bad'}):
            with self.assertRaises(ValueError):
                normalization_feature_shapes({'action': 19}, spec)

    def test_normalizer_feature_axes(self):
        from dexmani_policy.common.normalizer import LinearNormalizer
        values = torch.arange(90, dtype=torch.float32).reshape(6,5,3)
        normalizer = LinearNormalizer()
        normalizer.fit_field('fingertip_xyz', (values[:2], values[2:]))
        field = normalizer['fingertip_xyz']
        self.assertEqual(field.scale.shape, (5,3))
        scaled = field.normalize(values)
        torch.testing.assert_close(scaled.amin(0), -torch.ones(5,3))
        torch.testing.assert_close(scaled.amax(0), torch.ones(5,3))
        torch.testing.assert_close(field.unnormalize(scaled), values)

    def test_ema_timing(self):
        raw, averaged = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        ema = EMA(raw, averaged)
        for step in range(5):
            old = averaged.weight.detach().clone()
            with torch.no_grad():
                raw.weight.fill_(step + 1)
            decay = ema.get_decay(step)
            ema.step(step)
            torch.testing.assert_close(averaged.weight, old * decay + raw.weight * (1 - decay))
        self.assertFalse(hasattr(ema, 'optimization_step'))

    def test_rgb_policy_windows(self):
        from dexmani_policy.agents.policies.sat import SAT
        from dexmani_policy.agents.policies.dqrise import DQRISE
        from dexmani_policy.agents.policies.dexlatent import DexLatent
        from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
        base = dict(horizon=8, n_action_steps=2, num_points=8, hidden_dim=24,
                    n_layers=2, n_head=4, timestep_embed_dim=16, p_drop_attn=0., num_inference_steps=2,
                    pc_encoder_config=dict(channels=24, num_patches=2, group_size=2, depth=1, num_heads=4),
                    rgb_backbone_name='dino', rgb_backbone_config=dict(image_size=[32, 32], tune_mode='full',
                    architecture=dict(model_type='dinov2', hidden_size=24, num_hidden_layers=1,
                                      num_attention_heads=4, image_size=32, patch_size=16)))
        dims = dict(action=19, joint_state=19, point_cloud=3, rgb=3, camera_intrinsic=9, camera_extrinsic=12)
        for cls, extra in ((SAT, {'backbone_type': 'ditx'}), (SAT, {'backbone_type': 'dit'}),
                           (DQRISE, {'num_codes': 4}), (DexLatent, {})):
            for n, alpha in ((1, .5), (2, 1.), (4, 3.)):
                for bf16 in ([False, True] if n == 4 and DEVICE.startswith('cuda') else [False]):
                    with self.subTest(policy=cls.__name__, extra=extra, n=n, bf16=bf16):
                        model = cls(**base, **extra, n_obs_steps=n, time_shift_alpha=alpha)
                        model.configure_normalizer(dims, {'action': (19,), 'joint_state': (19,)})
                        if cls is DQRISE:
                            model.hand_codebook.sorted_hand_poses.uniform_(-1, 1)
                        model.to(DEVICE)
                        obs = dict(point_cloud=torch.randn(2, n, 8, 3, device=DEVICE) * .03,
                                   joint_state=torch.randn(2, n, 19, device=DEVICE),
                                   rgb=torch.rand(2, n, 3, 32, 32, device=DEVICE),
                                   camera_intrinsic=torch.tensor([16.,0,16,0,16,16,0,0,1], device=DEVICE).repeat(2,n,1),
                                   camera_extrinsic=torch.eye(3,4,device=DEVICE).flatten().repeat(2,n,1))
                        obs['point_cloud'][..., 2] += 1
                        batch = dict(obs=obs, action=torch.randn(2,8,19,device=DEVICE))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                        for step in range(3):
                            optimizer.zero_grad(set_to_none=True)
                            with torch.autocast(torch.device(DEVICE).type, dtype=torch.bfloat16, enabled=bf16):
                                loss, _ = model(batch)
                            self.assertTrue(torch.isfinite(loss))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                            if step == 2:
                                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.obs_encoder.parameters() if p.grad is not None), 0)
                            optimizer.step()
                        model.eval()
                        prediction = model.predict_action(obs)['future_actions']
                        self.assertEqual(prediction.shape, (2, 9-n, 19))
                        self.assertTrue(torch.isfinite(prediction).all())
        for alpha in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RectifiedFlow(torch.nn.Linear(1,1), time_shift_alpha=alpha)

    def test_future_force_episode_mask(self):
        import numpy as np
        from dexmani_policy.datasets.episode_store import EpisodeStore
        from dexmani_policy.datasets.variants.tactile import read_future_force
        from dexmani_policy.agents.policies.interact_dex import InterActDex
        with tempfile.TemporaryDirectory() as directory:
            source = make_source(Path(directory) / 'data.zarr')
            source.create_dataset('data/finger_tactile_force', data=np.arange(12, dtype=np.float32)[:,None,None] * np.ones((12,5,3), np.float32))
            store = EpisodeStore(Path(directory) / 'data.zarr', keys=['finger_tactile_force'])
            force, valid = read_future_force(store, 0, (4,), field='finger_tactile_force')
            self.assertTrue(valid[0])
            self.assertEqual(force[0,0,0], 4)
            force, valid = read_future_force(store, 3, (4,), field='finger_tactile_force')
            self.assertFalse(valid[0])
            self.assertEqual(force[0,0,0], 5)
        model = InterActDex(horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
                           future_force_horizon=20, hidden_dim=24, n_layers=1, n_head=4,
                           pc_encoder_config=dict(num_patches=2, group_size=2, depth=1))
        model.configure_normalizer(dict(action=19, joint_state=19, point_cloud=6,
                                        eef_pose=9, fingertip_xyz=15, finger_tactile_force=15),
                                   {'action': (19,), 'joint_state': (19,)})
        model.to(DEVICE)
        obs = dict(point_cloud=torch.randn(2,2,8,6,device=DEVICE),
                   joint_state=torch.randn(2,2,19,device=DEVICE),
                   eef_pose=torch.tensor([0.,0,0,1,0,0,0,1,0],device=DEVICE).repeat(2,2,1),
                   fingertip_xyz=torch.randn(2,2,5,3,device=DEVICE),
                   finger_tactile_force=torch.randn(2,2,5,3,device=DEVICE))
        loss, metrics = model(dict(obs=obs, action=torch.randn(2,4,19,device=DEVICE),
                            future_force=torch.randn(2,5,3,device=DEVICE),
                            future_force_valid=torch.tensor([True,False],device=DEVICE)))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {'loss', 'loss_action', 'loss_arm', 'loss_hand',
                                       'loss_future_force', 'loss_future_force_weighted',
                                       'future_force_valid_fraction', 'future_force_mae_scaled'})
        self.assertEqual(metrics['future_force_valid_fraction'].item(), .5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        prediction = torch.randn(2,5,3, requires_grad=True)
        batch = dict(future_force=torch.randn_like(prediction), future_force_valid=torch.zeros(2, dtype=torch.bool))
        loss, metrics = model._compute_future_force_loss(prediction, batch, log_metrics=True)
        self.assertTrue(all(value.item() == 0 for value in metrics.values()))
        unlogged_loss, metrics = model._compute_future_force_loss(prediction, batch, log_metrics=False)
        self.assertEqual(metrics, {})
        torch.testing.assert_close(unlogged_loss, loss, rtol=0, atol=0)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_afro_pretrain_resume_progress(self):
        from dexmani_policy.training.pretrain.obs.common import train
        from dexmani_policy.training.pretrain.obs.afro import AFRO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'source.zarr')
            epochs = []
            update = AFRO.update_teacher
            def record(model, epoch):
                epochs.append(epoch)
                update(model, epoch)
            args = ['--output-dir', str(root / 'run'), '--zarr-path', str(root / 'source.zarr'),
                    '--device', DEVICE, '--num-workers', '0', '--no-wandb', '--max-updates', '3',
                    'dataset.horizon=3', 'dataset.frame_stride=1', 'dataset.pad_after=0',
                    'dataloader.batch_size=2', 'training.num_epochs=1', '+normalization.unused=bad']
            with patch.object(AFRO, 'update_teacher', record):
                train('afro', AFRO, args)
                train('afro', AFRO, ['--resume-from', str(root / 'run'), '--total-steps', '6', '--no-wandb'])
            self.assertEqual(epochs, [0,0,0,0,1,1])
            saved = torch.load(root / 'run/checkpoints/latest.pt', weights_only=False)
            self.assertEqual(saved['global_step'], 6)
            self.assertNotIn('rng_state', saved)
            self.assertTrue((root / 'run/encoder.pt').is_file())

    def test_random_frozen_afro(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'source.zarr')
            cfg = load_policy_config('afro', [f'zarr_path={root / "source.zarr"}',
                f'workspace.output_dir={root / "run"}', f'training.device={DEVICE}',
                'workspace.wandb_cfg=null', 'horizon=4', 'n_action_steps=2',
                'agent.down_dims=[16,32]', 'agent.diffusion_step_embed_dim=16',
                'dataloader.batch_size=2', 'dataloader.num_workers=0',
                'training.loop.total_train_steps=1'])
            with closing(build_trainer(cfg)) as trainer:
                before = {k: v.clone() for k,v in trainer.model.obs_encoder.pc_encoder.state_dict().items()}
                trainer.train()
                torch.testing.assert_close(trainer.model.obs_encoder.pc_encoder.state_dict(), before)

    def test_vq_periodic_last(self):
        from dexmani_policy.training.pretrain.hand_prior.vqvae import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'source.zarr')
            cfg = load_policy_config('pretrain/hand_prior/vqvae', [
                f'zarr_path={root / "source.zarr"}', f'workspace.output_dir={root / "run"}',
                f'vq_vae.device={DEVICE}', 'vq_vae.num_epochs=3', 'vq_vae.save_epochs=2',
                'vq_vae.batch_size=3', 'vq_vae.latent_dim=4', 'vq_vae.hidden_dim=8',
                'vq_vae.num_layers=1', 'vq_vae.codebook_size=2', 'dataset.val_episodes=1'])
            published = []
            save = torch.save
            def record(payload, path, **kwargs):
                published.append(payload['epoch'])
                return save(payload, path, **kwargs)
            with patch('torch.save', side_effect=record):
                train(cfg)
            self.assertIn(2, published)
            self.assertEqual(torch.load(root / 'run/checkpoints/last.pt', weights_only=True)['epoch'], 3)
            self.assertTrue((root / 'run/checkpoints/best.pt').is_file())
            self.assertFalse(list((root / 'run/checkpoints').glob('epoch_*.pt')))

    def test_pretrained_resnet_freeze_group_norm(self):
        import torchvision
        from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNetEncoder
        source = torchvision.models.resnet18(weights=None)
        model = ResNetEncoder(tune_mode='freeze', norm_mode='group_norm', out_dim=16)
        self.assertFalse(torch.equal(model.backbone[0].weight, source.conv1.weight))
        # Isolate only the download; torchvision construction and loading remain real.
        with patch('torchvision.models._api.load_state_dict_from_url', return_value=source.state_dict()) as download:
            model.initialize_weights('DEFAULT')
        download.assert_called_once()
        torch.testing.assert_close(model.backbone[0].weight, source.conv1.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.backbone[7][1].conv2.weight, source.layer4[1].conv2.weight, rtol=0, atol=0)
        self.assertTrue(any(isinstance(m, torch.nn.GroupNorm) for m in model.backbone.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.backbone.modules()))
        model.to(DEVICE).train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        output = model(torch.rand(2, 3, 64, 64, device=DEVICE))
        self.assertEqual(output['patch_tokens'].shape, (2, 4, 16))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
        output['global_token'].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(torch.isfinite(model.proj.weight.grad).all())
        self.assertGreater(model.proj.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        torch.testing.assert_close(model.backbone[0].weight.cpu(), source.conv1.weight, rtol=0, atol=0)

    def test_resnet_policy_save_resume_and_deploy(self):
        import numpy as np
        import torchvision
        from dexmani_policy.test_tools.smoke_test import run_smoke
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_source(root / 'source.zarr')
            data.create_dataset('data/rgb', data=np.full((12, 64, 64, 3), 128, np.uint8))
            cfg = load_policy_config('dp', [
                f'zarr_path={root / "source.zarr"}', f'training.device={DEVICE}',
                'initialization.rgb=DEFAULT', 'agent.rgb_backbone_name=resnet',
                'horizon=4', 'n_action_steps=2', 'agent.down_dims=[16,32]',
                'agent.diffusion_step_embed_dim=16', 'agent.num_inference_steps=1',
                'training.use_compile=false', 'training.use_bfloat16=false',
                'training.lr_warmup_steps=0', 'training.loop.total_train_steps=2',
                'dataloader.num_workers=0', 'dataloader.batch_size=2',
                'dataset.rgb_preprocess_size=null', 'dataset.rgb_random_crop_size=null',
                'dataset.rgb_color_aug=null', 'dataset.augmentation_cfg=null',
            ])
            cfg.agent.rgb_backbone_config = dict(model_name='resnet18', tune_mode='freeze',
                                                 norm_mode='group_norm', out_dim=16, image_size=[64,64])
            source = torchvision.models.resnet18(weights=None).state_dict()
            with patch('torchvision.models._api.load_state_dict_from_url', return_value=source) as download:
                run_smoke(cfg, root / 'run', max_updates=1, resume_updates=1)
            download.assert_called_once()
            observation = {key: data[f'data/{key}'][:2] for key in cfg.dataset.sensor_modalities}
            for weights in ('raw', 'ema'):
                with self.subTest(weights=weights), closing(load_policy(
                    root / 'run/checkpoints/latest.pt', weights=weights, device=DEVICE,
                )) as runtime:
                    encoder = runtime.agent.obs_encoder.rgb_encoder
                    torch.testing.assert_close(encoder.backbone[0].weight.cpu(), source['conv1.weight'], rtol=0, atol=0)
                    self.assertFalse(any(p.requires_grad for p in encoder.backbone.parameters()))
                    prediction = runtime.predict(observation)
                    self.assertEqual(prediction.shape, (3, 19))
                    self.assertTrue(np.isfinite(prediction).all())

    def test_random_rgb_freeze_and_lora(self):
        from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
        for mode in ('freeze', 'lora'):
            model = ViTEncoder('dino', tune_mode=mode, out_dim=12, architecture=dict(
                model_type='dinov2', hidden_size=24, num_hidden_layers=1,
                num_attention_heads=4, image_size=32, patch_size=16)).to(DEVICE)
            model.initialize_weights(None)
            model(torch.rand(2,3,32,32,device=DEVICE))['global_token'].square().mean().backward()
            self.assertIsNotNone(model.proj.weight.grad)
            if mode == 'freeze':
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))


if __name__ == '__main__':
    unittest.main()
