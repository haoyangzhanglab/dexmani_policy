"""Controlled CPU replay through the actual fresh and original-run resume paths."""

import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state
from dexmani_policy.train import run_training
from dexmani_policy.training.common.config import (
    load_policy_config, prepare_training, validate_config, validate_config_cli,
)
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.common.resume import prepare_resume


class ConfigRecipeChecks(unittest.TestCase):
    def test_fixed_recipes_and_research_options(self):
        for name in ("dp", "dp3", "r3d", "sat", "dqrise", "adaptac", "interact_dex", "maniflow", "afro", "cordvip"):
            with self.subTest(config=name):
                cfg = load_policy_config(name)
                validate_config(cfg)
                self.assertEqual(cfg.normalization.action, "limits")
                self.assertEqual(set(cfg.normalization), set(cfg.dataset.sensor_modalities) | {"action"})
                self.assertEqual(cfg.agent.action_dim, 19)
        spatial = load_policy_config("sat")
        self.assertEqual(dict(spatial.normalization), dict(
            action="limits", joint_state="limits", point_cloud="identity", rgb="identity",
            camera_intrinsic="identity", camera_extrinsic="identity",
        ))
        for name, overrides in (
            ("dp3", ["normalization.point_cloud=gaussian", "normalization.joint_state=gaussian"]),
            ("maniflow", ["normalization.point_cloud=limits"]),
            ("adaptac", ["tactile=real_patch", "normalization.finger_tactile_force=max_abs",
                          "normalization.taxel_tactile_force=max_abs"]),
            ("interact_dex", ["tactile=real_force", "normalization.finger_tactile_force=max_abs"]),
        ):
            with self.subTest(config=name, overrides=overrides):
                validate_config(load_policy_config(name, overrides))

    def test_fixed_fields_are_not_input_options(self):
        for name, override in (
            ("dp3", "+normalization.action=limits"),
            ("dp", "+normalization.rgb=identity"),
            ("sat", "+normalization.joint_state=limits"),
            ("r3d", "+normalization.point_cloud=identity"),
            ("interact_dex", "+normalization.eef_pose=gaussian"),
            ("pretrain/hand_prior/vqvae", "+normalization.action=limits"),
        ):
            with self.subTest(config=name, override=override), self.assertRaisesRegex(ValueError, "fixed"):
                load_policy_config(name, [override])
        cfg = load_policy_config("adaptac", ["normalization.finger_tactile_force=gaussian"])
        with self.assertRaisesRegex(ValueError, "finger_tactile_force"):
            validate_config(cfg)

    def test_repository_config_names_and_pretraining(self):
        self.assertEqual(load_policy_config("dp3.yaml").agent.action_dim, 19)
        self.assertEqual(load_policy_config("pretrain/hand_prior/vqvae.yaml").normalization.action, "limits")
        self.assertEqual(load_policy_config("pretrain/obs/afro").normalization.point_cloud, "limits")
        self.assertEqual(load_policy_config("pretrain/obs/cordvip").normalization.joint_state, "limits")
        for name in ("/tmp/outside.yaml", "../configs/dp3", "~/.config/policy.yaml"):
            with self.subTest(config=name), self.assertRaisesRegex(ValueError, "Config"):
                load_policy_config(name)
        for arguments in (["-cp", "/tmp"], ["--config-dir=/tmp"], ["--config-p=/tmp"],
                          ["--config-n=../outside"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_config_cli(arguments)
        with self.assertRaisesRegex(ValueError, "searchpath"):
            load_policy_config("dp3", ["hydra.searchpath=[file:///tmp]"])


class TrainingResumeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_state_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def make_config(self, root, run):
        data = root / 'data.zarr'
        if not data.exists():
            store = zarr.open_group(str(data), mode='w')
            rng = np.random.default_rng(9)
            for name, shape in [('joint_state', (18, 19)), ('action', (18, 19)),
                                ('point_cloud', (18, 16, 3))]:
                store.create_dataset(f'data/{name}', data=rng.normal(size=shape).astype('float32'))
            store.create_dataset('meta/episode_ends', data=np.array([6, 12, 18]))
        cfg = load_policy_config('dp3', [
            f'zarr_path={data}', 'horizon=4', 'n_obs_steps=2', 'n_action_steps=2',
            'agent.num_points=16', 'agent.pc_dim=3', 'agent.pc_out_dim=16',
            '+agent.state_out_dim=8', 'agent.down_dims=[16]',
            'agent.diffusion_step_embed_dim=16', 'agent.n_groups=4',
            'agent.num_training_steps=10', 'agent.num_inference_steps=2',
            'dataset.augmentation_cfg=null', 'dataset.val_ratio=0',
            'dataloader.batch_size=2', 'dataloader.num_workers=0',
            'training.device=cpu', 'training.use_compile=false', 'training.use_bfloat16=false',
            'training.loop.total_train_steps=9', 'training.loop.log_interval_steps=1',
            'training.loop.gradient_accumulation_steps=2', 'training.lr_warmup_steps=1',
        ])
        cfg.workspace.output_dir = str(run)
        cfg.workspace.wandb_cfg = None
        return prepare_training(cfg)

    def test_continuous_matches_original_run_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous, resumed = root / 'continuous', root / 'resumed'
            run_training(self.make_config(root, continuous))
            run_training(self.make_config(root, resumed), max_updates=3)
            original_config = (resumed / 'config.yaml').read_bytes()
            cfg, checkpoint = prepare_resume(resumed, device='cpu', num_workers=0)
            self.assertEqual(cfg.training.loop.total_train_steps, 9)
            self.assertEqual(checkpoint.global_step, 3)
            run_training(cfg, checkpoint=checkpoint)
            self.assertEqual((resumed / 'config.yaml').read_bytes(), original_config)
            a = load_checkpoint(continuous / 'checkpoints/latest.pt')
            b = load_checkpoint(resumed / 'checkpoints/latest.pt')
            self.assertEqual(a.global_step, b.global_step)
            for name in ['model_state', 'ema_model_state', 'optimizer_state', 'scheduler_state']:
                self.assert_state_equal(getattr(a, name), getattr(b, name))
            self.assert_state_equal(a.rng_state['torch'], b.rng_state['torch'])
            for run in (continuous, resumed):
                steps = [json.loads(line)['step'] for line in (run / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual(steps, list(range(1, 10)))
            self.assertFalse(list(resumed.glob('metrics.*.bak')))
            with self.assertRaises(FileExistsError):
                ExperimentLogger(str(resumed))

    def test_signal_saves_at_update_boundary(self):
        from dexmani_policy.training.trainer import PolicyTrainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self.make_config(root, root / "interrupted")
            original_step = PolicyTrainer.optimizer_step

            def stop_after_update(trainer):
                original_step(trainer)
                os.kill(os.getpid(), signal.SIGTERM)

            with patch.object(PolicyTrainer, "optimizer_step", stop_after_update):
                self.assertEqual(run_training(cfg), 1)
            resumed_cfg, checkpoint = prepare_resume(root / "interrupted")
            self.assertEqual(checkpoint.global_step, 1)
            self.assertEqual(run_training(resumed_cfg, checkpoint=checkpoint, max_updates=1), 2)

    def test_resume_can_reenable_wandb_after_disabled_updates(self):
        with tempfile.TemporaryDirectory() as temporary, patch('wandb.init') as initialize:
            root = Path(temporary)
            run = root / 'logging'
            cfg = self.make_config(root, run)
            backend = {'project': 'resume-test', 'mode': 'offline'}
            with open_dict(cfg):
                cfg.workspace.wandb_cfg = backend
                cfg.workspace.wandb_enabled = False
            run_training(cfg, max_updates=1)
            cfg, checkpoint = prepare_resume(run)
            self.assertEqual(dict(cfg.workspace.wandb_cfg), backend)
            run_training(cfg, checkpoint=checkpoint, max_updates=1)
            initialize.assert_not_called()
            cfg, checkpoint = prepare_resume(run, wandb=True)
            self.assertEqual(dict(cfg.workspace.wandb_cfg), backend)
            run_training(cfg, checkpoint=checkpoint, max_updates=1)
            initialize.assert_called_once()
            self.assertEqual(initialize.call_args.kwargs['project'], backend['project'])
            initialize.return_value.finish.assert_called_once()

    def test_log_tail_is_backed_up_only_when_ahead(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = ExperimentLogger(str(root))
            for step in [1, 2, 3]:
                logger.log({'value': step}, step=step)
            logger.close()
            original = (root / 'metrics.jsonl').read_bytes()
            logger = ExperimentLogger(str(root), resume_step=2)
            logger.log({'value': 3}, step=3)
            logger.close()
            backups = list(root.glob('metrics.*.bak'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual((root / 'metrics.jsonl').read_bytes(), original)
            ExperimentLogger(str(root), resume_step=3).close()
            self.assertEqual(list(root.glob('metrics.*.bak')), backups)

    def test_cpu_rng_restore_accepts_saved_cuda_state(self):
        state = get_rng_state()
        state['torch_cuda'] = torch.tensor([1], dtype=torch.uint8)
        set_rng_state(state, device='cpu')
        self.assert_state_equal(torch.get_rng_state(), state['torch'])

    def test_action_layouts_fit_only_training_frames(self):
        from dexmani_policy.training.common.data import build_dataset, fit_normalizer

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / 'actions.zarr'
            store = zarr.open_group(str(data), mode='w')
            rows = np.array([2, 4, 6, 100, 200, 300], dtype=np.float32)[:, None]
            for key, width in [('action', 19), ('action_ee', 21), ('joint_state', 19)]:
                store.create_dataset(f'data/{key}', data=np.repeat(rows, width, axis=1))
            store.create_dataset('data/point_cloud', data=np.zeros((6, 4, 6), dtype=np.float32))
            store.create_dataset('meta/episode_ends', data=np.array([3, 6]))
            manifest = dict(episode_ends=[3, 6], episode_ids=None, source_id=None,
                            trial_ids=None, train_indices=[0], val_indices=[1])
            cases = [('dp3', 'action', False, 19), ('dp3', 'action_ee', False, 21),
                     ('r3d', 'action', True, 28)]
            for method, action_key, auxiliary, width in cases:
                with self.subTest(method=method, action_key=action_key):
                    overrides = [f'zarr_path={data}', f'action_key={action_key}',
                                 'horizon=3', 'n_obs_steps=1', 'n_action_steps=1',
                                 'dataset.augmentation_cfg=null']
                    if auxiliary:
                        overrides.append('use_aux_ee=true')
                    cfg = load_policy_config(method, overrides)
                    cfg.dataset.split_manifest = manifest
                    dataset, description = build_dataset(cfg)
                    self.assertEqual(description['normalization_dims']['action'], width)
                    affine = fit_normalizer(dataset, cfg.normalization,
                                            feature_dims=description['normalization_dims'])['action']
                    scale, offset = torch.full((width,), 0.5), torch.full((width,), -2.0)
                    if action_key == 'action_ee':
                        scale[3:9], offset[3:9] = 1, 0
                    torch.testing.assert_close(affine.scale, scale, rtol=0, atol=0)
                    torch.testing.assert_close(affine.offset, offset, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
