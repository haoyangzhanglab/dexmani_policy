from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.datasets.common.episode_split import describe_split
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset
from dexmani_policy.training.data import build_dataset_and_normalizer, build_normalizer
from dexmani_policy.smoke_test import load_smoke_config, validate_target_imports
from dexmani_policy.agents.policies.dp import DPPolicy
from dexmani_policy.agents.policies.multitask_dit import MultiTaskDiTPolicy
from dexmani_policy.agents.obs_encoder.pointcloud.projection import project_points
from dexmani_policy.common.rgb_preprocessing import preprocess_validation_rgb
from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS


def zarr_fixture(root, *, offset=0):
    path = Path(root) / f'data_{offset}.zarr'
    group = zarr.open_group(str(path), mode='w')
    group.attrs.update(task_name='fixture', dt=.0625)
    ends = np.array([8, 16, 24, 26], dtype=np.int64)
    values = np.repeat((np.arange(26, dtype=np.float32) + offset)[:, None], 19, axis=1)
    for name in ('action', 'joint_state'):
        group.create_dataset('data/' + name, data=values, chunks=(4, 19))
    group.create_dataset('meta/episode_ends', data=ends)
    group.create_dataset('meta/episode_seeds', data=np.array([1, 2, 3, 4], np.int64))
    manifest = describe_split(str(path), ends, dict(group.attrs),
                              np.array([True, False, False, True]),
                              np.array([False, True, False, False]))
    return path, group, manifest


class DataConfigTests(unittest.TestCase):
    def test_resize_crop_projection_preserves_point_order_and_mask(self):
        rgb = torch.zeros(1, 4, 6, 3, dtype=torch.uint8)
        _, transform = preprocess_validation_rgb(rgb, resize_hw=(8, 12), center_crop_hw=(6, 8),
                                      keep_uint8=True, return_transform=True)
        # Resize centers: (x+.5)*2-.5, then crop left=2/top=1.
        torch.testing.assert_close(transform, torch.tensor([[2., 0., -1.5],
                                                            [0., 2., -.5], [0., 0., 1.]]))
        points = torch.tensor([[[1., 1., 1.], [2., 2., 1.], [0., 0., -1.], [9., 9., 1.]]])
        result = project_points(points, torch.eye(3)[None, None], torch.eye(4)[None, None],
                                transform[None, None], (6, 8))
        torch.testing.assert_close(result.confidence, torch.tensor([[[1., 1., 0., 0.]]]))
        torch.testing.assert_close(result.grid_xy[0, 0, :2],
                                  torch.tensor([[-.75, -1/3], [-.25, 1/3]]))

    def test_latent_flow_query_windows_and_relative_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, group, manifest = zarr_fixture(tmp)
            group['data/action'][16:24] = np.nan
            dataset = LatentFlowDataset(str(path), sensor_modalities=['joint_state'],
                data_semantics=HAND_ACTION_SEMANTICS, horizon=4, obs_horizon=2, saved_split=manifest)
            self.assertEqual(len(dataset), 5)  # Short episode cannot supply a full future chunk.
            first, last = dataset[0], dataset[-1]
            torch.testing.assert_close(first['obs']['joint_state'][:, 0], torch.tensor([0., 0.]))
            torch.testing.assert_close(last['action'][:, 0], torch.tensor([4., 5., 6., 7.]))
            torch.testing.assert_close(last['action_context']['reference_state'], torch.full((19,), 4.))
            stats = build_normalizer(dataset, {'action': 'limits'}, 'action')['action'].input_stats
            torch.testing.assert_close(stats['min'], torch.zeros(19))
            torch.testing.assert_close(stats['max'], torch.full((19,), 3.))
            torch.testing.assert_close(stats['mean'], torch.full((19,), 1.5))

    def test_selected_episode_scanning_and_role_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, group, manifest = zarr_fixture(tmp)
            group['data/joint_state'][2, 0] = np.nan
            group['data/action'][10, 0] = np.inf
            group['data/action'][16:24] = np.nan
            reads = []
            original = zarr.Array.__getitem__
            def read(array, selection):
                if array.path.startswith('data/'):
                    rows = selection[0] if isinstance(selection, tuple) else selection
                    reads.append((array.path, rows))
                return original(array, selection)
            with patch.object(zarr.Array, '__getitem__', read):
                dataset = DemonstrationDataset(str(path), sensor_modalities=['joint_state'],
                    horizon=4, obs_horizon=2, pad_before=1, pad_after=3, saved_split=manifest)
            self.assertTrue(reads)
            for _, rows in reads:
                self.assertTrue(rows.stop <= 16 or rows.start >= 24, reads)
            self.assertFalse(dataset.action_valid[16:24].any())
            np.testing.assert_array_equal(dataset.sampler.observation_source_rows,
                                          [0, 1, 3, 4, 5, 6, 7, 24, 25])
            np.testing.assert_array_equal(dataset.sampler.action_source_rows,
                                          [0, 1, 2, 3, 4, 5, 6, 7, 24, 25])
            validation = dataset.get_validation_dataset()
            self.assertFalse(np.isin(10, validation.sampler.action_source_rows))
            for view in (dataset, validation):
                for index in range(len(view)):
                    batch = view[index]
                    self.assertTrue(torch.isfinite(batch['obs']['joint_state']).all())
                    self.assertTrue(torch.isfinite(batch['action']).all())
            stats = build_normalizer(dataset, {'action': 'limits', 'joint_state': 'limits'}, 'action')
            self.assertEqual(stats['action'].input_stats['max'][0], 25)

    def test_lamp_history_never_reads_excluded_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, group, manifest = zarr_fixture(tmp)
            group['data/action'][16:24] = np.nan
            dataset = LAMPDataset(str(path), sensor_modalities=['joint_state'], history_steps=3,
                                  horizon=1, obs_horizon=1, saved_split=manifest)
            np.testing.assert_array_equal(dataset.history_for_frame(24), np.full((3, 12), 24))
            np.testing.assert_array_equal(dataset.history_for_frame(25), np.full((3, 12), 24))
            self.assertFalse(dataset.history_valid[16:24].any())

    def test_zero_sampling_weight_does_not_change_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            children = []
            for offset in (0, 100):
                path, _, manifest = zarr_fixture(tmp, offset=offset)
                children.append(DemonstrationDataset(str(path), sensor_modalities=['joint_state'],
                    horizon=1, saved_split=manifest))
            ds = MultiTaskDataset(children, ['a', 'b'], sampling_strategy='weighted', task_weights=[1, 0])
            normalizer = build_normalizer(ds, {'action': 'limits'}, 'action')
            self.assertEqual(normalizer['action'].input_stats['max'][0], 125)

    def test_data_builder_returns_facts_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, manifest = zarr_fixture(tmp)
            cfg = load_smoke_config('dp3')
            cfg.dataset.sensor_modalities = ['joint_state']
            cfg.dataset.zarr_path = str(path)
            cfg.dataset.augmentation_cfg = None
            cfg.normalization = {'joint_state': 'limits', 'action': 'auto'}
            before = OmegaConf.to_container(cfg, resolve=True)
            dataset, normalizer, facts = build_dataset_and_normalizer(cfg, verbose=False)
            self.assertEqual(before, OmegaConf.to_container(cfg, resolve=True))
            self.assertEqual(facts['normalization_dims'], {'action': 19, 'joint_state': 19})

    def test_rgb_shared_weights_and_visual_freeze(self):
        rgb = dict(model_name='resnet18', weights=None, load_pretrained=False,
                   tune_mode='freeze', norm_mode='batch_norm', image_size=[32, 32])
        dp = DPPolicy(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19,
                      rgb_backbone_name='resnet', rgb_backbone_config=rgb, state_dim=19,
                      down_dims=(8, 16), diffusion_step_embed_dim=16, n_groups=4)
        multi = MultiTaskDiTPolicy(horizon=8, n_obs_steps=2, n_action_steps=3,
                      rgb_backbone_name='resnet', rgb_backbone_config=rgb,
                      task_texts=['fixture'], text_embed_dim=4, n_emb=16, n_layers=1, num_heads=4)
        left, right = dp.obs_encoder, multi.obs_encoder
        right.load_state_dict(left.state_dict(), strict=True)
        self.assertEqual(list(left.state_dict()), list(right.state_dict()))
        self.assertTrue(all(k.startswith(('backbone.', 'state_mlp.')) for k in left.state_dict()))
        obs = {'joint_state': torch.ones(2, 19), 'rgb': torch.zeros(2, 32, 32, 3, dtype=torch.uint8)}
        left.eval(); right.eval()
        torch.testing.assert_close(left(obs), right(obs), rtol=0, atol=0)
        left.train(); left(obs).sum().backward()
        self.assertFalse(left.backbone.backbone.training)
        self.assertTrue(all(p.grad is None for p in left.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in left.state_mlp.parameters()))

    def test_nine_default_configs(self):
        for name in ('dp', 'dp3', 'sat', 'r3d', 'maniflow', 'multitask_dit', 'dqrise', 'lamp', 'latent_flow'):
            with self.subTest(name=name):
                validate_target_imports(load_smoke_config(name))


if __name__ == '__main__':
    unittest.main()
