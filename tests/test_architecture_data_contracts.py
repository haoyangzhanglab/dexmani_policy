"""CPU regressions for SAT tokens, Dataset-owned RGB crop and caller masks."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from dexmani_policy.agents.core.dp import DPObsEncoder
from dexmani_policy.agents.core.sat import SATObsEncoder
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.smoke_test import load_config


class Tokenizer(nn.Module):
    supports_global_token = True
    out_shape = (2, 3)

    def forward(self, pc, **kwargs):
        patch_tokens = torch.arange(12.0).reshape(2, 2, 3)
        return patch_tokens, torch.zeros(2, 2, 3), patch_tokens.mean(1, keepdim=True)


def sat_encoder():
    with patch('dexmani_policy.agents.core.sat.build_pc_patch_tokenizer', return_value=Tokenizer()):
        encoder = SATObsEncoder('stub', 3, 1, 4, 2, state_out_dim=1)
    encoder.state_mlp = nn.Identity()
    return encoder


def test_sat_rejects_dense_at_construction():
    with pytest.raises(ValueError, match='global token'):
        SATObsEncoder('pointnet_dense', 3, 19, 4, 2)


def test_sat_preserves_temporal_feature_fusion():
    encoder = sat_encoder()
    obs = {'point_cloud': torch.zeros(2, 4, 3), 'joint_state': torch.tensor([[20.], [30.]])}
    with patch('dexmani_policy.agents.core.sat.preprocess_point_cloud', side_effect=lambda x, *a, **k: x):
        cond, _ = encoder(obs)
    tokens, _, glob = encoder.pc_encoder(obs['point_cloud'])
    frame_features = torch.cat([torch.cat([glob, tokens], 1), obs['joint_state'][:, None].expand(2, 3, 1)], -1)
    expected = torch.cat([frame_features[0], frame_features[1]], -1).unsqueeze(0)
    torch.testing.assert_close(cond, expected)
    assert cond.shape == (1, 3, 8)


@pytest.mark.parametrize('output', [(torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)),
                                    (torch.zeros(2, 2, 3), torch.zeros(2, 2, 3), None),
                                    (torch.zeros(2, 2, 3), torch.zeros(2, 2, 3), torch.zeros(2, 3))])
def test_sat_rejects_invalid_forward_contract(output):
    encoder = sat_encoder()
    with patch.object(encoder.pc_encoder, 'forward', return_value=output), patch('dexmani_policy.agents.core.sat.preprocess_point_cloud', side_effect=lambda x, *a, **k: x):
        with pytest.raises(ValueError, match='SAT tokenizer'):
            encoder({'point_cloud': torch.zeros(2, 4, 3), 'joint_state': torch.zeros(2, 1)})


@pytest.mark.parametrize('ratio', [None, .95])
def test_rgb_legacy_crop_rejected_before_backbone_build(ratio):
    with patch('dexmani_policy.agents.core.dp.build_backbone') as builder:
        with pytest.raises(ValueError, match='Dataset rgb_random_crop_size'):
            DPObsEncoder('resnet', 19, 2, rgb_backbone_config={'crop_ratio': ratio})
        builder.assert_not_called()


def test_rgb_crop_owned_by_dataset_only():
    dataset = object.__new__(BaseDataset)
    dataset._is_val = False
    dataset.rgb_preprocess_size = (12, 12)
    dataset.rgb_random_crop_size = (8, 8)
    dataset.rgb_keep_uint8 = True
    dataset.rgb_color_aug = None
    rgb = dataset._preprocess_rgb_cpu(np.zeros((2, 14, 14, 3), dtype=np.uint8))
    seen = []

    class Backbone(nn.Module):
        out_dim = 3
        def forward(self, value):
            seen.append(value.shape)
            return {'global_token': value.float().mean((-1, -2))}

    processor = SimpleNamespace(process_images=lambda value: {'image': value})
    with patch('dexmani_policy.agents.core.dp.build_backbone', return_value=(Backbone(), processor)):
        encoder = DPObsEncoder('stub', 19, 2)
    encoder.train()
    encoder({'rgb': rgb, 'joint_state': torch.zeros(2, 19)})
    assert seen == [torch.Size([2, 3, 8, 8])]


def test_sequence_sampler_does_not_mutate_readonly_mask():
    buffer = SimpleNamespace(episode_ends=np.array([2, 12]), keys=lambda: [])
    mask = np.array([True, True])
    mask.flags.writeable = False
    sampler = SequenceSampler(buffer, sequence_length=4, episode_mask=mask)
    assert mask.tolist() == [True, True]
    assert len(sampler) == 7
    assert np.all(sampler.indices[:, 0] >= 2)


@pytest.mark.parametrize('name', ['dp', 'multitask_dit', 'ddp/dp', 'ddp/multitask_dit'])
def test_rgb_configs_keep_dataset_crop_and_dimensions(name):
    cfg = load_config(name)
    assert 'crop_ratio' not in cfg.agent.rgb_backbone_config
    assert (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps) == (16, 2, 8)
    assert cfg.agent.action_dim == 19 and cfg.agent.state_dim == 19
    datasets = cfg.dataset.get('datasets', [cfg.dataset])
    for dataset in datasets:
        assert list(dataset.rgb_random_crop_size) == [224, 224]
        assert (dataset.pad_before, dataset.pad_after) == (1, 7)
