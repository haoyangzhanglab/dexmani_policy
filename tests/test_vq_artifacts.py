import copy
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

from dexmani_policy.agents.normalization import LinearNormalizer
from dexmani_policy.agents.vq_hand.vqvae import VQVAEHand
from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager, compact_artifact_metadata
from dexmani_policy.agents.vq_hand.export_codebook import extract_codebook, deployment_diagnostics


def source_checkpoint():
    config = dict(hand_dim=12, loss_weight=[1.] * 12, latent_dim=16, hidden_dim=32,
                  num_groups=2, codebook_size=4, num_layers=1, act_scale=1., vq_decay=.8,
                  threshold_ema_dead_code=0, kmeans_iters=10)
    torch.manual_seed(8)
    model = VQVAEHand(**config)
    normalizer = LinearNormalizer()
    normalizer.fit({'hand': torch.randn(40, 12) * torch.arange(1, 13)})
    return {'format_version': 3, 'epoch': 3, 'model_state_dict': model.state_dict(),
            'model_config': config, 'normalizer_state_dict': normalizer.state_dict(), 'metrics': {'val_mse': .3},
            'split_metadata': {'normalization_spec': {'action': 'limits'},
                               'normalization_scope': 'unique_valid_policy_train_action_source_rows',
                               'train_source_rows': list(range(10000)), 'val_source_rows': list(range(10000, 12000)),
                               'train_frame_count': 10000, 'val_frame_count': 2000}}


def deep_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(deep_size(v, seen) for v in value)
    return sys.getsizeof(value)


def test_vq_export_snapshot_survives_atomic_source_replacement(tmp_path, monkeypatch):
    original = source_checkpoint()
    path = tmp_path / 'best.pt'
    torch.save(original, path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement = copy.deepcopy(original)
    replacement['epoch'] = 9
    for key, value in replacement['model_state_dict'].items():
        if key.startswith('decoder.') and key.endswith('bias'):
            value.add_(.1)
    other = tmp_path / 'next.pt'
    torch.save(replacement, other)
    original_constructor = VQVAEHand.from_checkpoint
    def swap_then_construct(checkpoint, **kwargs):
        os.replace(other, path)
        return original_constructor(checkpoint, **kwargs)
    monkeypatch.setattr(VQVAEHand, 'from_checkpoint', swap_then_construct)
    manager = extract_codebook(str(path), str(tmp_path / 'codebook.npz'), device='cpu')
    assert manager.artifact_metadata['source_checkpoint_sha256'] == source_hash
    assert manager.artifact_metadata['source_epoch'] == 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() != source_hash
    monkeypatch.setattr(VQVAEHand, 'from_checkpoint', original_constructor)
    torch.save(original, tmp_path / 'original.pt')
    reference = extract_codebook(str(tmp_path / 'original.pt'), str(tmp_path / 'reference.npz'), device='cpu')
    torch.testing.assert_close(manager.sorted_hand_poses, reference.sorted_hand_poses)


def test_metadata_size_lookup_diagnostics_and_embedded_restore(tmp_path):
    checkpoint = source_checkpoint()
    path, output = tmp_path / 'source.pt', tmp_path / 'codebook.npz'
    torch.save(checkpoint, path)
    manager = extract_codebook(str(path), str(output), device='cpu')
    old = dict(manager.artifact_metadata, split_metadata=checkpoint['split_metadata'])
    compact = compact_artifact_metadata(old)
    assert 'split_metadata' not in compact
    assert compact['source_summary']['train_source_rows_count'] == 10000
    assert deep_size(compact) < deep_size(old) / 10
    with np.load(output) as data:
        payload = {key: data[key] for key in data.files}
    payload['metadata_json'] = np.asarray(json.dumps(old, sort_keys=True))
    old_path = tmp_path / 'old.npz'
    np.savez(old_path, **payload)
    old_bytes, new_bytes = old_path.stat().st_size, output.stat().st_size
    assert new_bytes < old_bytes / 5
    restored = CodebookManager(hand_dim=12, num_groups=2, codebook_size=4)
    restored.load(old_path)
    assert 'split_metadata' not in restored.artifact_metadata
    embedded = CodebookManager(hand_dim=12, num_groups=2, codebook_size=4)
    embedded.load_state_dict(restored.state_dict())
    old_path.unlink(); output.unlink()
    rows = torch.randn(31, 12)
    for other in (restored, embedded):
        torch.testing.assert_close(other.hand_pose_to_continuous_index(rows), manager.hand_pose_to_continuous_index(rows))
    assert all(torch.is_tensor(value) for value in embedded.state_dict().values())
    diagnostics = deployment_diagnostics(manager, {'train': rows, 'held_out': rows[:7]}, chunk_size=5)
    idx = manager.hand_pose_to_continuous_index(rows)
    reconstructed, discrete = manager.continuous_index_to_hand_pose(idx)
    # Deployment subtracts FP32 reconstructions first, then accumulates in FP64.
    error = (reconstructed.reshape_as(rows) - rows).double()
    # 31 * 12 positive terms: 1e-12 comfortably covers FP64 reduction-order
    # roundoff (~N * eps), while remaining well below an FP32 oracle's error.
    assert diagnostics['splits']['train']['normalized']['mae'] == pytest.approx(
        float(error.abs().mean()), rel=1e-12, abs=1e-12)
    physical = error / manager.hand_normalizer_scale.double()
    assert diagnostics['splits']['train']['physical']['mae'] == pytest.approx(
        float(physical.abs().mean()), rel=1e-12, abs=1e-12)
    assert sum(diagnostics['splits']['train']['prototype_counts']) == len(rows)
    assert diagnostics['splits']['held_out']['count'] == 7
    empty = deployment_diagnostics(manager, {'held_out': np.empty((0, 12))})
    assert empty['splits']['held_out']['status'].startswith('NOT VERIFIED')
    print('metadata dict bytes before/after:', deep_size(old), deep_size(compact))
    print('NPZ bytes before/after:', old_bytes, new_bytes)
