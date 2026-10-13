import pickle

import numpy as np
import pytest
import torch
import zarr
import numcodecs


class CountingBlosc(numcodecs.Blosc):
    codec_id = 'test_counting_blosc'
    calls = 0

    def decode(self, *args, **kwargs):
        type(self).calls += 1
        return super().decode(*args, **kwargs)


@pytest.mark.parametrize('chunk_rows', [2, 5])
def test_zarr_scan_decodes_physical_chunks_once_and_copies(tmp_path, monkeypatch, chunk_rows):
    from dexmani_policy.datasets.replay_buffer import ReplayBuffer
    root = zarr.open_group(str(tmp_path), mode='w')
    numcodecs.register_codec(CountingBlosc)
    arr = root.create_dataset('data/x', shape=(11, 512, 2048), chunks=(chunk_rows, 512, 2048),
                              dtype='f4', compressor=CountingBlosc())
    for i in range(11):
        arr[i] = np.full(arr.shape[1:], i, dtype='f4')
    root.create_dataset('meta/episode_ends', data=np.array([11]))
    buffer = ReplayBuffer.open(str(tmp_path), keys=['x'])
    # Count at the codec: aligned Zarr reads can bypass Array._decode_chunk
    # and decompress straight into the output array.
    CountingBlosc.calls = 0
    for start, values in buffer.iter_chunks('x'):
        assert values.dtype == np.float32
        for i, row in enumerate(values):
            assert np.all(row == start + i)
    assert CountingBlosc.calls == (11 + chunk_rows - 1) // chunk_rows
    assert not buffer._read_cache
    for start, end in [(0, 2), (3, 7), (9, 11)]:
        values = buffer.read('x', slice(start, end), columns=slice(0, 2))
        np.testing.assert_array_equal(values[:, 0, 0], np.arange(start, end))
        values[:] = -999
        assert (buffer.read('x', slice(start, end)) >= 0).all()
    assert sum(value[1].nbytes for value in buffer._read_cache.values()) <= 16 * 1024 * 1024
    restored = pickle.loads(pickle.dumps(buffer))
    assert not restored._read_cache and restored._pid is None
    np.testing.assert_array_equal(restored.read('x', slice(10, 11)), 10)
    buffer._pid = -1  # Same branch used after a fork: reopen handles, clear cache.
    buffer.read('x', slice(10, 11))
    assert buffer._pid != -1


def old_query(radius, k, support, query):
    from dexmani_policy.agents.obs_encoder.pointcloud.ops import _pointcloud_ops, knn_point
    idx = _pointcloud_ops().ball_query(query, support, K=min(k, len(support[0])), radius=radius,
                                     return_nn=False).idx
    valid = idx >= 0
    first = idx.gather(-1, valid.long().argmax(-1, keepdim=True))
    fallback = torch.where(valid.any(-1, keepdim=True), first, knn_point(1, support, query))
    return torch.where(valid, idx, fallback.expand_as(idx))


def test_ball_query_outputs_gradients_and_empty_fallback(monkeypatch):
    from dexmani_policy.agents.obs_encoder.pointcloud import ops
    support = torch.tensor([[[0., 0, 0], [.1, 0, 0], [.1, 0, 0], [1., 0, 0]]], requires_grad=True)
    original = ops.knn_point
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    for query in (support.detach(), torch.tensor([[[10., 0, 0], [.1, 0, 0]]])):
        for radius in (0., .1, .10001, 1.):
            expected = old_query(radius, 10, support, query)
            monkeypatch.setattr(ops, 'knn_point', counted)
            calls.clear()
            actual = ops.query_ball_point(radius, 10, support, query)
            torch.testing.assert_close(actual, expected)
            if query is not None and query.shape[1] == 4 and radius > 0:
                assert not calls
            elif radius == 0 or query.shape[1] == 2:
                assert len(calls) == 1
            a = ops.index_points(support, actual).square().sum()
            b = ops.index_points(support, expected).square().sum()
            torch.testing.assert_close(torch.autograd.grad(a, support)[0], torch.autograd.grad(b, support)[0])
            monkeypatch.setattr(ops, 'knn_point', original)
    assert ops.query_ball_point(.1, 4, support[:, :1], support[:, :1]).shape == (1, 1, 1)


@pytest.mark.parametrize('pre_norm', [False, True])
def test_consistency_flow_cached_and_uncached_outputs(pre_norm, monkeypatch):
    from dexmani_policy.agents.action_decoders.backbone.consistency_ditx import ConsistencyDiTX
    from dexmani_policy.agents.action_decoders.consistency_flow import ConsistencyFlowMatch
    torch.manual_seed(42)
    model = ConsistencyDiTX(4, 19, 2, 8, n_layers=2, hidden_dim=32, n_head=4,
                            timestep_embed_dim=16, target_t_embed_dim=16, pre_norm_modality=pre_norm)
    # Avoid the all-zero initialized output concealing a cache discrepancy.
    with torch.no_grad():
        for block in model.ditx_blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.03)
        model.final_layer.ffn_final.fc2.weight.normal_(std=.03)
    decoder = ConsistencyFlowMatch(model).eval()
    context, noise = torch.randn(2, 10, 8), torch.randn(2, 4, 19)
    original = model.prepare_context_kv
    calls = []
    def counted(value):
        calls.append(1)
        return original(value)
    for steps in (1, 2, 4):
        monkeypatch.setattr(model, 'prepare_context_kv', counted)
        calls.clear()
        actual = decoder.sample_ode(noise.clone(), steps, context)
        assert len(calls) == int(steps > 1)
        monkeypatch.setattr(model, 'prepare_context_kv', lambda value: None)
        expected = decoder.sample_ode(noise.clone(), steps, context)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def trainer_for_gradients():
    from dexmani_policy.training.trainer import Trainer
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
    trainer.use_ema = False
    trainer.max_grad_norm = 0
    trainer.current_epoch = trainer.global_step = 0
    return trainer


def test_gradient_healthy_path_and_failure_diagnostics(monkeypatch):
    trainer = trainer_for_gradients()
    original = trainer._find_nonfinite_gradients
    called = []
    def locate():
        called.append(1)
        return original()
    monkeypatch.setattr(trainer, '_find_nonfinite_gradients', locate)
    trainer.apply_gradient_step()  # No gradients.
    trainer.model(torch.ones(2, 2)).sum().backward()
    trainer.apply_gradient_step()
    assert not called
    for value in (float('nan'), float('inf')):
        next(trainer.model.parameters()).grad = torch.full_like(next(trainer.model.parameters()), value)
        with pytest.raises(RuntimeError, match='Non-finite gradient'):
            trainer.apply_gradient_step()
        assert all(p.grad is None for p in trainer.model.parameters())
    assert len(called) == 2
    trainer.max_grad_norm = 1.
    next(trainer.model.parameters()).grad = torch.full_like(next(trainer.model.parameters()), 1e30)
    with pytest.raises(RuntimeError, match='non-finite'):
        trainer.apply_gradient_step()  # Aggregate norm overflow despite finite elements.
    trainer.max_grad_norm = 0
    param = next(trainer.model.parameters())
    for value in (1., float('nan')):
        param.grad = torch.sparse_coo_tensor([[0], [0]], [value], param.shape)
        if value == 1.:
            trainer.apply_gradient_step()
        else:
            with pytest.raises(RuntimeError, match='Non-finite gradient'):
                trainer.apply_gradient_step()
