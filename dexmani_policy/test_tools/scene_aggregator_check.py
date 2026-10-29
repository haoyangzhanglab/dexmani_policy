"""Synthetic scene contracts, optional compile checks, and controlled module benchmarks."""

import argparse
import copy
import json
import statistics
import time
from unittest import TestCase

import torch

from dexmani_policy.agents.obs_encoder.interaction.scene_aggregator import SceneAggregator


GEOMETRY_KEYS = ('scene_center', 'scene_covariance', 'scene_mass', 'scene_mask')
OUTPUT_KEYS = ('scene_token', *GEOMETRY_KEYS)


def _assert_gradients(model, inputs, output, mask=None):
    model.zero_grad(set_to_none=True)
    for value in inputs:
        value.grad = None
    probe = torch.randn_like(output['scene_token'])
    (output['scene_token'] * probe).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name in ('geometry_embed', 'cross_refinement', 'ffn'):
        module = getattr(model, name)
        if module is not None:
            assert all(p.grad.abs().sum() > 0 for p in module.parameters()), name
    if model.learned_query is not None:
        assert model.learned_query.grad.abs().sum() > 0
    for value in inputs:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    if mask is not None:
        for value in inputs[:2]:
            assert not value.grad[~mask].any()


def _assert_geometry(output, mask):
    assignment, weight = output['assignment'], output['pool_weight']
    torch.testing.assert_close(assignment.sum(1), mask.float())
    torch.testing.assert_close(weight.sum(-1), torch.ones_like(output['scene_mass']))
    torch.testing.assert_close(output['scene_mass'].sum(-1), mask.sum(-1).float())
    assert (output['scene_mass'] >= 1).all()
    assert not assignment.masked_select(~mask[:, None]).any()
    covariance = output['scene_covariance']
    torch.testing.assert_close(covariance, covariance.transpose(-1, -2))
    assert torch.linalg.eigvalsh(covariance.double()).min() >= -1e-8
    for name in ('assignment', 'pool_weight', 'pooled_token', *GEOMETRY_KEYS[:-1]):
        assert output[name].dtype == torch.float32, name
    assert output['scene_mask'].dtype == torch.bool and output['scene_mask'].all()
    assert all(torch.isfinite(output[name]).all() for name in OUTPUT_KEYS)


def _check_content_reading(device):
    # Two distant groups. Keep the probed region's pool/query fixed and change
    # only a patch outside its support, so the effect must pass through K/V.
    points = torch.tensor([[0., 0., 0.], [.01, .003, 0.], [.02, 0., .002],
                           [1., 0., 0.], [1.01, .005, 0.], [1.03, 0., .001]], device=device)[None]
    tokens = torch.randn(1, 6, 192, device=device)
    model = SceneAggregator(num_scene_tokens=2, nearest_slots=1).to(device).eval()
    reference = model(tokens, points, True)
    region = reference['scene_center'][0, :, 0].argmin().item()
    far = 4
    assert reference['pool_weight'][0, region, far] == 0
    changed = tokens.clone()
    changed[:, far] = torch.randn_like(changed[:, far]) * 4
    modified = model(changed, points, True)
    torch.testing.assert_close(reference['base_scene_token'][:, region], modified['base_scene_token'][:, region])
    assert (reference['refinement_update'][:, region] - modified['refinement_update'][:, region]).abs().max() > 1e-5
    assert (reference['scene_token'][:, region] - modified['scene_token'][:, region]).abs().max() > 1e-5

    mask = torch.ones(1, 6, device=device, dtype=torch.bool)
    mask[:, far] = False
    masked = model(tokens, points, patch_mask=mask)
    torch.testing.assert_close(masked, model(changed, points, patch_mask=mask), rtol=0, atol=0)

    hard = SceneAggregator(num_scene_tokens=2, nearest_slots=1, refinement_mode='hard').to(device)
    hard.load_state_dict(model.state_dict())
    hard_a, hard_b = hard(tokens, points, True), hard(changed, points, True)
    torch.testing.assert_close(hard_a['scene_token'][:, region], hard_b['scene_token'][:, region], rtol=0, atol=0)
    assert torch.isneginf(hard_a['attention_bias'][0, :, region, far]).all()

    # Equal QK scores isolate the positional bias from feature-dependent scores.
    reader = model.cross_refinement
    query = torch.randn(1, 1, 192, device=device)
    memory = tokens[:, :3]
    distance_bias = torch.tensor([[[0., -1., -5.]]], device=device)
    with torch.no_grad():
        reader.q.weight.zero_()
        reader.kv.weight[:reader.heads * reader.head_dim].zero_()
    weight = torch.ones_like(distance_bias) / 3
    bias = reader.attention_bias(distance_bias, weight, None)
    moved_bias = reader.attention_bias(distance_bias.flip(-1), weight, None)
    probabilities = bias.softmax(-1)
    assert (probabilities[:, :3, :, 0] > probabilities[:, :3, :, 2]).all()
    torch.testing.assert_close(probabilities[:, 3], torch.full_like(probabilities[:, 3], 1 / 3))
    torch.testing.assert_close(bias[:, 3], moved_bias[:, 3], rtol=0, atol=0)
    # Select only global-head values at the output, then only local-head values.
    with torch.no_grad():
        reader.out.weight[:, :3 * reader.head_dim].zero_()
    torch.testing.assert_close(reader(query, memory, bias), reader(query, memory, moved_bias), rtol=0, atol=0)
    with torch.no_grad():
        reader.out.weight.normal_()
        reader.out.weight[:, 3 * reader.head_dim:].zero_()
    assert not torch.allclose(reader(query, memory, bias), reader(query, memory, moved_bias))
    valid = torch.tensor([[True, True, False]], device=device)
    masked_bias = reader.attention_bias(distance_bias, weight, valid)
    assert torch.isneginf(masked_bias[..., -1]).all()
    for global_heads in (0, 4):
        variant = SceneAggregator(num_scene_tokens=2, global_heads=global_heads).to(device)
        out = variant(tokens, points, True)
        assert torch.isfinite(out['attention_bias']).all()
        if global_heads == 4:
            assert not out['attention_bias'].any()


def validate_scene_aggregator(device):
    torch.manual_seed(47)
    parameters = {}
    mixed_dtypes = []
    if device.type == 'cpu' or (device.type == 'cuda' and torch.cuda.is_bf16_supported()):
        mixed_dtypes.append(torch.bfloat16)
    if device.type == 'cuda':
        mixed_dtypes.append(torch.float16)
    for channels in (192, 384):
        model = SceneAggregator(token_channels=channels, num_scene_tokens=6).to(device).eval()
        tokens = torch.randn(2, 31, channels, device=device, requires_grad=True)
        points = (torch.randn(2, 31, 3, device=device) * .15).requires_grad_()
        context = torch.randn(2, channels, device=device, requires_grad=True)
        mask = torch.ones(2, 31, device=device, dtype=torch.bool)
        mask[0, -4:] = False
        mask[1, -7:] = False
        with torch.no_grad():
            tokens[~mask] = float('nan')
            points[0, -4:] = float('inf')
            points[1, -7:] = float('nan')
        output = model(tokens, points, True, patch_mask=mask, query_context=context)
        assert output['scene_token'].shape == (2, 6, channels)
        assert model.out_shape == (6, channels) and model.out_dim == channels
        _assert_geometry(output, mask)
        assert torch.isfinite(output['attention_bias'].masked_select(mask[:, None, None])).all()
        assert torch.isneginf(output['attention_bias'].masked_select(~mask[:, None, None])).all()

        # Deleting padding must match masking it, including all geometry and content.
        with torch.no_grad():
            for row in range(2):
                keep = mask[row]
                deleted = model(tokens[row:row + 1, keep], points[row:row + 1, keep], True,
                                query_context=context[row:row + 1])
                for name in (*OUTPUT_KEYS, 'pooled_token', 'refinement_update'):
                    torch.testing.assert_close(output[name][row:row + 1], deleted[name])
                torch.testing.assert_close(output['assignment'][row:row + 1, :, keep], deleted['assignment'])
            changed = model(torch.randn_like(tokens), points, True, patch_mask=mask,
                            query_context=-context)
            for name in (*GEOMETRY_KEYS, 'assignment', 'pool_weight', 'region_anchors'):
                torch.testing.assert_close(changed[name], output[name], rtol=0, atol=0)
            changed_context = model(tokens, points, True, patch_mask=mask, query_context=-context)
            for name in (*GEOMETRY_KEYS, 'pooled_token', 'base_scene_token', 'attention_bias'):
                torch.testing.assert_close(changed_context[name], output[name], rtol=0, atol=0)
            assert not torch.allclose(changed_context['refinement_update'], output['refinement_update'])
            assert not torch.allclose(changed_context['scene_token'], output['scene_token'])
            no_context = model(tokens, points, patch_mask=mask)
            zero_context = model(tokens, points, patch_mask=mask, query_context=torch.zeros_like(context))
            torch.testing.assert_close(no_context, zero_context, rtol=0, atol=0)

            perm = torch.randperm(tokens.shape[1], device=device)
            shuffled = model(tokens[:, perm], points[:, perm], True, patch_mask=mask[:, perm],
                             query_context=context)
            for name in OUTPUT_KEYS:
                torch.testing.assert_close(shuffled[name], output[name], rtol=1e-4, atol=2e-6)
            torch.testing.assert_close(perm[shuffled['anchor_indices']], output['anchor_indices'])
            torch.testing.assert_close(shuffled['assignment'], output['assignment'][:, :, perm])

            rotation, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
            rotation[:, 0] *= torch.linalg.det(rotation)
            translation = points.new_tensor([.3, -.2, .1])
            moved = model(tokens, points @ rotation.T + translation, True, patch_mask=mask)
            torch.testing.assert_close(moved['assignment'], output['assignment'], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(moved['scene_center'], output['scene_center'] @ rotation.T + translation)
            torch.testing.assert_close(moved['scene_covariance'], rotation @ output['scene_covariance'] @ rotation.T)
            torch.testing.assert_close(moved['scene_mass'], output['scene_mass'])

        _assert_gradients(model, (tokens, points, context), output, mask)
        for dtype in mixed_dtypes:
            with torch.autocast(device.type, dtype=dtype):
                mixed = model(tokens, points, True, patch_mask=mask, query_context=context)
            _assert_geometry(mixed, mask)
            for name in GEOMETRY_KEYS:
                torch.testing.assert_close(mixed[name], output[name], rtol=0, atol=0)
            torch.testing.assert_close(mixed['scene_token'], output['scene_token'], rtol=.03, atol=.03)
            _assert_gradients(model, (tokens, points, context), mixed, mask)

        pooled = SceneAggregator(token_channels=channels, use_cross_refinement=False).to(device)
        torch.testing.assert_close(pooled(tokens, points, patch_mask=mask),
                                   pooled(tokens, points, patch_mask=mask, query_context=context), rtol=0, atol=0)
        learned = SceneAggregator(token_channels=channels, num_scene_tokens=6, query_source='learned').to(device)
        result = learned(tokens, points, patch_mask=mask, query_context=context)
        for name in GEOMETRY_KEYS:
            torch.testing.assert_close(result[name], output[name], rtol=0, atol=0)
        _assert_gradients(learned, (tokens, points, context), result, mask)
        parameters[channels] = sum(p.numel() for p in SceneAggregator(token_channels=channels).parameters())

    # K=1 has uniform assignment: compare moments against an independent FP64 oracle.
    model = SceneAggregator(num_scene_tokens=1).to(device)
    points = torch.tensor([[[0., 0., 0.], [.2, 0., 0.], [0., .3, 0.], [0., 0., .4]]], device=device)
    output = model(torch.randn(1, 4, 192, device=device), points, True)
    centered = points.double() - points.double().mean(1, keepdim=True)
    torch.testing.assert_close(output['scene_center'], points.mean(1, keepdim=True))
    expected_covariance = (centered.transpose(1, 2) @ centered / 4)[:, None].float()
    torch.testing.assert_close(output['scene_covariance'], expected_covariance)
    torch.testing.assert_close(output['assignment'], torch.ones_like(output['assignment']))

    # A nonuniform analytic Cauchy allocation; seed ownership overrides the kernel.
    points = torch.zeros(1, 4, 3, device=device)
    points[0, :, 0] = points.new_tensor([0., .05, .10, .30])
    model = SceneAggregator(num_scene_tokens=2, nearest_slots=2).to(device)
    output = model(torch.randn(1, 4, 192, device=device), points, True)
    expected_assignment = points.new_tensor([[[0., 5/34, 2/7, 1.], [1., 29/34, 5/7, 0.]]])
    torch.testing.assert_close(output['anchor_indices'], torch.tensor([[3, 0]], device=device))
    torch.testing.assert_close(output['assignment'], expected_assignment)
    for region in range(2):
        weights = expected_assignment[0, region].double()
        weights = weights / weights.sum()
        mean = sum(weights[i] * points[0, i].double() for i in range(4))
        covariance = sum(weights[i] * torch.outer(points[0, i].double() - mean,
                                                 points[0, i].double() - mean) for i in range(4))
        torch.testing.assert_close(output['scene_center'][0, region], mean.float())
        torch.testing.assert_close(output['scene_covariance'][0, region], covariance.float())

    # Coincident points, distance ties, and exactly K valid indices stay nonempty.
    model = SceneAggregator(num_scene_tokens=4).to(device)
    tokens = torch.randn(1, 9, 192, device=device)
    for valid_count in (4, 9):
        mask = torch.arange(9, device=device)[None] < valid_count
        output = model(tokens, torch.zeros(1, 9, 3, device=device), True, patch_mask=mask)
        _assert_geometry(output, mask)
        assert output['anchor_indices'].unique().numel() == 4
        assert not output['scene_center'].any() and not output['scene_covariance'].any()
    tc = TestCase()
    with tc.assertRaisesRegex(ValueError, 'at least K'):
        model(tokens, torch.zeros(1, 9, 3, device=device), patch_mask=mask & False)
    with tc.assertRaisesRegex(ValueError, 'at least K'):
        model(tokens, torch.zeros(1, 9, 3, device=device), patch_mask=torch.arange(9, device=device)[None] < 3)
    with tc.assertRaises(ValueError):
        model(tokens[:, :3], torch.zeros(1, 3, 3, device=device))
    with tc.assertRaises(ValueError):
        model(tokens, torch.zeros(1, 9, 3, device=device), patch_mask=mask.float())
    with tc.assertRaises(ValueError):
        model(tokens, torch.zeros(1, 9, 3, device=device), query_context=tokens[:, :1])
    with tc.assertRaises(TypeError):
        model(tokens.long(), torch.zeros(1, 9, 3, device=device))
    _check_content_reading(device)
    return {'status': 'passed', 'device': str(device), 'channels': [192, 384], 'parameters': parameters,
            'autocast_forward_backward': [str(dtype) for dtype in mixed_dtypes],
            'checks': 'shape/mask, coverage/moments/PSD, permutation/rigid transform, masked NaN/Inf deletion, '
                      'input/coordinate/context/parameter gradients, context isolation, distant reads, local/global heads, '
                      'pool/hard/soft/learned ablations, coincident seeds, insufficient support'}


def validate_compile(device, backend):
    reports = []
    for masked in (False, True):
        torch.manual_seed(59)
        eager = SceneAggregator(num_scene_tokens=4).to(device).eval()
        candidate = copy.deepcopy(eager)
        compiled = torch.compile(candidate, backend=backend, fullgraph=not masked)
        tokens = torch.randn(2, 24, 192, device=device, requires_grad=True)
        points = (torch.randn(2, 24, 3, device=device) * .1).requires_grad_()
        context = torch.randn(2, 192, device=device, requires_grad=True)
        mask = torch.arange(24, device=device)[None].expand(2, -1) < 21 if masked else None
        expected = eager(tokens, points, patch_mask=mask, query_context=context)
        actual = compiled(tokens, points, patch_mask=mask, query_context=context)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        probe = torch.randn_like(expected['scene_token'])
        expected_grads = torch.autograd.grad(
            (expected['scene_token'] * probe).mean(), (tokens, points, context, *eager.parameters())
        )
        actual_grads = torch.autograd.grad(
            (actual['scene_token'] * probe).mean(), (tokens, points, context, *candidate.parameters())
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=2e-5)
        if masked:
            # A backend must not silently drop the valid-count boundary check.
            with TestCase().assertRaisesRegex(ValueError, 'at least K'):
                compiled(tokens, points, patch_mask=mask & False, query_context=context)
        trace = torch._dynamo.explain(candidate)(tokens, points, patch_mask=mask, query_context=context)
        reports.append({'mask': masked, 'forward_backward': 'passed', 'fullgraph': not masked,
                        'graph_count': trace.graph_count, 'graph_break_count': trace.graph_break_count,
                        'dtype': 'float32', 'query_context': True})
    return {'backend': backend, 'device': str(device), 'cases': reports}


def benchmark(model, *, device, batch, channels, dtype, backward, warmup, iterations, repeats):
    """Time eager module batches; excludes data transfer, optimizer and upstream/action networks."""
    torch.manual_seed(67)
    model = model.to(device).train(backward)
    tokens = torch.randn(batch, 96, channels, device=device, requires_grad=backward)
    points = torch.randn(batch, 96, 3, device=device) * .15
    probe = torch.randn(batch, model.num_scene_tokens, channels, device=device)

    def step():
        if backward:
            model.zero_grad(set_to_none=True)
            tokens.grad = None
        with torch.set_grad_enabled(backward), torch.autocast(
            device.type, dtype=dtype, enabled=dtype != torch.float32
        ):
            output = model(tokens, points)['scene_token']
            if backward:
                loss = (output * probe).mean()
        if backward:
            loss.backward()

    for _ in range(warmup):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000 / iterations)
    return {'batch': batch, 'channels': channels, 'dtype': str(dtype), 'backward': backward,
            'parameters': sum(p.numel() for p in model.parameters()),
            'median_ms': statistics.median(elapsed), 'min_ms': min(elapsed), 'max_ms': max(elapsed),
            'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--compile-backend', choices=('aot_eager', 'inductor'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--channels', type=int, nargs='+', default=[192, 384])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 128])
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    print(json.dumps(validate_scene_aggregator(device), ensure_ascii=False), flush=True)
    if args.compile_backend:
        print(json.dumps(validate_compile(device, args.compile_backend)), flush=True)
    if args.benchmark:
        print(json.dumps({'torch': torch.__version__, 'device': str(device),
                          'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                          'threads': torch.get_num_threads(), 'warmup': args.warmup,
                          'iterations': args.iterations, 'repeats': args.repeats,
                          'M': 96, 'K': 12, 'mask': False, 'context': None,
                          'input_dtype': 'float32', 'parameter_dtype': 'float32',
                          'timing': 'perf_counter; CUDA synchronized per block; median block mean',
                          'execution': 'eager; forward or forward+backward; no optimizer'}), flush=True)
        for channels in args.channels:
            for name, options in (('pool', {'use_cross_refinement': False}),
                                  ('hard', {'refinement_mode': 'hard'}), ('soft', {})):
                for batch in args.batch_sizes:
                    for backward in (False, True):
                        result = benchmark(
                            SceneAggregator(token_channels=channels, **options), device=device,
                            batch=batch, channels=channels, dtype=getattr(torch, args.dtype),
                            backward=backward, warmup=args.warmup, iterations=args.iterations,
                            repeats=args.repeats,
                        )
                        print(json.dumps({'variant': name, **result}), flush=True)


if __name__ == '__main__':
    main()
