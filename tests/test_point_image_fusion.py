"""Fusion-only regression tests; no vision weights, point backends or policy imports."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_PATH = Path(__file__).resolve().parents[1] / "dexmani_policy/agents/obs_encoder/interaction/point_image_fusion.py"
_SPEC = importlib.util.spec_from_file_location("_point_image_fusion_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PointImageFusion = _MODULE.PointImageFusion
build_patch_image_weights = _MODULE.build_patch_image_weights


def inputs(dtype=torch.float32, views=2, grid=(3, 5)):
    torch.manual_seed(17)
    b, n, m, k, ci, cp = 2, 19, 5, 6, 12, 8
    local = torch.randn(b, m, cp, dtype=dtype, requires_grad=True)
    image = torch.randn(b, views, grid[0] * grid[1], ci, dtype=dtype, requires_grad=True)
    idx = torch.randint(n, (b, m, k))
    uv = torch.rand(b, views, n, 2, dtype=dtype) * torch.tensor([40., 24.], dtype=dtype)
    confidence = torch.rand(b, views, n, dtype=dtype)
    uv[0, 0, 0] = float("nan")
    confidence[1, :, 1] = 0
    return local, idx, image, uv, confidence, (24, 40), grid


def pointwise_reference(image, idx, uv, confidence, hw, grid):
    """Independent high-memory reference, intentionally NOT the production path."""
    b, v, _, ci = image.shape
    finite = torch.isfinite(uv).all(-1) & torch.isfinite(confidence)
    u, y = uv.unbind(-1)
    valid = finite & (u >= -.5) & (u < hw[1] - .5) & (y >= -.5) & (y < hw[0] - .5)
    weights = torch.where(valid, confidence.clamp(0, 1), 0)
    den = weights.sum(1, keepdim=True)
    weights = weights / torch.where(den > 0, den, 1)
    safe_uv = torch.where(valid[..., None], uv, 0)
    scale = safe_uv.new_tensor([hw[1], hw[0]])
    coords = (2 * (safe_uv + .5) / scale - 1).reshape(b * v, -1, 1, 2)
    fmap = image.permute(0, 1, 3, 2).reshape(b * v, ci, *grid)
    sampled = F.grid_sample(fmap, coords, padding_mode="border", align_corners=False)
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, v, -1, ci)
    points = (sampled * weights[..., None]).sum(1)
    members = torch.stack([points[s][idx[s]] for s in range(b)])
    member_valid = torch.stack([(den[:, 0] > 0)[s][idx[s]] for s in range(b)])
    return members.sum(2) / member_valid.sum(2).clamp_min(1)[..., None]


def baseline_fuse(model, local, pooled, geometry):
    semantic = model.proj(model.image_norm(pooled))
    quality = torch.stack((geometry["semantic_coverage"], geometry["semantic_confidence"]), -1)
    gate = model.gate(torch.cat((model.geometry_norm(local), model.semantic_norm(semantic), quality), -1)).sigmoid()
    return local + torch.where(geometry["semantic_valid_mask"][..., None], gate * semantic, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("views", [1, 3])
def test_pool_matches_pointwise_forward_and_all_parameter_gradients(dtype, views):
    data = inputs(dtype, views)
    local, idx, image, uv, confidence, hw, grid = data
    model = PointImageFusion(12, 8).to(dtype)
    with torch.no_grad():
        model.gate.weight.normal_(std=.05)
    other = copy.deepcopy(model)
    local_ref = local.detach().clone().requires_grad_()
    image_ref = image.detach().clone().requires_grad_()
    out = model(*data, return_intermediate=True)
    expected_pool = pointwise_reference(image_ref, idx, uv, confidence, hw, grid)
    expected = baseline_fuse(other, local_ref, expected_pool, out)
    tol = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(out["semantic_token"], expected_pool, atol=tol, rtol=tol)
    torch.testing.assert_close(out["fused_token"], expected, atol=tol, rtol=tol)
    actual_grads = torch.autograd.grad(out["fused_token"].square().mean(), (local, image, *model.parameters()))
    expected_grads = torch.autograd.grad(expected.square().mean(), (local_ref, image_ref, *other.parameters()))
    for actual, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected_grad, atol=tol, rtol=tol)


def test_pool_parameter_schema_is_unchanged():
    model = PointImageFusion(12, 8)
    expected = {f"{name}.{suffix}" for name in ("image_norm", "proj", "geometry_norm", "semantic_norm", "gate") for suffix in ("weight", "bias")}
    assert set(model.state_dict()) == expected
    restored = PointImageFusion(12, 8)
    restored.load_state_dict(model.state_dict(), strict=True)
    data = inputs()
    torch.testing.assert_close(model(*data)["fused_token"], restored(*data)["fused_token"], atol=0, rtol=0)


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_correspondence_reuse_and_stop_gradient(alignment):
    local, idx, image, uv, confidence, hw, grid = inputs()
    uv.requires_grad_()
    confidence.requires_grad_()
    model = PointImageFusion(12, 8, alignment=alignment)
    geometry = build_patch_image_weights(idx, uv, confidence, hw, grid)
    assert all(not tensor.requires_grad for tensor in geometry.values())
    out = model.fuse_tokens(local, image, geometry, True)
    ref = model(local, idx, image, uv, confidence, hw, grid, True)
    for name in out:
        torch.testing.assert_close(out[name], ref[name], atol=0, rtol=0)
    out["fused_token"].sum().backward()
    assert uv.grad is None and confidence.grad is None
    assert local.grad is not None and image.grad is not None


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_all_invisible_is_exact_residual_with_finite_backward(alignment):
    local, idx, image, uv, confidence, hw, grid = inputs()
    uv[:] = float("nan")
    image = torch.full_like(image, float("nan"), requires_grad=True)
    model = PointImageFusion(12, 8, alignment=alignment)
    # Learned biases must not create a semantic update without observation.
    with torch.no_grad():
        model.proj.bias.fill_(5)
        model.gate.bias.fill_(8)
        if alignment == "content":
            model.content_scale.fill_(2)
    out = model(local, idx, image, uv, confidence, hw, grid, True)
    torch.testing.assert_close(out["fused_token"], local, atol=0, rtol=0)
    assert not out["semantic_valid_mask"].any()
    assert torch.count_nonzero(out["semantic_gate"]) == 0
    assert torch.count_nonzero(out["semantic_weight"]) == 0
    out["fused_token"].sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local), atol=0, rtol=0)
    assert torch.count_nonzero(image.grad) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_masked_image_tokens_never_contribute_even_with_nan(alignment):
    local = torch.randn(1, 1, 8, requires_grad=True)
    image = torch.randn(1, 1, 4, 12)
    image[:, :, 1:] = float("nan")
    image.requires_grad_()
    idx = torch.zeros(1, 1, 1, dtype=torch.long)
    uv = torch.tensor([[[[1.5, 1.5]]]])  # four equal bilinear corners
    confidence = torch.tensor([[[.8]]])
    mask = torch.tensor([[[True, False, False, False]]])
    model = PointImageFusion(12, 8, alignment=alignment)
    out = model(local, idx, image, uv, confidence, (4, 4), (2, 2), True, image_token_valid=mask)
    torch.testing.assert_close(out["pool_weight"], torch.tensor([[[1., 0, 0, 0]]]))
    torch.testing.assert_close(out["semantic_confidence"], torch.tensor([[.2]]))
    torch.testing.assert_close(out["semantic_token"], image[:, 0, :1])
    out["fused_token"].square().sum().backward()
    assert torch.isfinite(local.grad).all() and torch.isfinite(image.grad).all()
    assert not image.grad[:, :, 1:].any()


def test_member_weights_coverage_and_duplicate_edge_coalescing():
    # Three members: first point twice and second once. Zero-weight duplicate
    # does not affect the prior; no high-dimensional member features needed.
    idx = torch.tensor([[[0, 0, 1], [0, 0, 1]]])
    uv = torch.tensor([[[[.5, .5], [2.5, .5]]]])
    confidence = torch.ones(1, 1, 2)
    members = torch.tensor([[[1., 0., 3.], [0., float("nan"), -1.]]])
    geo = build_patch_image_weights(idx, uv, confidence, (2, 4), (1, 2), member_weight=members)
    torch.testing.assert_close(geo["pool_weight"], torch.tensor([[[.25, .75], [0., 0.]]]))
    assert geo["semantic_valid_mask"].tolist() == [[True, False]]
    scaled = build_patch_image_weights(idx, uv, confidence, (2, 4), (1, 2), member_weight=members * 1e20)
    torch.testing.assert_close(geo["pool_weight"], scaled["pool_weight"])
    confidence[..., 1] = 0
    geo = build_patch_image_weights(idx, uv, confidence, (2, 4), (1, 2), member_weight=members)
    torch.testing.assert_close(geo["semantic_coverage"], torch.tensor([[.25, 0.]]))
    torch.testing.assert_close(geo["pool_weight"], torch.tensor([[[1., 0.], [0., 0.]]]))


def test_borders_nonfinite_uv_and_zero_weights():
    uv = torch.tensor([[[[-.5, -.5], [3.499, 1.499], [3.5, .5], [.5, 1.5], [float("inf"), 0], [0, 0]]]])
    idx = torch.arange(6).reshape(1, 6, 1)
    confidence = torch.tensor([[[1., 1., 1., 1., 1., float("nan")]]])
    geo = build_patch_image_weights(idx, uv, confidence, (2, 4), (1, 2))
    assert geo["semantic_valid_mask"].tolist() == [[True, True, False, False, False, False]]
    torch.testing.assert_close(geo["pool_weight"].sum(-1), geo["semantic_valid_mask"].float())
    torch.testing.assert_close(geo["pool_weight"][0, :2], torch.eye(2))


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_view_member_and_point_permutations(alignment):
    local, idx, image, uv, confidence, hw, grid = inputs(views=3)
    model = PointImageFusion(12, 8, alignment=alignment)
    if alignment == "content":
        with torch.no_grad():
            model.content_scale.fill_(.8)
    expected = model(local, idx, image, uv, confidence, hw, grid)["fused_token"]
    perm = torch.randperm(uv.shape[2])
    inverse = torch.argsort(perm)
    actual = model(local, inverse[idx.flip(-1)], image.flip(1), uv.flip(1)[:, :, perm], confidence.flip(1)[:, :, perm], hw, grid)["fused_token"]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_duplicate_identical_views_do_not_amplify_semantics():
    local, idx, image, uv, confidence, hw, grid = inputs(views=1)
    model = PointImageFusion(12, 8)
    a = model(local, idx, image, uv, confidence, hw, grid, True)
    b = model(local, idx, image.repeat(1, 3, 1, 1), uv.repeat(1, 3, 1, 1), confidence.repeat(1, 3, 1), hw, grid, True)
    torch.testing.assert_close(a["semantic_token"], b["semantic_token"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a["fused_token"], b["fused_token"], atol=2e-6, rtol=2e-6)


def test_zero_content_scale_recovers_pool_and_can_start_learning():
    data = inputs()
    pool = PointImageFusion(12, 8)
    content = PointImageFusion(12, 8, alignment="content")
    result = content.load_state_dict(pool.state_dict(), strict=False)
    assert set(result.missing_keys) == {"content_scale", "query.weight", "key.weight"}
    assert not result.unexpected_keys
    a, b = pool(*data, True), content(*data, True)
    torch.testing.assert_close(a["fused_token"], b["fused_token"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a["pool_weight"], b["semantic_weight"], atol=2e-6, rtol=2e-6)
    b["fused_token"].square().mean().backward()
    assert content.content_scale.grad.abs() > 1e-10
    assert content.query.weight.grad is not None and not content.query.weight.grad.any()
    with torch.no_grad():
        content.content_scale.fill_(.5)
    content.zero_grad(set_to_none=True)
    content(*data)["fused_token"].square().mean().backward()
    assert content.query.weight.grad.abs().sum() > 0
    assert content.key.weight.grad.abs().sum() > 0


def test_geometry_content_selects_only_supported_image_tokens():
    model = PointImageFusion(4, 4, alignment="content", attention_dim=4)
    with torch.no_grad():
        model.query.weight.copy_(torch.eye(4))
        model.key.weight.copy_(torch.eye(4))
        model.content_scale.fill_(3)
    image = torch.tensor([[[[1., 0, 0, 0], [0, 1., 0, 0], [float("nan")] * 4]]], requires_grad=True)
    local = torch.tensor([[[1., 0, 0, 0], [0, 1., 0, 0]]], requires_grad=True)
    geometry = {"pool_weight": torch.tensor([[[.5, .5, 0], [.5, .5, 0]]]),
                "semantic_valid_mask": torch.ones(1, 2, dtype=torch.bool),
                "semantic_coverage": torch.ones(1, 2), "semantic_confidence": torch.ones(1, 2)}
    out = model.fuse_tokens(local, image, geometry, True)
    assert out["semantic_weight"][0, 0, 0] > .9
    assert out["semantic_weight"][0, 1, 1] > .9
    assert not out["semantic_weight"][..., 2].any()
    out["fused_token"].sum().backward()
    assert torch.isfinite(image.grad).all() and not image.grad[:, :, 2].any()


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_cpu_bfloat16_autocast_forward_backward(alignment):
    data = inputs()
    model = PointImageFusion(12, 8, alignment=alignment)
    if alignment == "content":
        with torch.no_grad():
            model.content_scale.fill_(.4)
    ref = model(*data, True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(*data, True)
    assert out["semantic_gate"].dtype == torch.float32
    assert out["semantic_token"].dtype == torch.float32
    assert out["semantic_weight"].dtype == torch.float32
    torch.testing.assert_close(out["fused_token"], ref["fused_token"], atol=3e-3, rtol=2e-2)
    out["fused_token"].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("grid", [(1, 1), (1, 7), (7, 1), (14, 14), (16, 16)])
def test_grid_sizes_and_constant_semantic_preservation(grid):
    local, idx, image, uv, confidence, hw, _ = inputs(grid=grid)
    image = torch.full_like(image, 2.)
    out = PointImageFusion(12, 8)(local, idx, image, uv, confidence, hw, grid, True)
    torch.testing.assert_close(out["semantic_token"], torch.full_like(out["semantic_token"], 2.))
    torch.testing.assert_close(out["pool_weight"].sum(-1), out["semantic_valid_mask"].float())


@pytest.mark.parametrize("kwargs", [{"alignment": "qformer"}, {"attention_dim": 0}, {"gate_init": 0}, {"gate_init": float("nan")}])
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        PointImageFusion(12, 8, **kwargs)


def test_reject_bad_shapes_masks_and_indices():
    local, idx, image, uv, confidence, hw, grid = inputs()
    model = PointImageFusion(12, 8)
    with pytest.raises(ValueError, match="prefix"):
        model(local, idx, image[:, :, :-1], uv, confidence, hw, grid)
    with pytest.raises(ValueError, match="int64"):
        build_patch_image_weights(idx.float(), uv, confidence, hw, grid)
    with pytest.raises(ValueError, match="positive integers"):
        build_patch_image_weights(idx, uv, confidence, hw, (0, 5))
    with pytest.raises(ValueError, match="positive integers"):
        build_patch_image_weights(idx, uv, confidence, 224, grid)
    with pytest.raises(ValueError, match="positive integers"):
        build_patch_image_weights(idx, uv, confidence, hw, ("16", 16))
    with pytest.raises(ValueError, match="image_token_valid"):
        model(local, idx, image, uv, confidence, hw, grid, image_token_valid=torch.ones_like(image[..., 0]))
    with pytest.raises(ValueError, match="member_weight"):
        model(local, idx, image, uv, confidence, hw, grid, member_weight=torch.ones(2, 5))
    bad_idx = idx.clone()
    bad_idx[0, 0, 0] = -1
    with pytest.raises(RuntimeError):
        build_patch_image_weights(bad_idx, uv, confidence, hw, grid)


def test_float64_autograd_gradcheck_with_fixed_correspondence():
    model = PointImageFusion(3, 3, alignment="content", attention_dim=2).double()
    with torch.no_grad():
        model.content_scale.fill_(.3)
    local = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    image = torch.randn(1, 1, 3, 3, dtype=torch.float64, requires_grad=True)
    geometry = {"pool_weight": torch.tensor([[[.6, .4, 0], [0, .2, .8]]], dtype=torch.float64),
                "semantic_valid_mask": torch.ones(1, 2, dtype=torch.bool),
                "semantic_coverage": torch.ones(1, 2, dtype=torch.float64),
                "semantic_confidence": torch.ones(1, 2, dtype=torch.float64)}
    assert torch.autograd.gradcheck(lambda x, y: model.fuse_tokens(x, y, geometry)["fused_token"], (local, image))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_cuda_bfloat16_forward_backward():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 not supported")
    data = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs())
    for alignment in ("pool", "content"):
        model = PointImageFusion(12, 8, alignment=alignment).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(*data, True)
        assert torch.isfinite(out["fused_token"]).all()
        out["fused_token"].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_aot_eager_fullgraph_forward_backward(alignment):
    # Graph-capture/autograd check, NOT an Inductor or CUDA performance claim.
    data = inputs()
    model = PointImageFusion(12, 8, alignment=alignment)
    if alignment == "content":
        with torch.no_grad():
            model.content_scale.fill_(.4)
    expected = model(*data)["fused_token"]
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    actual = compiled(*data)["fused_token"]
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_masked_view_support_is_not_given_full_view_confidence():
    idx = torch.zeros(1, 1, 1, dtype=torch.long)
    uv = torch.full((1, 2, 1, 2), 1.5)
    confidence = torch.ones(1, 2, 1)
    mask = torch.tensor([[[True, False, False, False], [True, True, True, True]]])
    geometry = build_patch_image_weights(idx, uv, confidence, (4, 4), (2, 2), image_token_valid=mask)
    # View 0 supplies only 1/4 interpolation mass; view 1 supplies full mass.
    torch.testing.assert_close(geometry["pool_weight"], torch.tensor([[[.2, 0, 0, 0, .2, .2, .2, .2]]]))
    mask[:, 0] = False
    geometry = build_patch_image_weights(idx, uv, confidence, (4, 4), (2, 2), image_token_valid=mask)
    torch.testing.assert_close(geometry["pool_weight"], torch.tensor([[[0., 0, 0, 0, .25, .25, .25, .25]]]))
    mask[:] = False
    geometry = build_patch_image_weights(idx, uv, confidence, (4, 4), (2, 2), image_token_valid=mask)
    assert not geometry["pool_weight"].any() and not geometry["semantic_valid_mask"].any()


@pytest.mark.parametrize("alignment", ["pool", "content"])
def test_noncontiguous_features_and_strict_support(alignment):
    local, idx, image, uv, confidence, hw, grid = inputs()
    model = PointImageFusion(12, 8, alignment=alignment)
    expected = model(local, idx, image, uv, confidence, hw, grid)["fused_token"]
    noncontiguous = image.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not noncontiguous.is_contiguous()
    actual = model(local, idx, noncontiguous, uv, confidence, hw, grid)["fused_token"]
    torch.testing.assert_close(actual, expected)


def test_double_member_weights_are_normalized_before_downcast():
    idx = torch.tensor([[[0, 1]]])
    uv = torch.tensor([[[[.5, .5], [2.5, .5]]]], dtype=torch.float32)
    geometry = build_patch_image_weights(idx, uv, torch.ones(1, 1, 2), (2, 4), (1, 2), member_weight=torch.tensor([[[1e300, 3e300]]], dtype=torch.float64))
    assert geometry["pool_weight"].dtype == torch.float64
    torch.testing.assert_close(geometry["pool_weight"], torch.tensor([[[.25, .75]]], dtype=torch.float64))
