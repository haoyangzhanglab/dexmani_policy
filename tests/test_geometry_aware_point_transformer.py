"""GAPT module checks. CPU reference FPS/kNN do not validate native kernels."""
import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from dexmani_policy.agents.obs_encoder.pointcloud import geometry_patch as gp


@torch.no_grad()
def reference_fps(points, num_samples, **kwargs):
    b, n, _ = points.shape
    selected = torch.empty(b, num_samples, dtype=torch.long, device=points.device)
    nearest = torch.full((b, n), float("inf"), device=points.device)
    first = torch.zeros(b, dtype=torch.long, device=points.device)
    if kwargs.get("use_random", True) and kwargs.get("use_random_start", True):
        first = torch.randint(n, (b,), device=points.device)
    for i in range(num_samples):
        selected[:, i] = first
        center = gp.index_points(points, first[:, None])
        nearest = torch.minimum(nearest, (points - center).square().sum(-1))
        first = nearest.argmax(-1)
    if kwargs.get("use_random", True) and kwargs.get("use_shuffle_output", True):
        perm = torch.rand(b, num_samples, device=points.device).argsort(-1)
        selected = selected.gather(1, perm)
    return gp.index_points(points, selected), selected


def reference_knn(k, support, query):
    return torch.cdist(query, support, compute_mode="donot_use_mm_for_euclid_dist").topk(k, largest=False, sorted=False).indices


@pytest.fixture
def reference_geometry(monkeypatch):
    monkeypatch.setattr(gp, "farthest_point_sample", reference_fps)
    monkeypatch.setattr(gp, "knn_point", reference_knn)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    yield
    torch.set_num_threads(threads)


def small_encoder(**kwargs):
    return gp.GeometryAwarePointTransformer(channels=48, num_patches=16, group_size=8, depth=2, **kwargs)


@pytest.mark.parametrize("input_channels", [3, 6])
def test_shapes_geometry_backward_and_single_output(reference_geometry, input_channels):
    model = small_encoder(input_channels=input_channels).eval()
    points = (torch.randn(2, 64, input_channels) * 0.12).requires_grad_()
    result = model(points, return_geometry=True)
    assert result["tokens"].shape == (2, 16, 48)
    assert result["xyz"].dtype == torch.float32
    assert result["mask"].dtype == torch.bool and result["mask"].all()
    assert model.out_shape == (16, 48) and model.out_dim == 48
    assert set(model(points)) == {"tokens", "xyz", "mask"}
    assert not hasattr(gp, "GeometryPatchEncoder")
    torch.testing.assert_close(result["xyz"], gp.index_points(points[..., :3], result["center_indices"]))
    radius = (gp.index_points(points[..., :3], result["neighbor_indices"]) - result["xyz"][:, :, None]).norm(dim=-1).amax(dim=2, keepdim=True)
    torch.testing.assert_close(radius, result["patch_radius"])
    (result["tokens"] - torch.randn_like(result["tokens"])).square().mean().backward()
    assert torch.isfinite(points.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_full_default_budget_and_optimizer_step(reference_geometry):
    model = gp.GeometryAwarePointTransformer().train()
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 11814084 and parameter_count < 30_000_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.patch_embedding.member_mlp[0].weight.detach().clone()
    result = model(torch.randn(1, 1024, 6) * 0.12)
    assert result["tokens"].shape == (1, 256, 384)
    loss = (result["tokens"] - torch.randn_like(result["tokens"])).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.patch_embedding.member_mlp[0].weight)


def test_eval_determinism_and_strict_reload(reference_geometry):
    model = small_encoder().eval()
    p = torch.randn(2, 64, 6) * 0.12
    a = model(p)
    torch.manual_seed(100)
    b = model(p)
    restored = small_encoder().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    c = restored(p)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)


def test_member_order_pooling_and_degenerate_patches(reference_geometry):
    stem = gp.GeometricPatchEmbedding(6, 48, 0.1).eval()
    rel = torch.randn(2, 16, 8, 3) * 0.03
    rgb = torch.rand_like(rel)
    perm = torch.randperm(8)
    a, radius = stem(rel, rgb)
    b, radius_b = stem(rel[:, :, perm], rgb[:, :, perm])
    torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(radius, radius_b)
    zero, radius_zero = stem(torch.zeros_like(rel), rgb)
    assert torch.isfinite(zero).all() and radius_zero.eq(0).all()


def test_metric_translation_and_physical_scale(reference_geometry):
    model = small_encoder().eval()
    p = torch.randn(1, 64, 6) * 0.1
    a = model(p)
    moved = p.clone()
    shift = torch.tensor([0.13, -0.11, 0.17])
    moved[..., :3] += shift
    b = model(moved)
    torch.testing.assert_close(a["tokens"], b["tokens"], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(a["xyz"] + shift, b["xyz"], atol=1e-6, rtol=1e-6)
    resized = p.clone()
    resized[..., :3] *= 1.7
    assert not torch.allclose(model(resized)["tokens"], a["tokens"])


def test_fixed_anchor_permutation_and_cross_patch_influence(reference_geometry):
    model = small_encoder().eval()
    features = torch.randn(2, 16, 48)
    xyz = torch.randn(2, 16, 3) * 0.1
    perm = torch.randperm(16)

    def contextualize(f, c):
        distance = torch.cdist(c, c, compute_mode="donot_use_mm_for_euclid_dist")
        distance = distance.masked_fill(torch.eye(16, dtype=torch.bool)[None], float("inf"))
        idx = distance.topk(12, largest=False, sorted=False).indices
        delta = (gp.index_points(c, idx) - c[:, :, None]) / 0.1
        edges = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), dim=-1)
        f = model.local_relation(f, idx, edges)
        cos, sin = model.rope(c)
        for block in model.blocks:
            f = block(f, cos, sin)
        return model.norm(f)

    a = contextualize(features, xyz)
    b = contextualize(features[:, perm], xyz[:, perm])
    torch.testing.assert_close(a[:, perm], b, atol=3e-6, rtol=2e-5)
    changed = features.clone()
    changed[:, -1] += torch.linspace(-2, 2, 48)
    assert not torch.allclose(contextualize(changed, xyz)[:, 0], a[:, 0])


def test_duplicate_inputs_and_autocast(reference_geometry):
    model = small_encoder()
    points = torch.zeros(2, 64, 6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(points)
        loss = (result["tokens"].float() - torch.randn(2, 16, 48)).square().mean()
    loss.backward()
    assert torch.isfinite(result["tokens"]).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kwargs", [
    {"input_channels": 4}, {"channels": 256}, {"num_patches": 1},
    {"depth": 0}, {"num_heads": 0}, {"group_size": True},
    {"rope_scale": 0.0}, {"rope_base": 1.0},
    {"geometry_scale": float("nan")}, {"mlp_ratio": -1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        gp.GeometryAwarePointTransformer(**kwargs)


def test_invalid_inputs(reference_geometry):
    model = small_encoder()
    for points in (torch.zeros(0, 64, 6), torch.zeros(2, 2, 6), torch.zeros(2, 64, 2), torch.zeros(2, 64, 6, dtype=torch.int32)):
        with pytest.raises(ValueError):
            model(points)
    for value in (float("nan"), float("inf")):
        points = torch.randn(2, 64, 6)
        points[0, 0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            model(points)


def test_repository_wiring_is_single_memory():
    root = Path(__file__).resolve().parents[1]
    caller = (root / "dexmani_policy/agents/obs_encoder/interaction/encoder.py").read_text()
    reader = (root / "dexmani_policy/agents/obs_encoder/interaction/hand_scene_relation.py").read_text()
    ast.parse(caller)
    ast.parse(reader)
    assert "GeometryAwarePointTransformer" in caller
    assert 'patches["tokens"] = self.geometry_projection(patches["tokens"])' in caller
    assert "local_token" not in reader and "patch_token" not in reader
    assert reader.count("query, memory,") == 2
    assert 'patches["neighbor_indices"]' in reader


def test_native_backend_when_available():
    pytest.importorskip("pytorch3d.ops", reason="Native PyTorch3D FPS/kNN not installed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = small_encoder().to(device).eval()
    result = model(torch.randn(1, 64, 6, device=device) * 0.1)
    assert result["tokens"].shape == (1, 16, 48)
    assert torch.isfinite(result["tokens"]).all()


def test_reader_contract_uses_one_memory_and_honors_missing_geometry(reference_geometry):
    # Execute the actual reader implementation with recording attention modules.
    # This is an interface test, not a full-policy / native attention evaluation.
    import math
    from collections.abc import Mapping, Sequence

    path = Path(__file__).resolve().parents[1] / "dexmani_policy/agents/obs_encoder/interaction/hand_scene_relation.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

    class RecordingAttention(nn.Module):
        def __init__(self, channels, *args):
            super().__init__()
            self.channels = channels
            self.memory = None

        def forward(self, query, memory, relative, distance, valid):
            self.memory = memory
            b, q, _ = query.shape
            output = memory.mean(dim=1, keepdim=True).expand(b, q, self.channels)
            attention = torch.zeros(b, q, 4, memory.shape[1] + 1)
            support = valid.any(-1, keepdim=True).expand(b, q, 4)
            return output, attention, support

    class CachedHandOnly(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    ns = dict(torch=torch, nn=nn, math=math, Mapping=Mapping, Sequence=Sequence,
              GeometryCrossAttention=RecordingAttention, HandKinematicsEncoder=CachedHandOnly,
              index_points=gp.index_points)
    exec(compile(tree, str(path), "exec"), ns)
    reader = ns["HandSceneRelationEncoder"](token_channels=48)
    pointcloud = torch.randn(1, 64, 6) * 0.1
    patches = small_encoder().eval()(pointcloud, return_geometry=True)
    hand = {"hand_token": torch.randn(1, 6, 48), "hand_center": torch.randn(1, 6, 3) * 0.1,
            "wrist_rotation": torch.eye(3).unsqueeze(0)}
    result = reader(pointcloud, patches, torch.zeros(1, 9), torch.zeros(1, 5, 3),
                    hand_encoding=hand, return_intermediate=True)
    assert reader.context.memory is patches["tokens"]
    assert reader.near.memory is patches["tokens"]
    assert result["relation_update"].shape == (1, 6, 48)
    patches["mask"] = torch.zeros_like(patches["mask"])
    missing = reader(pointcloud, patches, torch.zeros(1, 9), torch.zeros(1, 5, 3),
                     hand_encoding=hand, return_intermediate=True)
    assert missing["near_update"].eq(0).all()
    assert torch.isinf(missing["min_observed_distance"]).all()
