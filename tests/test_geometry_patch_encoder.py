"""CPU unit tests; FPS/kNN are explicit reference substitutes, not CUDA coverage."""
import pytest
import torch
from torch import nn

from dexmani_policy.agents.obs_encoder.pointcloud import geometry_patch as gp


@torch.no_grad()
def reference_fps(points, num_samples, **kwargs):
    batch, count, _ = points.shape
    selected = torch.empty(batch, num_samples, device=points.device, dtype=torch.long)
    distance = torch.full((batch, count), float("inf"), device=points.device)
    farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
    if kwargs.get("use_random", True) and kwargs.get("use_random_start", True):
        farthest = torch.randint(count, (batch,), device=points.device)
    for i in range(num_samples):
        selected[:, i] = farthest
        center = gp.index_points(points, farthest[:, None])
        distance = torch.minimum(distance, (points - center).square().sum(-1))
        farthest = distance.argmax(-1)
    if kwargs.get("use_random", True) and kwargs.get("use_shuffle_output", True):
        perm = torch.rand(batch, num_samples, device=points.device).argsort(-1)
        selected = selected.gather(1, perm)
    return gp.index_points(points, selected), selected


def reference_knn(k, support, query):
    return torch.cdist(query, support).topk(k, largest=False, sorted=False).indices


@pytest.fixture
def geometry_ops(monkeypatch):
    monkeypatch.setattr(gp, "farthest_point_sample", reference_fps)
    monkeypatch.setattr(gp, "knn_point", reference_knn)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def make_encoder(kind="compact", local=0, channels=6, **kwargs):
    return gp.GeometryPatchEncoder(
        input_channels=channels, token_channels=48, num_patches=8,
        group_size=8, depth=2, num_heads=4,
        fps_random_config={"use_random": False},
        patch_encoder_type=kind, local_mixer_neighbors=local, **kwargs,
    )


@pytest.mark.parametrize("kind", ["uni3d", "compact"])
@pytest.mark.parametrize("local", [0, 3])
@pytest.mark.parametrize("channels", [3, 6])
def test_shapes_membership_and_backward(geometry_ops, kind, local, channels):
    encoder = make_encoder(kind, local, channels)
    points = (torch.randn(2, 64, channels) * 0.2).requires_grad_()
    output = encoder(points, return_intermediate=True)
    assert output["patch_token"].shape == (2, 8, 48)
    assert encoder.out_shape == (8, 48) and encoder.out_dim == 48
    assert output["patch_center"].dtype == torch.float32
    members = gp.index_points(points[..., :channels], output["neighbor_idx"])
    patch_input = members.clone()
    patch_input[..., :3] -= output["patch_center"].unsqueeze(2)
    # The near-field reader must still receive pre-mixing raw-patch features.
    torch.testing.assert_close(output["local_token"], encoder.patch_encoder(patch_input))
    torch.testing.assert_close(
        output["patch_center"], gp.index_points(points[..., :3], output["patch_center_idx"])
    )
    target = torch.randn_like(output["patch_token"])
    (output["patch_token"] - target).square().mean().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.isfinite(points.grad).all()


def test_default_preserves_original_path_and_keys(geometry_ops):
    encoder = make_encoder("uni3d").eval()
    points = torch.randn(2, 64, 6) * 0.2
    output = encoder(points, return_intermediate=True)
    assert isinstance(encoder.patch_encoder, gp.PatchEncoder)
    assert encoder.local_mixer is None
    assert not any(key.startswith("local_mixer.") for key in encoder.state_dict())
    # Original path: Uni3D patch features -> same RoPE blocks -> final LN.
    expected = output["local_token"]
    cos, sin = encoder.rope(output["patch_center"])
    for block in encoder.blocks:
        expected = block(expected, cos, sin)
    expected = encoder.norm(expected)
    torch.testing.assert_close(output["patch_token"], expected, rtol=0, atol=0)
    assert set(encoder(points)) == {"patch_token", "patch_center"}


@pytest.mark.parametrize("kind", ["uni3d", "compact"])
def test_reload_and_eval_determinism(geometry_ops, kind):
    encoder = make_encoder(kind, 3).eval()
    points = torch.randn(2, 64, 6) * 0.2
    saved = encoder.state_dict()
    restored = make_encoder(kind, 3).eval()
    restored.load_state_dict(saved, strict=True)
    first = encoder(points)["patch_token"]
    torch.manual_seed(99)
    second = encoder(points)["patch_token"]
    third = restored(points)["patch_token"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, third, rtol=0, atol=0)


def test_local_mixer_preserves_raw_local_features(geometry_ops):
    plain = make_encoder(local=0).eval()
    mixed = make_encoder(local=3).eval()
    shared = {k: v for k, v in plain.state_dict().items()}
    missing, unexpected = mixed.load_state_dict(shared, strict=False)
    assert missing and all(k.startswith("local_mixer.") for k in missing)
    assert not unexpected
    points = torch.randn(2, 64, 6) * 0.2
    a, b = plain(points, True), mixed(points, True)
    for key in ("patch_center", "neighbor_idx", "patch_center_idx", "local_token"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert not torch.equal(a["patch_token"], b["patch_token"])


def test_patch_members_are_a_set(geometry_ops):
    patch = gp.CompactPatchEncoder(6, 48).eval()
    points = torch.randn(2, 8, 8, 6)
    perm = torch.randperm(8)
    torch.testing.assert_close(patch(points), patch(points[:, :, perm]), atol=2e-6, rtol=1e-5)


def test_fixed_anchor_permutation_equivariance(geometry_ops):
    # Fixed anchors, not an assertion about first-point FPS under raw permutations.
    encoder = make_encoder(local=3).eval()
    features = torch.randn(2, 8, 48)
    xyz = torch.randn(2, 8, 3) * 0.2
    perm = torch.randperm(8)

    def contextualize(x, c):
        x = encoder.local_mixer(x, c)
        cos, sin = encoder.rope(c)
        for block in encoder.blocks:
            x = block(x, cos, sin)
        return encoder.norm(x)

    expected = contextualize(features, xyz)[:, perm]
    actual = contextualize(features[:, perm], xyz[:, perm])
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)


def test_relative_geometry_translation_and_interpatch_interaction(geometry_ops):
    encoder = make_encoder(local=3).eval()
    features = torch.randn(2, 8, 48)
    xyz = torch.randn(2, 8, 3) * 0.2
    mixer = encoder.local_mixer
    a = mixer(features, xyz)
    b = mixer(features, xyz + torch.tensor([0.3, -0.1, 0.2]))
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    cos, sin = encoder.rope(xyz)
    cos2, sin2 = encoder.rope(xyz + 0.3)
    block = encoder.blocks[0]
    torch.testing.assert_close(block(a, cos, sin), block(a, cos2, sin2), atol=2e-6, rtol=1e-5)
    changed = a.clone()
    changed[:, -1] += torch.linspace(-3, 3, 48)
    assert not torch.allclose(block(a, cos, sin)[:, 0], block(changed, cos, sin)[:, 0])


def test_duplicate_centers_are_finite(geometry_ops):
    mixer = gp.LocalPatchMixer(48, 3)
    features = torch.randn(2, 8, 48, requires_grad=True)
    output = mixer(features, torch.zeros(2, 8, 3))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("kind", ["uni3d", "compact"])
def test_cpu_autocast(geometry_ops, kind):
    encoder = make_encoder(kind, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder(torch.randn(2, 64, 6) * 0.2)["patch_token"]
        loss = (output.float() - torch.randn_like(output.float())).square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_semantic_residual_stays_aligned(geometry_ops):
    encoder = make_encoder(local=3, semantic_channels=12).eval()
    points = torch.randn(2, 64, 6) * 0.2
    images = torch.randn(2, 1, 4, 12, requires_grad=True)
    kwargs = dict(image_tokens=images, point_uv=torch.rand(2, 1, 64, 2) * 7,
                  view_weight=torch.ones(2, 1, 64), image_hw=(8, 8), patch_grid_size=(2, 2))
    plain = encoder(points, True)
    semantic = encoder(points, True, **kwargs)
    torch.testing.assert_close(plain["local_token"], semantic["local_token"], rtol=0, atol=0)
    torch.testing.assert_close(plain["neighbor_idx"], semantic["neighbor_idx"])
    semantic["patch_token"].mul(torch.randn_like(semantic["patch_token"])).mean().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert encoder.semantic_fusion.proj.weight.grad.abs().sum() > 0
    kwargs["view_weight"] = torch.zeros_like(kwargs["view_weight"])
    invisible = encoder(points, True, **kwargs)
    torch.testing.assert_close(plain["patch_token"], invisible["patch_token"], rtol=0, atol=0)
    torch.testing.assert_close(invisible["fused_local_token"], invisible["local_token"], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"patch_encoder_type": "other"}, {"local_mixer_neighbors": -1},
    {"local_mixer_neighbors": 8}, {"local_mixer_neighbors": 1.5},
    {"local_mixer_neighbors": True},
    {"local_mixer_scale": 0}, {"local_mixer_scale": float("nan")},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        gp.GeometryPatchEncoder(num_patches=8, **kwargs)


def test_invalid_inputs(geometry_ops):
    encoder = make_encoder()
    with pytest.raises(ValueError, match="at least"):
        encoder(torch.randn(2, 4, 6))
    points = torch.randn(2, 64, 6)
    points[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        encoder(points)
    with pytest.raises(ValueError, match="semantic"):
        encoder(torch.randn(2, 64, 6), image_tokens=torch.randn(2, 1, 4, 12))


def linear_macs(module, value):
    result = [0]
    def hook(layer, args, output):
        result[0] += output.numel() * layer.in_features
    hooks = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, nn.Linear)]
    try:
        with torch.no_grad():
            module(value)
    finally:
        for handle in hooks:
            handle.remove()
    return result[0]


def test_compact_stem_linear_cost(geometry_ops):
    points = torch.randn(1, 128, 32, 6)
    baseline = gp.PatchEncoder(6, 192, [128, 256])
    compact = gp.CompactPatchEncoder(6, 192)
    assert linear_macs(compact, points) < 0.2 * linear_macs(baseline, points)


def test_real_pytorch3d_backend_when_installed():
    pytest.importorskip("pytorch3d.ops", reason="native FPS/kNN backend not installed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = make_encoder(local=3).to(device).eval()
    output = encoder(torch.randn(2, 64, 6, device=device) * 0.2)
    assert output["patch_token"].shape == (2, 8, 48)
    assert torch.isfinite(output["patch_token"]).all()
