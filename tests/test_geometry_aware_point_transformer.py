"""GAPT module checks. CPU reference FPS/kNN do not validate native kernels."""
import math

import pytest
import torch

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


def test_eval_determinism_and_strict_reload(reference_geometry, tmp_path):
    model = small_encoder().eval()
    p = torch.randn(2, 64, 6) * 0.12
    a = model(p)
    torch.manual_seed(100)
    b = model(p)
    restored = small_encoder().eval()
    path = tmp_path / "gapt.pt"
    torch.save(model.state_dict(), path)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
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


@pytest.mark.parametrize("amp", [False, True])
def test_metric_fourier_gaussian_and_centered_covariance(reference_geometry, amp):
    stem = gp.GeometricPatchEmbedding(6, 48, 0.1).eval()
    # Asymmetric support with a nonzero mean detects uncentered second moments.
    rel = torch.tensor([[[[0., 0., 0.], [0.02, 0.01, 0.], [0.04, -0.01, 0.03], [0.12, 0.02, -0.01]]]])
    rgb = torch.rand_like(rel)
    captured = {}

    def record_input(key):
        def hook(module, args):
            captured[key] = args[0].detach()
        return hook

    handles = [stem.member_mlp.register_forward_pre_hook(record_input("features")),
               stem.projection.register_forward_pre_hook(record_input("pooled")),
               stem.refine.register_forward_hook(lambda m, args, out: captured.update(member=out.detach()))]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        _, radius = stem(rel, rgb)
    for handle in handles:
        handle.remove()
    wavelengths = torch.logspace(math.log10(0.02), math.log10(0.32), 8, dtype=torch.float64)
    torch.testing.assert_close(stem.wavelengths.double(), wavelengths)
    phases = rel.double().unsqueeze(-1) * (2 * math.pi / wavelengths)
    expected_fourier = torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(-2)
    features = captured["features"]
    assert features.dtype == torch.float32
    torch.testing.assert_close(features[..., :3], rel / 0.1)
    torch.testing.assert_close(features[..., 3:4], rel.norm(dim=-1, keepdim=True) / 0.1)
    torch.testing.assert_close(features[..., 4:52], expected_fourier.float(), atol=5e-6, rtol=1e-5)
    torch.testing.assert_close(features[..., 52:], rgb)
    pooled = captured["pooled"]
    member = captured["member"]
    torch.testing.assert_close(pooled[..., :128], member.amax(2))
    # Direct normalized Gaussian and double accumulation provide an independent oracle.
    distance = rel.double().norm(dim=-1)
    for pool, sigma in enumerate((0.04, 0.12)):
        weights = torch.exp(-distance.square() / (2 * sigma ** 2))
        weights = weights / weights.sum(-1, keepdim=True)
        expected = (weights[..., None] * member.double()).sum(2)
        torch.testing.assert_close(pooled[..., 128 * (pool + 1):128 * (pool + 2)], expected.to(pooled.dtype))
    covariance = torch.cov((rel[0, 0].double() / 0.1).T, correction=0)
    triangle = torch.triu_indices(3, 3)
    torch.testing.assert_close(pooled[0, 0, -7:-1], covariance[triangle[0], triangle[1]].to(pooled.dtype))
    torch.testing.assert_close(pooled[..., -1:], (radius / 0.1).to(pooled.dtype))


def test_axial_rope_metric_phase_and_qk_norm_order(reference_geometry, monkeypatch):
    rope = gp.RotaryPositionEncoding3D(48)
    xyz = torch.tensor([[[0., 0., 0.], [0.05, 0., 0.]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cos, sin = rope(xyz)
    assert cos.dtype == sin.dtype == torch.float32
    # A 5 cm X displacement is one radian in the highest-frequency X pair.
    torch.testing.assert_close(cos[0, 0, :, :2], torch.full((2, 2), math.cos(0.5)))
    torch.testing.assert_close(sin[0, 0, :, :2], torch.tensor([[-math.sin(0.5)] * 2, [math.sin(0.5)] * 2]))
    assert cos[..., 16:].eq(1).all() and sin[..., 16:].eq(0).all()
    x = torch.randn(1, 8, 2, 48)
    rotated = rope.apply_rotary(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    shifted_cos, shifted_sin = rope(xyz + torch.tensor([0.2, -0.1, 0.3]))
    torch.testing.assert_close(shifted_cos, cos)
    torch.testing.assert_close(shifted_sin, sin)

    block = gp.PointRelationBlock(384, 8, 4)
    features = torch.randn(1, 2, 384)
    q, k, _ = block.qkv(block.norm1(features)).reshape(1, 2, 3, 8, 48).permute(2, 0, 3, 1, 4).unbind(0)
    expected_q = rope.apply_rotary(block.q_norm(q), cos, sin)
    expected_k = rope.apply_rotary(block.k_norm(k), cos, sin)
    sdpa = gp.F.scaled_dot_product_attention

    def checked_sdpa(q, k, v, **kwargs):
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(k, expected_k)
        return sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(gp.F, "scaled_dot_product_attention", checked_sdpa)
    assert torch.isfinite(block(features, cos, sin)).all()


def test_forward_fixed_anchor_permutation_and_single_geometry_graph(reference_geometry, monkeypatch):
    model = small_encoder().eval()
    points = torch.randn(2, 64, 6) * 0.1
    indices = torch.arange(16).expand(2, -1)
    calls = {"fps": 0, "knn": 0, "graph": 0}
    cdist = torch.cdist

    def fixed_fps(xyz, count, **kwargs):
        calls["fps"] += 1
        return gp.index_points(xyz, indices), indices

    def counted_knn(k, support, query):
        calls["knn"] += 1
        return cdist(query, support, compute_mode="donot_use_mm_for_euclid_dist").topk(k, largest=False).indices

    def counted_graph(*args, **kwargs):
        calls["graph"] += 1
        return cdist(*args, **kwargs)

    monkeypatch.setattr(gp, "farthest_point_sample", fixed_fps)
    monkeypatch.setattr(gp, "knn_point", counted_knn)
    monkeypatch.setattr(torch, "cdist", counted_graph)
    a = model(points, return_geometry=True)
    assert calls == {"fps": 1, "knn": 1, "graph": 1}
    perm = torch.randperm(16)
    indices = indices[:, perm]
    b = model(points, return_geometry=True)
    for key in a:
        torch.testing.assert_close(a[key][:, perm], b[key], atol=3e-6, rtol=2e-5)


def test_coincident_centers_exclude_only_self_index(reference_geometry):
    model = small_encoder().eval()

    def check_edges(module, args):
        _, indices, edges = args
        assert indices.shape == (2, 16, 12)
        assert (indices != torch.arange(16)[None, :, None]).all()
        assert edges.eq(0).all()

    handle = model.local_relation.register_forward_pre_hook(check_edges)
    assert torch.isfinite(model(torch.zeros(2, 64, 6))["tokens"]).all()
    handle.remove()


@pytest.mark.parametrize("noise", [0., 0.002])
def test_native_random_fps_alignment_with_duplicates(noise):
    pytest.importorskip("pytorch3d.ops")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = torch.randn(2, 64, 3, device=device) * 0.1
    points[0] = points[0, :1].clone()
    points[1, 32:] = points[1, :32].clone()
    centers, indices = gp.farthest_point_sample(points, 16, random_noise_scale=noise)
    assert indices.min() >= 0 and indices.max() < 64
    torch.testing.assert_close(centers, gp.index_points(points, indices), rtol=0, atol=0)


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


def test_repository_wiring_is_single_memory(reference_geometry):
    from dexmani_policy.agents.obs_encoder.interaction.encoder import InteractionObsEncoder

    model = InteractionObsEncoder(n_obs_steps=2).eval()
    memories = []
    handles = [model.scene.register_forward_pre_hook(lambda m, args: memories.append(args[0]))]
    for reader in (model.relation.context, model.relation.near):
        handles.append(reader.register_forward_pre_hook(lambda m, args: memories.append(args[1])))
    obs = {
        "point_cloud": torch.randn(2, 256, 6) * 0.1,
        "joint_state": torch.randn(2, 19),
        "eef_pose": torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0.]]).expand(2, -1),
        "fingertip_xyz": torch.randn(2, 5, 3) * 0.1,
        "finger_tactile_force": torch.randn(2, 5, 3),
    }
    cond, aux = model(obs, return_intermediate=True)
    for handle in handles:
        handle.remove()
    assert isinstance(model.geometry, gp.GeometryAwarePointTransformer)
    assert len(memories) == 3 and all(memory is memories[0] for memory in memories)
    assert memories[0].shape == (2, 256, 192)
    assert cond.shape == (1, 46, 192) and aux["frame_tokens"].shape == (2, 23, 192)
    (cond - torch.randn_like(cond)).square().mean().backward()
    for module in (model.geometry, model.geometry_projection, model.scene, model.relation):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("input_channels", [3, 6])
def test_native_backend_when_available(device, amp, input_channels):
    pytest.importorskip("pytorch3d.ops", reason="Native PyTorch3D FPS/kNN not installed")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable; CPU checks do not validate CUDA")
    model = gp.GeometryAwarePointTransformer(input_channels=input_channels).to(device).train()
    points = torch.randn(1, 1024, input_channels, device=device) * 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.patch_embedding.member_mlp[0].weight.detach().clone()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        result = model(points, return_geometry=True)
        loss = (result["tokens"].float() - torch.randn_like(result["tokens"], dtype=torch.float32)).square().mean()
    assert result["tokens"].shape == (1, 256, 384)
    assert torch.isfinite(result["tokens"]).all()
    torch.testing.assert_close(result["xyz"], gp.index_points(points[..., :3], result["center_indices"]), rtol=0, atol=0)
    neighbors = result["neighbor_indices"]
    assert neighbors.min() >= 0 and neighbors.max() < points.shape[1]
    actual_distances = (gp.index_points(points[..., :3], neighbors) - result["xyz"][:, :, None]).norm(dim=-1)
    expected_distances = torch.cdist(result["xyz"], points[..., :3].contiguous(), compute_mode="donot_use_mm_for_euclid_dist").topk(32, largest=False).values
    torch.testing.assert_close(actual_distances.sort(-1).values, expected_distances)
    torch.testing.assert_close(result["patch_radius"], actual_distances.amax(-1, keepdim=True))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.patch_embedding.member_mlp[0].weight)


def test_reader_contract_uses_one_memory_and_honors_missing_geometry(reference_geometry):
    from dexmani_policy.agents.obs_encoder.interaction.hand_scene_relation import HandSceneRelationEncoder

    reader = HandSceneRelationEncoder(token_channels=48)
    # Nonzero learned null/out bias must still produce exactly zero near update
    # if no visible member satisfies a head's radius constraint.
    with torch.no_grad():
        reader.near.null_value.fill_(0.5)
        reader.near.out.bias.fill_(0.7)
    memories = []
    handles = [module.register_forward_pre_hook(lambda m, args: memories.append(args[1]))
               for module in (reader.context, reader.near)]
    pointcloud = torch.randn(1, 64, 6) * 0.1
    patches = small_encoder().eval()(pointcloud, return_geometry=True)
    wrist = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0.]])
    fingers = pointcloud[:, :5, :3].clone()
    result = reader(pointcloud, patches, wrist, fingers, return_intermediate=True)
    for handle in handles:
        handle.remove()
    assert len(memories) == 2 and all(memory is patches["tokens"] for memory in memories)
    assert result["relation_update"].shape == (1, 6, 48)
    anchors = torch.cat((wrist[:, None, :3], fingers), dim=1)
    members = gp.index_points(pointcloud[..., :3], patches["neighbor_indices"])
    expected = (members[:, None] - anchors[:, :, None, None]).norm(dim=-1).amin(-1)
    torch.testing.assert_close(result["surface_distance"], expected)
    allowed = expected[:, :, None] <= 3 * reader.near.sigmas[None, None, :, None]
    assert torch.equal(result["near_has_support"], allowed.any(-1))
    assert result["near_attention"][..., :-1][~allowed].eq(0).all()
    assert result["near_has_support"].any()

    far_wrist = wrist.clone()
    far_wrist[:, :3] += 10
    far = reader(pointcloud, patches, far_wrist, fingers + 10, return_intermediate=True)
    assert not far["near_has_support"].any() and far["near_update"].eq(0).all()
    missing_members = reader(pointcloud, patches, wrist, fingers,
                             member_valid=torch.zeros_like(patches["neighbor_indices"], dtype=torch.bool),
                             return_intermediate=True)
    assert missing_members["near_update"].eq(0).all()
    assert torch.isinf(missing_members["min_observed_distance"]).all()
    patches["mask"] = torch.zeros_like(patches["mask"])
    missing = reader(pointcloud, patches, wrist, fingers, return_intermediate=True)
    assert missing["near_update"].eq(0).all()
    assert torch.isinf(missing["min_observed_distance"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compiled policy integration")
def test_cuda_policy_compiled_training_and_strict_reload(tmp_path):
    """Synthetic observations exercise the real policy; this is not a dataset smoke."""
    pytest.importorskip("pytorch3d.ops")
    import hydra
    from dexmani_policy.agents.normalization import LinearNormalizer
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import (
        build_model_and_ema, build_optimizer_and_scheduler, compile_models,
    )
    from dexmani_policy.training.checkpoint import fix_state_dict

    torch.manual_seed(53)
    cfg = load_config("interaction_flow")
    # Only the synthetic fixture's sample count is small; model, normalization
    # modes, optimizer, warmup, action layout and compile path come from config.
    obs = {
        "point_cloud": torch.randn(2, 2, 1024, 6) * 0.1,
        "joint_state": torch.randn(2, 2, 19),
        "eef_pose": torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.]).expand(2, 2, -1),
        "fingertip_xyz": torch.randn(2, 2, 5, 3) * 0.1,
        "finger_tactile_force": torch.randn(2, 2, 5, 3),
    }
    actions = torch.randn(2, 16, 19)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions, "joint_state": obs["joint_state"]}, mode="limits")
    normalizer.fit_field_chunks("finger_tactile_force", [obs["finger_tactile_force"]], mode="gaussian")
    model, ema, updater = build_model_and_ema(cfg, "cuda", normalizer)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=False)
    geometry_keys = tuple(model.obs_encoder.geometry.state_dict())
    compile_models(model, ema, mode="reduce-overhead")
    assert hasattr(model.action_decoder.model, "_orig_mod")
    assert not hasattr(model.obs_encoder.geometry, "_orig_mod")
    for selected in (model, ema):
        geometry = selected.obs_encoder.geometry
        assert tuple(geometry.state_dict()) == geometry_keys
        assert geometry._global_forward.__wrapped__.__self__ is geometry
    obs = {key: value.cuda() for key, value in obs.items()}
    batch = {"obs": obs, "action": actions.cuda()}
    weight = model.obs_encoder.geometry.patch_embedding.member_mlp[0].weight
    before = weight.detach().clone()
    # Zero-initialized DiT output/gates and the unchanged 500-step warmup delay
    # observable FP32 stem updates. Keep the real recipe and allow 12 updates.
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        updater.step(model)
    assert not torch.equal(before, weight)

    for name, selected in (("raw", model), ("ema", ema)):
        selected.eval()
        torch.manual_seed(71)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = selected.predict_action(obs)
        assert prediction["control_action"].shape == (2, 8, 19)
        assert torch.isfinite(prediction["pred_action"]).all()
        selected.action_decoder.model = selected.action_decoder.model._orig_mod
        geometry = selected.obs_encoder.geometry
        geometry._global_forward = geometry._global_forward.__wrapped__
        torch.manual_seed(71)
        expected = selected.predict_action(obs)["pred_action"]
        path = tmp_path / f"{name}.pt"
        torch.save(fix_state_dict(selected.state_dict()), path)
        restored = hydra.utils.instantiate(cfg.agent).cuda().eval()
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        torch.manual_seed(71)
        actual = restored.predict_action(obs)["pred_action"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del restored


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for BF16 Inductor regression")
def test_cuda_compiled_patch_pooling_backward():
    # Regression for Torch 2.4 promoting autocast cat to FP32 while retaining
    # BF16 projection weights. Compile the actual module without a fallback.
    torch.manual_seed(79)
    model = gp.GeometricPatchEmbedding(6, 384, 0.1).cuda()
    relative = torch.randn(1, 256, 32, 3, device="cuda") * 0.05
    attributes = torch.rand_like(relative)
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, expected_radius = model(relative, attributes)
        actual, radius = compiled(relative, attributes)
        loss = (actual.float() - torch.randn_like(actual, dtype=torch.float32)).square().mean()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(radius, expected_radius)
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for global compile comparison")
@pytest.mark.parametrize("amp", [False, True])
def test_cuda_global_compile_preserves_outputs_gradients_and_state(amp):
    pytest.importorskip("pytorch3d.ops")
    torch.manual_seed(17)
    model = gp.GeometryAwarePointTransformer().cuda().eval()
    points = (torch.randn(1, 1024, 6, device="cuda") * 0.1).requires_grad_()
    target = torch.randn(1, 256, 384, device="cuda")
    keys = tuple(model.state_dict())
    parameter_ids = [id(p) for p in model.parameters()]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        expected = model(points, return_geometry=True)
        (expected["tokens"].float() - target).square().mean().backward()
    expected_grad = {name: p.grad.clone() for name, p in model.named_parameters()}
    expected_input_grad = points.grad.clone()
    model.zero_grad(set_to_none=True)
    points.grad = None

    eager_global = model._global_forward
    compiled_global = torch.compile(eager_global, mode="reduce-overhead", fullgraph=True)
    model._global_forward = compiled_global
    assert tuple(model.state_dict()) == keys
    assert [id(p) for p in model.parameters()] == parameter_ids
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        actual = model(points, return_geometry=True)
        (actual["tokens"].float() - target).square().mean().backward()
    for key in expected:
        tolerance = 0.025 if amp else 2e-5
        torch.testing.assert_close(actual[key], expected[key],
                                   atol=tolerance if key == "tokens" else 0,
                                   rtol=tolerance if key == "tokens" else 0)
    # Compare the complete gradient, including the eager geometric stages.
    # Relative L2 avoids dividing by individual near-zero gradient elements.
    for prefix in ("patch_embedding.", "local_relation.", "blocks.", "norm."):
        reference = torch.cat([g.flatten() for name, g in expected_grad.items() if name.startswith(prefix)])
        gradient = torch.cat([p.grad.flatten() for name, p in model.named_parameters() if name.startswith(prefix)])
        assert torch.isfinite(gradient).all() and reference.norm() > 0
        assert (gradient - reference).norm() / reference.norm() < (0.025 if amp else 2e-4)
    assert torch.isfinite(points.grad).all() and expected_input_grad.norm() > 0
    assert (points.grad - expected_input_grad).norm() / expected_input_grad.norm() < (0.025 if amp else 2e-4)

    # Check more geometries with the original BF16 output tolerance unchanged.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for seed in (37, 53):
            torch.manual_seed(seed)
            sample = torch.randn(1, 1024, 6, device="cuda") * 0.1
            actual = model(sample)["tokens"].clone()
            model._global_forward = eager_global
            expected = model(sample)["tokens"]
            model._global_forward = compiled_global
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    invalid = points.detach().clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)
