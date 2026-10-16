"""Spatial encoding and hand-scene wiring; no dataset or robot-motion tests."""
import ast
import copy
import math
from pathlib import Path

import pytest
import torch

from dexmani_policy.agents.obs_encoder.proprio.hand_spatial_encoder import HandSpatialEncoder


@pytest.mark.parametrize("frequencies", [0, 1, 2, 4])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_shapes_gradients_and_noncontiguous_input(frequencies, offset, amp):
    torch.manual_seed(31)
    model = HandSpatialEncoder(token_dim=48, num_frequencies=frequencies, use_wrist_offset=offset)
    xyz = torch.randn(4, 3, 6).transpose(1, 2).requires_grad_()
    before = xyz.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        tokens = model(xyz)
        loss = (tokens - torch.randn_like(tokens)).square().mean()
    loss.backward()
    assert tokens.shape == (4, 6, 48) and tokens.dtype == torch.float32
    assert model.out_dim == 48 and model.out_shape == (6, 48)
    torch.testing.assert_close(xyz.detach(), before, rtol=0, atol=0)
    assert torch.isfinite(xyz.grad).all() and xyz.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if offset:
        assert model.relative_proj.weight.grad.abs().sum() > 0


def test_fourier_layout_raw_coordinates_and_physical_periods():
    model = HandSpatialEncoder(token_dim=16)
    xyz = torch.tensor([[[0.031, 0.117, -0.051]]]).expand(2, 6, 3).clone()
    captured = []
    handle = model.position_mlp.register_forward_pre_hook(lambda m, args: captured.append(args[0].clone()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(xyz)
    handle.remove()
    u = xyz.double() / model.position_scale
    expected = [u]
    for axis in range(3):
        phase = math.pi * u[..., axis:axis + 1] * torch.tensor([1., 2.], dtype=torch.float64)
        expected.extend([phase.sin(), phase.cos()])
    torch.testing.assert_close(captured[0], torch.cat(expected, -1).float(), atol=1e-6, rtol=1e-6)
    # Default wavelengths are 2*s and s, not a claim of millimetre resolution.
    assert 2 * model.position_scale == 0.5


def test_identity_and_frame_order_are_preserved():
    torch.manual_seed(7)
    model = HandSpatialEncoder(token_dim=32)
    assert model.KEYPOINTS == ("wrist", "thumb", "index", "middle", "ring", "pinky")
    tokens = model(torch.zeros(2, 6, 3))
    assert not torch.equal(tokens[:, 0], tokens[:, 1])
    xyz = torch.randn(2, 3, 6, 3)
    batched = model(xyz.reshape(6, 6, 3)).reshape(2, 3, 6, 32)
    for b in range(2):
        for t in range(3):
            torch.testing.assert_close(batched[b, t], model(xyz[b, t:t + 1])[0])


def test_zero_initialization_matches_absolute_only_and_learns():
    torch.manual_seed(9)
    model = HandSpatialEncoder(token_dim=32)
    baseline = HandSpatialEncoder(token_dim=32, use_wrist_offset=False)
    baseline.load_state_dict({k: v for k, v in model.state_dict().items() if k != "relative_proj.weight"})
    xyz = torch.randn(2, 6, 3) * 0.1
    torch.testing.assert_close(model(xyz), baseline(xyz), rtol=0, atol=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    (model(xyz) - torch.randn(2, 6, 32)).square().mean().backward()
    assert model.relative_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert model.relative_proj.weight.abs().sum() > 0


def test_offsets_use_world_axes_not_rotations_or_cross_finger_pooling():
    model = HandSpatialEncoder(token_dim=32)
    torch.nn.init.normal_(model.relative_proj.weight, std=0.1)
    captured = []
    handle = model.relative_proj.register_forward_pre_hook(lambda m, args: captured.append(args[0].clone()))
    xyz = torch.randn(2, 6, 3) * 0.1
    a = model(xyz)
    b = model(xyz + torch.tensor([0.25, -0.5, 0.75]))
    handle.remove()
    torch.testing.assert_close(captured[0], (xyz - xyz[:, :1]) / model.hand_scale)
    torch.testing.assert_close(captured[0], captured[1], atol=2e-6, rtol=1e-5)
    assert captured[0][:, 0].eq(0).all()
    assert not torch.allclose(a, b)  # Only the offset branch is translation invariant.
    moved = xyz.clone()
    moved[:, 2, 0] += 0.02
    changed = model(moved)
    torch.testing.assert_close(changed[:, [0, 1, 3, 4, 5]], a[:, [0, 1, 3, 4, 5]], rtol=0, atol=0)
    assert not torch.allclose(changed[:, 2], a[:, 2])


def test_small_offset_update_survives_bf16_residual_addition():
    torch.manual_seed(11)
    model = HandSpatialEncoder(token_dim=32)
    xyz = torch.zeros(1, 6, 3)
    xyz[:, 1:, 0] = 0.1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = model(xyz).detach()
    # Weight updates occur outside autocast, as in the training loop; a single
    # surrounding autocast context would reuse its cached pre-update weights.
    with torch.no_grad():
        model.relative_proj.weight[0, 0] = 1e-5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        after = model(xyz).detach()
    assert after.dtype == torch.float32
    assert (after[:, 1:] - before[:, 1:]).abs().max() > 1e-6
    torch.testing.assert_close(after[:, 0], before[:, 0], rtol=0, atol=0)


def test_strict_save_reload_and_parameter_budget(tmp_path):
    torch.manual_seed(13)
    model = HandSpatialEncoder().eval()
    assert sum(p.numel() for p in model.parameters()) == 28928
    path = tmp_path / "spatial.pt"
    torch.save(model.state_dict(), path)
    restored = HandSpatialEncoder().eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    xyz = torch.randn(3, 6, 3)
    torch.testing.assert_close(restored(xyz), model(xyz), rtol=0, atol=0)
    bad = dict(model.state_dict())
    bad["orientation_embed.0.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        restored.load_state_dict(bad, strict=True)


@pytest.mark.parametrize("kwargs", [
    {"token_dim": 0}, {"token_dim": 1}, {"token_dim": True}, {"hidden_dim": 0},
    {"position_scale": 0}, {"position_scale": float("nan")},
    {"hand_scale": float("inf")}, {"hand_scale": -1},
    {"num_frequencies": -1}, {"num_frequencies": True}, {"num_frequencies": 17},
    {"use_wrist_offset": "false"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        HandSpatialEncoder(**kwargs)


@pytest.mark.parametrize("shape", [(2, 5, 3), (2, 6, 4), (2, 2, 6, 3), (0, 6, 3)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError):
        HandSpatialEncoder()(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_positions_rejected(value):
    xyz = torch.zeros(1, 6, 3)
    xyz[0, 1, 0] = value
    with pytest.raises(ValueError, match="finite"):
        HandSpatialEncoder()(xyz)


def test_integer_positions_rejected():
    with pytest.raises(TypeError):
        HandSpatialEncoder()(torch.zeros(1, 6, 3, dtype=torch.int64))


def test_config_reaches_both_constructor_levels():
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "dexmani_policy/configs/interaction_flow.yaml").read_text())
    assert "edge_frame" not in config["agent"]
    spatial = config["agent"]["hand_spatial"]
    assert HandSpatialEncoder(token_dim=config["agent"]["token_dim"], **spatial).out_dim == 192
    for path in ("agents/core/interaction_flow.py", "agents/obs_encoder/interaction/encoder.py"):
        tree = ast.parse((root / "dexmani_policy" / path).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        args = {a.arg for a in init.args.args}
        assert "hand_spatial" in args and "edge_frame" not in args


@pytest.fixture
def reference_geometry(monkeypatch):
    # Test-only FPS/kNN: real GAPT/attention/MLPs run unchanged on CPU.
    from dexmani_policy.agents.obs_encoder.pointcloud import geometry_patch as gp

    @torch.no_grad()
    def fps(points, count, **kwargs):
        batch, n, _ = points.shape
        indices = torch.empty(batch, count, dtype=torch.long, device=points.device)
        minimum = torch.full((batch, n), float("inf"), device=points.device)
        farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
        for i in range(count):
            indices[:, i] = farthest
            center = gp.index_points(points, farthest[:, None])
            minimum = torch.minimum(minimum, (points - center).square().sum(-1))
            farthest = minimum.argmax(-1)
        return gp.index_points(points, indices), indices

    monkeypatch.setattr(gp, "farthest_point_sample", fps)
    monkeypatch.setattr(gp, "knn_point", lambda k, support, query:
                        torch.cdist(query, support, compute_mode="donot_use_mm_for_euclid_dist").topk(k, largest=False).indices)


def make_observation():
    torch.manual_seed(17)
    return {
        "point_cloud": torch.randn(2, 32, 6) * 0.1,
        "joint_state": torch.randn(2, 19),
        "eef_pose": torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0.]]).repeat(2, 1),
        "fingertip_xyz": torch.randn(2, 5, 3) * 0.1,
        "finger_tactile_force": torch.randn(2, 5, 3),
        "tactile_valid": torch.ones(2, 5, dtype=torch.bool),
    }


def make_observation_encoder(**kwargs):
    from dexmani_policy.agents.obs_encoder.interaction.encoder import InteractionObsEncoder
    return InteractionObsEncoder(n_obs_steps=2, token_dim=48, num_patches=8,
                                 group_size=4, patch_depth=1, num_scene_tokens=4, **kwargs).eval()


@pytest.mark.parametrize("amp", [False, True])
def test_single_encoding_world_edges_and_backward(reference_geometry, amp):
    model = make_observation_encoder()
    obs = make_observation()
    calls, memories = [], []
    handles = [model.hand_encoder.register_forward_hook(lambda m, args, out: calls.append((args[0], out))),
               model.scene.register_forward_pre_hook(lambda m, args: memories.append(args[0]))]
    handles.extend(reader.register_forward_pre_hook(lambda m, args: memories.append(args[1]))
                   for reader in (model.relation.context, model.relation.near))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cond, aux = model(obs, return_intermediate=True)
        loss = (cond - torch.randn_like(cond)).square().mean()
    for h in handles:
        h.remove()
    expected = torch.cat((obs["eef_pose"][:, None, :3], obs["fingertip_xyz"]), 1)
    assert len(calls) == 1 and not hasattr(model.relation, "hand_encoder")
    torch.testing.assert_close(calls[0][0], expected, rtol=0, atol=0)
    assert calls[0][1] is aux["hand_query"]
    assert all(m is memories[0] for m in memories) and len(memories) == 3
    torch.testing.assert_close(aux["interaction_center"], expected, rtol=0, atol=0)
    # Distances are physical metres, not normalized spatial-embedding coordinates.
    torch.testing.assert_close(aux["surface_relative"].norm(dim=-1), aux["surface_distance"])
    assert cond.shape == (1, 22, 48) and aux["frame_tokens"].shape == (2, 11, 48)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.hand_encoder.relative_proj.weight.grad.abs().sum() > 0


def test_query_late_controls_only_retrieval_query(reference_geometry):
    model = make_observation_encoder(use_tactile_valid=True)
    obs = make_observation()
    seen = []
    hook = model.relation.context.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
    _, query = model(obs, return_intermediate=True)
    model.tactile_fusion = "late"
    _, late = model(obs, return_intermediate=True)
    hook.remove()
    for key in ("hand_query", "finger_evidence", "gated_finger_evidence", "geometry_gain", "tactile_gain"):
        torch.testing.assert_close(query[key], late[key], rtol=0, atol=0)
    torch.testing.assert_close(seen[0] - seen[1], query["gated_finger_evidence"], atol=1e-6, rtol=1e-5)
    assert query["finger_evidence"][:, 0].eq(0).all()
    obs["tactile_valid"].zero_()
    q, _ = model(obs)
    model.tactile_fusion = "query"
    l, _ = model(obs)
    torch.testing.assert_close(q, l, rtol=0, atol=0)


def test_orientation_is_not_used_and_metric_anchors_are_unscaled(reference_geometry):
    model = make_observation_encoder()
    obs = make_observation()
    a, aux_a = model(obs, return_intermediate=True)
    other = copy.deepcopy(obs)
    other["eef_pose"][:, 3:] = torch.tensor([0., 1., 0., -1., 0., 0.])
    b, _ = model(other)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.hand_encoder.position_scale = 0.8
    _, aux_b = model(obs, return_intermediate=True)
    for key in ("interaction_center", "surface_distance", "surface_relative", "context_relative"):
        torch.testing.assert_close(aux_a[key], aux_b[key], rtol=0, atol=0)


def test_missing_tactile_does_not_become_false_geometry(reference_geometry):
    model = make_observation_encoder(use_tactile_valid=True)
    obs = make_observation()
    obs["tactile_valid"][:, 2] = False
    a, _ = model(obs)
    obs["finger_tactile_force"][:, 2] = float("nan")
    b, aux = model(obs, return_intermediate=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert aux["finger_evidence"][:, 3].eq(0).all()
    assert aux["tactile_gain"][:, 3].eq(0).all()
