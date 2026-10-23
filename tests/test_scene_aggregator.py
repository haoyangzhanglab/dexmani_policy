"""Regression checks for routing-prior scene refinement."""

import pytest
import torch

from dexmani_policy.agents.obs_encoder.interaction.scene_aggregator import (
    SceneAggregator,
)


def make_scene(masked=False):
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randn(2, 32, 32, generator=generator)
    centers = torch.randn(2, 32, 3, generator=generator) * 0.1
    valid = torch.ones(2, 32, dtype=torch.bool)
    if masked:
        valid[:, -4:] = False
    return tokens, centers, valid


def make_aggregator(**kwargs):
    config = dict(
        token_channels=32,
        num_scene_tokens=6,
        route_dim=24,
        relation_dim=32,
        refinement_dim=32,
    )
    return SceneAggregator(**(config | kwargs))


def test_default_behavior_is_unchanged_when_refinement_is_disabled():
    tokens, centers, valid = make_scene()
    baseline = make_aggregator().eval()
    upgraded = make_aggregator(use_cross_refinement=True).eval()
    missing, unexpected = upgraded.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        *[f"cross_refinement.{name}" for name in upgraded.cross_refinement.state_dict()],
        "refinement_gain",
    }

    with torch.no_grad():
        upgraded.refinement_gain.zero_()
        original = baseline(tokens, centers, patch_mask=valid)
        refined = upgraded(tokens, centers, patch_mask=valid)
    for key, value in original.items():
        torch.testing.assert_close(value, refined[key])


def test_refinement_gradients_and_invalid_patch_mask():
    tokens, centers, valid = make_scene(masked=True)
    tokens.requires_grad_()
    model = make_aggregator(use_cross_refinement=True)
    result = model(tokens, centers, patch_mask=valid, return_intermediate=True)
    assert result["scene_token"].shape == (2, 6, 32)
    assert result["refinement_update"].shape == (2, 6, 32)
    result["scene_token"].square().mean().backward()
    assert torch.isfinite(model.cross_refinement.q.weight.grad).all()
    assert torch.isfinite(model.route_projection.weight.grad).all()
    assert torch.isfinite(tokens.grad).all()

    corrupted_tokens, corrupted_centers = tokens.detach().clone(), centers.clone()
    corrupted_tokens[~valid] = float("nan")
    corrupted_centers[~valid] = float("inf")
    with torch.no_grad():
        original = model(tokens.detach(), centers, patch_mask=valid)
        corrupted = model(corrupted_tokens, corrupted_centers, patch_mask=valid)
    for key, value in original.items():
        torch.testing.assert_close(value, corrupted[key])


def test_refinement_is_invariant_to_input_permutation():
    tokens, centers, valid = make_scene()
    permutation = torch.randperm(tokens.shape[1], generator=torch.Generator().manual_seed(4))
    model = make_aggregator(use_cross_refinement=True).eval()
    with torch.no_grad():
        original = model(tokens, centers, patch_mask=valid)
        permuted = model(tokens[:, permutation], centers[:, permutation], patch_mask=valid[:, permutation])
    for key, value in original.items():
        torch.testing.assert_close(value, permuted[key], atol=1e-5, rtol=1e-5)


def test_refinement_rejects_incompatible_heads():
    with pytest.raises(ValueError, match="divisible"):
        make_aggregator(use_cross_refinement=True, refinement_dim=31)
