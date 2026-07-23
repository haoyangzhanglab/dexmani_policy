#!/usr/bin/env python3
"""Verify ChunkOverlapBlender against ACT temporal ensembling formulas.

Six verification goals:
  1. Weight correctness — blend formula matches ACT's exp(-k*i) weights exactly
  2. Numerical correctness — known inputs produce known outputs
  3. NaN safety — extreme values in prev_tail do not propagate NaN
  4. Reset behavior — reset() clears state correctly
  5. Shape consistency — output shapes match input shapes
  6. Reproducibility — same inputs produce same outputs

Usage:
    conda run -n policy python dexmani_policy/common/test_temporal_ensembler.py
"""

from __future__ import annotations

import math

import torch

from dexmani_policy.common.temporal_ensembler import ChunkOverlapBlender


# ---------------------------------------------------------------------------
# Reference: ACT exact formula
# ---------------------------------------------------------------------------

def _act_ref_blend(old: torch.Tensor, new: torch.Tensor, coeff: float) -> torch.Tensor:
    """ACT formula for 2-prediction blend: (old * w0 + new * w1) / (w0 + w1)."""
    w0 = 1.0
    w1 = math.exp(-coeff)
    return (old * w0 + new * w1) / (w0 + w1)


# ---------------------------------------------------------------------------
# 1. Weight correctness — ACT's exp(-k*i) weights exactly
# ---------------------------------------------------------------------------

def test_weight_construction():
    """Pre-computed weights match ACT's exp(-coeff*k) formula."""
    for coeff in [0.0, 0.01, 0.05, 1.0, 5.0]:
        b = ChunkOverlapBlender(temporal_ensemble_coeff=coeff)
        assert b.w0 == 1.0, f"w0 should always be 1.0 (exp(-coeff*0)), got {b.w0}"
        expected_w1 = math.exp(-coeff)
        assert abs(b.w1 - expected_w1) < 1e-10, (
            f"w1 mismatch: got {b.w1}, expected {expected_w1}"
        )
        assert abs(b.wsum - (1.0 + expected_w1)) < 1e-10, (
            f"wsum mismatch: got {b.wsum}, expected {1.0 + expected_w1}"
        )
    print("PASS: weight_construction")


def test_blend_formula_exact():
    """Blended result matches ACT formula exactly for every overlapping position."""
    coeff = 0.05
    B, T, A, n_act = 2, 16, 19, 8

    pred0 = torch.randn(B, T, A)
    pred1 = torch.randn(B, T, A)

    b = ChunkOverlapBlender(temporal_ensemble_coeff=coeff)
    b.update(pred0, n_action_steps=n_act)
    ctrl1 = b.update(pred1, n_action_steps=n_act)

    overlap = T - 1 - n_act  # = 7
    for b_ in range(B):
        for i in range(overlap):
            expected = _act_ref_blend(
                pred0[b_, 1 + n_act + i], pred1[b_, 1 + i], coeff
            )
            torch.testing.assert_close(
                ctrl1[b_, i], expected,
                msg=f"Position (batch={b_}, step={i}): blend mismatch"
            )

    # Last step (no counterpart in prev_tail) passes through unchanged.
    torch.testing.assert_close(
        ctrl1[:, -1], pred1[:, 1 + n_act - 1],
        msg="Last head step should pass through unchanged"
    )

    print("PASS: blend_formula_exact")


# ---------------------------------------------------------------------------
# 2. Numerical correctness — known inputs → known outputs
# ---------------------------------------------------------------------------

def test_known_inputs_outputs():
    """Simple known inputs produce arithmetically verifiable outputs."""
    coeff = math.log(3.0)  # so exp(-coeff) = 1/3, w0=1, w1=1/3, wsum=4/3

    # Build two chunks with controlled values.
    # chunk 0: head (pos 1..8) = 100, tail (pos 9..15) = old_tail_value
    # chunk 1: head (pos 1..8) = new_head_value, tail = don't care
    old_tail_val = 100.0
    new_head_val = 200.0
    n_act = 8

    c0 = torch.full((1, 16, 1), -999.0)  # fill with garbage
    c0[:, 1:1 + n_act] = 999.0            # first head (not used for blend)
    c0[:, 1 + n_act:] = old_tail_val       # tail becomes prev_tail

    c1 = torch.full((1, 16, 1), -999.0)
    c1[:, 1:1 + n_act] = new_head_val

    b = ChunkOverlapBlender(temporal_ensemble_coeff=coeff)
    b.update(c0, n_action_steps=n_act)
    ctrl = b.update(c1, n_action_steps=n_act)

    # Blended positions (0..5, overlap=6):
    #   blended = (old_tail * 1.0 + new_head * (1/3)) / (4/3)
    #           = (100.0 + 200.0/3) / (4/3)
    #           = (100.0 + 66.666...) / 1.333...
    #           = 166.666... / 1.333... = 125.0
    expected_blended = (old_tail_val * 1.0 + new_head_val * (1.0 / 3.0)) / (4.0 / 3.0)
    # = (100 + 66.666...) * 0.75 = 125.0
    assert abs(expected_blended - 125.0) < 1e-6

    # Overlap = min(7, 8-1) = 7 → positions 0..6 are blended, position 7 passes through.
    # Positions 0..6 should all be blended = 125.0
    for i in range(7):
        torch.testing.assert_close(
            ctrl[0, i, 0],
            torch.tensor(expected_blended),
            msg=f"Known-input test: position {i} should be blended to {expected_blended}"
        )

    # Position 7 (last head step, no overlap counterpart) = new_head_value = 200.0
    torch.testing.assert_close(
        ctrl[0, 7, 0], torch.tensor(new_head_val),
        msg="Last head step should equal new_head_value"
    )

    print(f"PASS: known_inputs_outputs (blended={ctrl[0,0,0].item():.1f}, "
          f"last={ctrl[0,7,0].item():.1f})")


def test_first_chunk_identity():
    """First chunk passes through unchanged (no prev_tail)."""
    b = ChunkOverlapBlender()
    pred = torch.randn(2, 16, 19)
    ctrl = b.update(pred, n_action_steps=8)
    expected = pred[:, 1:9, :]
    torch.testing.assert_close(ctrl, expected)
    print("PASS: first_chunk_identity")


def test_coeff_effect():
    """Higher coeff gives more weight to the older prediction."""
    n_act = 8

    def make_chunk(head_val, tail_val, B=1, A=3):
        c = torch.zeros(B, 16, A)
        c[:, 1:1 + n_act] = head_val
        c[:, 1 + n_act:] = tail_val
        return c

    old_tail = torch.tensor([10.0, 0.0, 0.0])
    new_head = torch.tensor([0.0, 0.0, 0.0])

    # coeff ≈ 0: both predictions weighted ~equally → blend ≈ 5.0
    b_low = ChunkOverlapBlender(temporal_ensemble_coeff=0.0)
    b_low.update(make_chunk(old_tail, old_tail), n_action_steps=n_act)
    r_low = b_low.update(make_chunk(new_head, new_head), n_action_steps=n_act)
    assert abs(r_low[0, 0, 0] - 5.0) < 1e-4, (
        f"coeff=0 should give ~5.0, got {r_low[0, 0, 0]}"
    )

    # coeff = 5: almost all weight on old → blend ≈ 10.0
    b_high = ChunkOverlapBlender(temporal_ensemble_coeff=5.0)
    b_high.update(make_chunk(old_tail, old_tail), n_action_steps=n_act)
    r_high = b_high.update(make_chunk(new_head, new_head), n_action_steps=n_act)
    assert r_high[0, 0, 0] > 9.9, (
        f"coeff=5 should give >9.9, got {r_high[0, 0, 0]}"
    )

    print(f"PASS: coeff_effect  (coeff=0→{r_low[0,0,0]:.2f}, coeff=5→{r_high[0,0,0]:.2f})")


# ---------------------------------------------------------------------------
# 3. NaN safety — extreme values in prev_tail
# ---------------------------------------------------------------------------

def test_nan_safety_normal_inputs():
    """Normal random inputs across many trials never produce NaN."""
    b = ChunkOverlapBlender()
    n_act = 8
    for _ in range(100):
        pred = torch.randn(4, 16, 19)
        ctrl = b.update(pred, n_action_steps=n_act)
        assert not torch.isnan(ctrl).any(), "NaN with normal random inputs"
        assert not torch.isinf(ctrl).any(), "Inf with normal random inputs"
    print("PASS: nan_safety_normal_inputs (100 trials)")


def test_nan_safety_extreme_prev_tail():
    """Edge case: prev_tail contains extreme values, new_head is normal."""
    n_act = 8
    b = ChunkOverlapBlender()

    # First chunk: extreme values in the tail region (positions 9..15).
    p0 = torch.randn(2, 16, 19)  # head normal, tail will be overwritten
    for extreme in [1e7, -1e7, 1e-7, -1e-7]:
        b.reset()
        p0[:, 1 + n_act:, :] = extreme
        b.update(p0, n_action_steps=n_act)

        # Second chunk: perfectly normal values.
        p1 = torch.randn(2, 16, 19)
        ctrl = b.update(p1, n_action_steps=n_act)

        assert not torch.isnan(ctrl).any(), (
            f"NaN with prev_tail={extreme}"
        )
        assert not torch.isinf(ctrl).any(), (
            f"Inf with prev_tail={extreme}"
        )

    print("PASS: nan_safety_extreme_prev_tail")


def test_nan_safety_extreme_new_head():
    """Edge case: new_head contains extreme values, prev_tail is normal."""
    n_act = 8
    b = ChunkOverlapBlender()

    p0_normal = torch.randn(2, 16, 19)
    b.update(p0_normal, n_action_steps=n_act)

    for extreme in [1e7, -1e7, 1e-7, -1e-7]:
        b2 = ChunkOverlapBlender()
        b2.update(p0_normal, n_action_steps=n_act)
        p1 = torch.randn(2, 16, 19)
        p1[:, 1:1 + n_act, :] = extreme
        ctrl = b2.update(p1, n_action_steps=n_act)

        assert not torch.isnan(ctrl).any(), (
            f"NaN with new_head={extreme}"
        )
        assert not torch.isinf(ctrl).any(), (
            f"Inf with new_head={extreme}"
        )

    print("PASS: nan_safety_extreme_new_head")


def test_nan_safety_full_range():
    """Full spectrum of float values: large, small, zero, negative."""
    n_act = 8
    for val in [1e6, -1e6, 0.0, 1e-6, -1e-6, 1e10, -1e10]:
        b = ChunkOverlapBlender()
        p0 = torch.full((2, 16, 19), val)
        p1 = torch.full((2, 16, 19), -val)
        b.update(p0, n_action_steps=n_act)
        ctrl = b.update(p1, n_action_steps=n_act)
        assert not torch.isnan(ctrl).any(), f"NaN with val={val}"
        assert not torch.isinf(ctrl).any(), f"Inf with val={val}"

    print("PASS: nan_safety_full_range")


# ---------------------------------------------------------------------------
# 4. Reset behavior — reset() clears state
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """reset() sets _prev_tail to None."""
    b = ChunkOverlapBlender()
    assert b._prev_tail is None, "Fresh blender should have no state"
    b.update(torch.randn(1, 16, 5), n_action_steps=8)
    assert b._prev_tail is not None, "After update, prev_tail should exist"
    b.reset()
    assert b._prev_tail is None, "After reset, prev_tail should be None"
    print("PASS: reset_clears_state")


def test_reset_restores_identity():
    """After reset(), next chunk passes through as identity."""
    b = ChunkOverlapBlender()
    b.update(torch.randn(1, 16, 5), n_action_steps=8)
    b.reset()

    pred = torch.randn(1, 16, 5)
    ctrl = b.update(pred, n_action_steps=8)
    torch.testing.assert_close(ctrl, pred[:, 1:9, :])
    print("PASS: reset_restores_identity")


def test_reset_idempotent():
    """Calling reset() multiple times is safe."""
    b = ChunkOverlapBlender()
    b.reset()
    b.reset()
    assert b._prev_tail is None
    b.update(torch.randn(1, 16, 5), n_action_steps=8)
    b.reset()
    b.reset()
    assert b._prev_tail is None
    print("PASS: reset_idempotent")


# ---------------------------------------------------------------------------
# 5. Shape consistency — output shapes match input shapes
# ---------------------------------------------------------------------------

def test_shapes():
    """Output shape is always (B, n_action_steps, A) regardless of input batch."""
    for B in [1, 2, 4, 8]:
        for A in [3, 19, 21]:
            for n_act in [8]:
                b = ChunkOverlapBlender()
                pred = torch.randn(B, 16, A)
                ctrl = b.update(pred, n_action_steps=n_act)
                expected = (B, n_act, A)
                assert ctrl.shape == expected, (
                    f"Shape mismatch: got {ctrl.shape}, expected {expected}"
                )
    print("PASS: shapes (12 combos)")


def test_shape_after_blend():
    """After blending (not first chunk), output shape is still correct."""
    B, T, A, n_act = 4, 16, 19, 8
    b = ChunkOverlapBlender()
    b.update(torch.randn(B, T, A), n_action_steps=n_act)
    ctrl = b.update(torch.randn(B, T, A), n_action_steps=n_act)
    assert ctrl.shape == (B, n_act, A), (
        f"Post-blend shape mismatch: got {ctrl.shape}"
    )
    print("PASS: shape_after_blend")


# ---------------------------------------------------------------------------
# 6. Reproducibility — same inputs produce same outputs
# ---------------------------------------------------------------------------

def test_deterministic_same_session():
    """Same inputs in same session produce identical outputs."""
    torch.manual_seed(42)
    p0 = torch.randn(2, 16, 19)
    p1 = torch.randn(2, 16, 19)

    b1 = ChunkOverlapBlender()
    b1.update(p0, n_action_steps=8)
    r1 = b1.update(p1, n_action_steps=8)

    b2 = ChunkOverlapBlender()
    b2.update(p0.clone(), n_action_steps=8)
    r2 = b2.update(p1.clone(), n_action_steps=8)

    torch.testing.assert_close(r1, r2)
    print("PASS: deterministic_same_session")


def test_deterministic_multiple_steps():
    """Determinism holds across many consecutive chunks."""
    torch.manual_seed(123)
    chunks = [torch.randn(3, 16, 19) for _ in range(10)]
    n_act = 8

    results_a = []
    b = ChunkOverlapBlender()
    for c in chunks:
        results_a.append(b.update(c, n_action_steps=n_act))

    results_b = []
    b = ChunkOverlapBlender()
    for c in chunks:
        results_b.append(b.update(c.clone(), n_action_steps=n_act))

    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        torch.testing.assert_close(ra, rb, msg=f"Step {i} mismatch")
    print("PASS: deterministic_multiple_steps (10 chunks)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # 1. Weight correctness
        test_weight_construction,
        test_blend_formula_exact,
        # 2. Numerical correctness
        test_known_inputs_outputs,
        test_first_chunk_identity,
        test_coeff_effect,
        # 3. NaN safety
        test_nan_safety_normal_inputs,
        test_nan_safety_extreme_prev_tail,
        test_nan_safety_extreme_new_head,
        test_nan_safety_full_range,
        # 4. Reset behavior
        test_reset_clears_state,
        test_reset_restores_identity,
        test_reset_idempotent,
        # 5. Shape consistency
        test_shapes,
        test_shape_after_blend,
        # 6. Reproducibility
        test_deterministic_same_session,
        test_deterministic_multiple_steps,
    ]

    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    if failed:
        print(f"{len(tests) - failed}/{len(tests)} passed, {failed} FAILED")
    else:
        print(f"All {len(tests)} tests passed.")
