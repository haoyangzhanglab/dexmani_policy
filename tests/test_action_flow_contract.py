"""Contract regression tests for ActionFlow.

Covers the fail-fast gaps and inference contracts fixed in Stabilization v1:
n_obs_steps=2 only, horizon/control-slice bounds, flow/NFE parameter validation,
KV-cache forward parity, and exception-safe cache cleanup. Pure-CPU, no Zarr data
dependency, construction- and decoder-level only (the CUDA/bf16/compile self-tests
in the ``__main__`` blocks are deliberately not duplicated here).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.action_flow_flowmatch import (
    SimpleRectifiedFlowDecoder,
)
from dexmani_policy.agents.action_decoders.backbone.action_flow_dit import (
    CrossAttentionWithCache,
)
from dexmani_policy.agents.core.action_flow import ActionFlowAgent


# Minimal-but-valid dims. GeoFormer's 3D RoPE requires head_dim % 6 == 0, hence
# geo_hidden_dim=48 / geo_num_heads=8 -> head_dim=6. Everything else is free.
_MINIMAL = dict(
    horizon=16,
    n_obs_steps=2,
    n_action_steps=8,
    state_dim=19,
    pc_dim=6,
    num_points=32,
    pc_encoder_config={
        "num_patches": 8,
        "stem_channels": 32,
        "token_channels": 64,
        "patch_radii": [0.04, 0.08],
        "patch_neighbors": [16, 32],
        "use_patch_self_attn": False,
    },
    geo_hidden_dim=48,
    geo_depth=2,
    geo_num_heads=8,
    geo_ffn_hidden_dim=64,
    hidden_dim=64,
    context_dim=64,
    depth=2,
    num_heads=8,
    ffn_hidden_dim=128,
    cond_bottleneck_dim=64,
    timestep_embed_dim=32,
    step_embed_dim=32,
    state_embed_hidden_dim=64,
    solver="midpoint",
    denoise_steps=2,
)


class _DummyModel(nn.Module):
    """Minimal model exposing the KV-cache contract for decoder-level tests."""

    def __init__(self, fail: bool = False):
        super().__init__()
        self._cached_k = None
        self._cached_v = None
        self.fail = fail

    def setup_kv_cache(self, context):
        self._cached_k = context.clone()
        self._cached_v = context.clone()

    def clear_kv_cache(self):
        self._cached_k = None
        self._cached_v = None

    def forward(self, x, timestep, context, state, step_size=0.0):
        if self.fail:
            raise RuntimeError("injected sampling failure")
        return x


class ActionFlowContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent19 = ActionFlowAgent(action_dim=19, **_MINIMAL)
        cls.agent21 = ActionFlowAgent(action_dim=21, **_MINIMAL)

    # -- §2.1 fail-fast -------------------------------------------------

    def test_n_obs_steps_2_builds(self):
        self.assertEqual(self.agent19.n_obs_steps, 2)

    def test_n_obs_steps_other_fails(self):
        for bad in (1, 3):
            with self.subTest(n_obs_steps=bad):
                with self.assertRaises(ValueError):
                    ActionFlowAgent(
                        horizon=16,
                        n_obs_steps=bad,
                        n_action_steps=8,
                        action_dim=19,
                        state_dim=19,
                        pc_dim=6,
                        num_points=32,
                    )

    def test_horizon_slice_fails(self):
        with self.assertRaises(ValueError):
            ActionFlowAgent(
                horizon=5,
                n_obs_steps=2,
                n_action_steps=8,
                action_dim=19,
                state_dim=19,
                pc_dim=6,
                num_points=32,
            )

    # -- §2.4 items 2/3: state history is 38 regardless of action_key ----

    def test_state_hist_dim_is_38_for_action_and_ee(self):
        self.assertEqual(self.agent19.obs_encoder.state_hist_dim, 38)
        self.assertEqual(self.agent21.obs_encoder.state_hist_dim, 38)
        self.assertEqual(self.agent19.action_dim, 19)
        self.assertEqual(self.agent21.action_dim, 21)
        self.assertEqual(self.agent21.obs_encoder.state_dim, 19)

    # -- §2.2 flow/NFE validation --------------------------------------

    def test_flow_validation_rejects_invalid(self):
        invalid = [
            dict(solver="midpoint", denoise_steps=3),  # odd midpoint NFE
            dict(solver="euler", denoise_steps=0),
            dict(solver="euler", denoise_steps=2.5),
            dict(solver="euler", denoise_steps=True),  # bool is not an int
            dict(solver="euler", noise_shift_ratio=-0.1),
            dict(solver="euler", noise_shift_ratio=1.5),
            dict(solver="euler", noise_shift_alpha=0.0),
            dict(solver="euler", noise_shift_alpha=-1.0),
        ]
        for kw in invalid:
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    SimpleRectifiedFlowDecoder(model=nn.Module(), **kw)

    def test_flow_validation_accepts_valid(self):
        for kw in (
            dict(solver="midpoint", denoise_steps=2),
            dict(solver="midpoint", denoise_steps=4),
            dict(solver="euler", denoise_steps=3),  # euler allows odd NFE
        ):
            with self.subTest(kw=kw):
                SimpleRectifiedFlowDecoder(model=nn.Module(), **kw)

    def test_resolve_nfe_rejects_non_integer_and_odd_midpoint(self):
        dec = SimpleRectifiedFlowDecoder(
            model=_DummyModel(), solver="midpoint", denoise_steps=2
        )
        with self.assertRaises(ValueError):
            dec._resolve_nfe(2.5)
        with self.assertRaises(ValueError):
            dec._resolve_nfe(3)  # odd, midpoint
        self.assertEqual(dec._resolve_nfe(4), 4)

    # -- §2.4 item 5: KV-cache parity ----------------------------------

    def test_kv_cache_forward_parity(self):
        torch.manual_seed(0)
        ca = CrossAttentionWithCache(
            query_dim=64, context_dim=64, num_heads=8, qk_norm=True
        ).eval()
        x = torch.randn(2, 16, 64)
        ctx = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_uncached = ca(x, ctx)
            ca.setup_kv_cache(ctx)
            out_cached = ca(x, ctx)
            ca.clear_kv_cache()
        torch.testing.assert_close(out_cached, out_uncached)

    # -- §2.4 item 6: exception-safe cleanup ---------------------------

    def test_sampling_exception_clears_cache(self):
        model = _DummyModel(fail=True)
        dec = SimpleRectifiedFlowDecoder(model=model, solver="euler", denoise_steps=2)
        cond = {"memory": torch.randn(2, 8, 64), "state": torch.randn(2, 38)}
        template = torch.zeros(2, 16, 19)
        with self.assertRaises(RuntimeError):
            dec.predict_action(cond, template)
        self.assertIsNone(model._cached_k)
        self.assertIsNone(model._cached_v)


if __name__ == "__main__":
    unittest.main()
