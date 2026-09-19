"""CUDA dtype checks; opt into actual Inductor runs with FLOW_TEST_COMPILE=1."""

import copy
import os

import pytest
import torch

from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
from dexmani_policy.common.pytorch_util import compile_models, fix_state_dict
from dexmani_policy.training.build_utils import build_model_and_ema
from test_ditx_rms import _make_model
from test_flow_agents import normalizer_for, sample_batch, small_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
compile_only = pytest.mark.skipif(
    os.environ.get("FLOW_TEST_COMPILE") != "1",
    reason="set FLOW_TEST_COMPILE=1 for Inductor validation",
)


def test_ditx_rms_cuda_bf16_qk_norm():
    model = _make_model().cuda()
    # Nonzero output and open residual gates expose dtype/mask issues upstream.
    with torch.no_grad():
        model.final_proj.weight.normal_(std=0.1)
        for block in model.blocks:
            block.modulation[-1].bias.fill_(0.2)
    for length in (5, 9):
        model.zero_grad(set_to_none=True)
        x, context = torch.randn(2, 4, 3, device="cuda"), torch.randn(
            2, length, 6, device="cuda"
        )
        mask = torch.rand(2, length, device="cuda") > 0.3
        mask[1] = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, 0.5, context, mask)
            assert out.dtype == torch.bfloat16
            loss = (out.float() - torch.ones_like(out)).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )


@compile_only
def test_ditx_rms_inductor_forward_backward_and_euler():
    torch.manual_seed(17)
    eager = _make_model().cuda()
    with torch.no_grad():
        eager.final_proj.weight.normal_(std=0.1)
        for block in eager.blocks:
            block.modulation[-1].bias.fill_(0.2)
    compiled = torch.compile(copy.deepcopy(eager))
    for length in (5, 9):
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        x, context = torch.randn(2, 4, 3, device="cuda"), torch.randn(
            2, length, 6, device="cuda"
        )
        t = torch.rand(2, device="cuda")
        expected = eager(x, t, context)
        actual = compiled(x, t, context)
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for p, q in zip(eager.parameters(), compiled.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=2e-5)
    compiled.eval()
    eager.eval()
    for steps in (1, 2, 4):
        torch.manual_seed(18)
        a = RectifiedFlow(eager).predict_action(context, x, steps)
        torch.manual_seed(18)
        b = RectifiedFlow(compiled).predict_action(context, x, steps)
        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)


@compile_only
@pytest.mark.parametrize("policy", ["sat", "maniflow"])
def test_flow_agents_compile_online_and_ema(policy):
    cfg, batch = small_config(policy), sample_batch("cuda")
    model, ema, updater = build_model_and_ema(cfg, "cuda", normalizer_for(batch))
    eager = copy.deepcopy(model.action_decoder.model)
    compile_models(model, ema)  # production reduce-overhead; SAT must override.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch, **model.get_training_loss_kwargs(ema))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        optimizer.step()
        updater.step(model)
    eager.load_state_dict(
        fix_state_dict(model.action_decoder.model.state_dict(), False)
    )
    eager.eval()
    model.eval()
    cond, _ = model._build_cond(batch["obs"])
    x = (
        torch.randn(2, 3, 4, device="cuda")
        if policy == "sat"
        else torch.randn(2, 4, 3, device="cuda")
    )
    kwargs = {"x": x, "timestep": torch.full((2,), 0.5, device="cuda"), "context": cond}
    if policy == "maniflow":
        kwargs["target_t"] = torch.full((2,), 0.25, device="cuda")
    torch.compiler.cudagraph_mark_step_begin()
    with torch.no_grad():
        expected = eager(**kwargs)
        actual = model.action_decoder.model(**kwargs)
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)
        assert torch.isfinite(ema.action_decoder.model(**kwargs)).all()
