import importlib

import pytest
import torch
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones import action_io, ditx
from dexmani_policy.agents.action_decoders.backbones.dit import DiT


def test_sat_readout_formula_with_nonzero_output():
    torch.manual_seed(3)
    io = action_io.SATActionIO(4, 16)
    readout = io.final_layer
    with torch.no_grad():
        readout.output.weight.normal_()
        readout.output.bias.normal_()
    tokens = torch.randn(2, 19, 16, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_(True)
    actual = io.decode_velocity(tokens)
    normalized = reference * (reference.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * readout.norm.weight
    hidden = F.gelu(F.linear(normalized, readout.hidden.weight, readout.hidden.bias), approximate="tanh")
    expected = F.linear(hidden, readout.output.weight, readout.output.bias).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    parameters = tuple(io.final_layer.parameters())
    actual_grads = torch.autograd.grad(actual.square().mean(), (tokens, *parameters))
    expected_grads = torch.autograd.grad(expected.square().mean(), (reference, *parameters))
    torch.testing.assert_close(actual_grads, expected_grads)


@pytest.mark.parametrize("backbone_class", [DiT, ditx.DiTX])
def test_sat_backbones(backbone_class):
    io = action_io.SATActionIO(4, 16)
    backbone = backbone_class(
        horizon=4, action_dim=19, n_obs_steps=1, obs_token_dim=8,
        hidden_dim=16, n_head=2, n_layers=2, timestep_embed_dim=8,
        p_drop_attn=0, action_io=io,
    )
    actions = torch.randn(2, 4, 19, requires_grad=True)
    tokens = io.encode(actions)
    assert tokens.shape == (2, 19, 16)
    context = torch.randn(2, 3, 8)
    output = backbone(actions, torch.tensor([0.2, 0.7]), context)
    assert output.shape == actions.shape
    assert torch.count_nonzero(output) == 0
    (output - 1).square().mean().backward()
    head = io.final_layer.output
    assert head.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(head.bias) == 0
    with torch.no_grad():
        head.weight.normal_()
    backbone.zero_grad()
    backbone(actions, 0.5, context).square().mean().backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0


def test_dit_branch_does_not_construct_ditx_output(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SAT must own its readout")

    monkeypatch.setattr(ditx, "DiTXOutput", forbidden)
    module = importlib.reload(action_io)
    io = module.SATActionIO(4, 16)
    backbone = DiT(
        horizon=4, action_dim=19, n_obs_steps=1, obs_token_dim=8,
        hidden_dim=16, n_head=2, n_layers=1, action_io=io,
    )
    assert backbone(torch.randn(2, 4, 19), 0.5, torch.randn(2, 3, 8)).shape == (2, 4, 19)
