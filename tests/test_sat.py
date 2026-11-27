import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones import ditx, sat_action_io
from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.modules import FlowTimestepEmbedding
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow

from tests.helpers import (
    check_context_cache, check_flow_recipe_defaults, check_observation_identity_constraints,
    check_recipe_composition, check_rgb_flow_lifecycle,
)


def test_sat_readout_formula_with_nonzero_output():
    torch.manual_seed(3)
    io = sat_action_io.SATActionIO(4, 16)
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
    io = sat_action_io.SATActionIO(4, 16)
    layout = {"horizon": 4, "action_dim": 19} if backbone_class is ditx.DiTX else {}
    backbone = backbone_class(
        n_obs_steps=1, obs_token_dim=8, hidden_dim=16, n_head=2, n_layers=2,
        timestep_embed_dim=8, p_drop_attn=0, action_io=io, **layout,
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


@pytest.mark.parametrize("options,tune_mode", [
    (["agent.backbone_type=ditx"], "full"), (["agent.backbone_type=dit"], "full"),
    (["agent.backbone_type=ditx"], "freeze"), (["agent.backbone_type=ditx"], "lora"),
])
def test_rgb_flow_initialization_and_checkpoint_lifecycle(tmp_path, make_demo, monkeypatch, options, tune_mode):
    check_rgb_flow_lifecycle(tmp_path, make_demo, monkeypatch, "sat", options, tune_mode)


def test_recipe_composition():
    cfg = check_recipe_composition("sat")
    check_flow_recipe_defaults("sat", cfg)
    check_observation_identity_constraints("sat", ['camera_extrinsic', 'camera_intrinsic', 'point_cloud'])


def test_context_cache_outputs_gradients_and_new_observation_dit(monkeypatch):
    torch.manual_seed(4)
    layout = {}
    layout["action_io"] = sat_action_io.SATActionIO(4, 16)
    backbone = DiT(
        n_obs_steps=1, obs_token_dim=8, hidden_dim=16, n_head=2, n_layers=2,
        timestep_embed_dim=8, p_drop_attn=0, **layout,
    ).eval()
    check_context_cache(monkeypatch, backbone, RectifiedFlow, {})


def test_context_cache_outputs_gradients_and_new_observation_ditx(monkeypatch):
    torch.manual_seed(4)
    layout = {"horizon": 4, "action_dim": 19}
    layout["action_io"] = sat_action_io.SATActionIO(4, 16)
    backbone = ditx.DiTX(
        n_obs_steps=1, obs_token_dim=8, hidden_dim=16, n_head=2, n_layers=2,
        timestep_embed_dim=8, p_drop_attn=0, **layout,
    ).eval()
    check_context_cache(monkeypatch, backbone, RectifiedFlow, {})


def test_flow_backbone_time_and_qk_defaults_dit():
    layout = {"horizon": 4, "action_dim": 19}
    layout = {"action_io": sat_action_io.SATActionIO(4, 16)}
    model = DiT(n_obs_steps=2, obs_token_dim=8, hidden_dim=16, n_layers=2, n_head=2, **layout)
    assert isinstance(model.timestep_embedder, FlowTimestepEmbedding)
    attentions = model.dit_blocks
    assert len(attentions) == (2)
    for attention in attentions:
        assert isinstance(attention.q_norm, nn.LayerNorm)
        assert isinstance(attention.k_norm, nn.LayerNorm)
        assert attention.q_norm.normalized_shape == attention.k_norm.normalized_shape == (8,)
