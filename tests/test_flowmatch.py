from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones import ditx
from dexmani_policy.agents.action_decoders.backbones.factorized_action_io import FactorizedActionIO
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow

from tests.helpers import (
    check_flow_recipe_defaults, check_observation_identity_constraints, check_recipe_composition,
    check_rgb_flow_lifecycle,
)


@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("batch,obs_steps,steps", [(1, 1, 1), (2, 2, 5), (2, 3, 5), (2, 2, 10)])
def test_joint_flow_coupling_and_sampling(factorized, batch, obs_steps, steps):
    torch.manual_seed(7)
    model = ditx.DiTX(
        horizon=4, action_dim=19, n_obs_steps=obs_steps, obs_token_dim=8,
        hidden_dim=16, n_layers=2, n_head=2, p_drop_attn=0,
        action_io=FactorizedActionIO(4, 16) if factorized else None,
    ).eval()
    heads = (model.action_io.arm_head, model.action_io.hand_head) if factorized else (model.final_layer,)
    projections = (model.action_io.arm_embedder, model.action_io.hand_embedder) if factorized else (model.input_embedder,)
    # Open the zero-initialized gates and heads to test actual dependencies.
    with torch.no_grad():
        model.adaln_modulation.projection.weight.normal_(std=0.1)
        for head in heads:
            head.ffn_final.fc2.weight.normal_(std=0.1)
    actions = torch.randn(batch, 4, 19, requires_grad=True)
    context = torch.randn(batch, obs_steps * 3, 8, requires_grad=True)
    velocity = model(actions, 0.4, context)
    arm_grad, = torch.autograd.grad(velocity[..., :7].square().mean(), actions, retain_graph=True)
    hand_grad, = torch.autograd.grad(velocity[..., 7:].square().mean(), actions, retain_graph=True)
    assert arm_grad[..., 7:].abs().sum() > 0
    assert hand_grad[..., :7].abs().sum() > 0
    torch.testing.assert_close(velocity, model(actions, 0.4, context, context_kv=model.prepare_context_kv(context)))
    decoder = RectifiedFlow(model, num_inference_steps=5, time_shift_alpha=3).eval()
    torch.manual_seed(17)
    xt, t, target = decoder.sample_training_pair(actions)
    expected = F.mse_loss(model(xt, t, context).float(), target.float())
    torch.manual_seed(17)
    loss, _ = decoder.compute_loss(context, actions)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for projection in projections:
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0
    with ExitStack() as stack:
        calls = [stack.enter_context(patch.object(block.cross_attn, "prepare_kv", wraps=block.cross_attn.prepare_kv))
                 for block in model.ditx_blocks]
        sample = decoder.predict_action(context, actions.shape, inference_steps=steps,
                                        generator=torch.Generator().manual_seed(23))
        assert all(call.call_count == 1 for call in calls)
    reference = torch.randn(actions.shape, generator=torch.Generator().manual_seed(23))
    with torch.no_grad():
        for i in range(steps):
            reference = reference + model(reference, i / steps, context) / steps
    torch.testing.assert_close(sample, reference)
    guided = decoder.predict_action(context, actions.shape, rtc_prefix=actions.detach()[:, 1:3],
                                    history_offset=1, rtc_guidance_cap=0.2)
    assert guided.shape == actions.shape and torch.isfinite(guided).all()


@pytest.mark.parametrize("horizon", [1, 4, 16])
def test_factorized_action_io(horizon):
    io = FactorizedActionIO(horizon, 16)
    actions = torch.randn(2, horizon, 19)
    tokens = io.encode(actions)
    assert tokens.shape == (2, 2 * horizon, 16)
    torch.testing.assert_close(tokens[:, :horizon], io.arm_embedder(actions[..., :7]) + io.time_pos_embed)
    torch.testing.assert_close(tokens[:, horizon:], io.hand_embedder(actions[..., 7:]) + io.time_pos_embed)
    for part, token_slice in ((slice(0, 7), slice(horizon, None)), (slice(7, None), slice(0, horizon))):
        changed = actions.clone()
        changed[..., part] += 1
        torch.testing.assert_close(io.encode(changed)[:, token_slice], tokens[:, token_slice])
    velocity = io.decode_velocity(tokens)
    assert velocity.shape == actions.shape
    assert torch.count_nonzero(velocity) == 0
    (velocity - 1).square().mean().backward()
    for head in (io.arm_head, io.hand_head):
        assert head.ffn_final.fc2.weight.grad.abs().sum() > 0
    with torch.no_grad():
        io.arm_head.ffn_final.fc2.weight.normal_()
        io.hand_head.ffn_final.fc2.weight.normal_()
    torch.testing.assert_close(io.decode_velocity(tokens), torch.cat((
        io.arm_head(tokens[:, :horizon]), io.hand_head(tokens[:, horizon:]),
    ), dim=-1))


@pytest.mark.parametrize("options", [[], ["agent.factorized=true"]])
def test_rgb_flow_initialization_and_checkpoint_lifecycle(tmp_path, make_demo, monkeypatch, options):
    model, cfg = check_rgb_flow_lifecycle(tmp_path, make_demo, monkeypatch, "flowmatch", options, "freeze")
    backbone = model.action_decoder.model
    assert isinstance(backbone.action_io, FactorizedActionIO) == cfg.agent.factorized
    assert hasattr(backbone, "input_embedder") == (not cfg.agent.factorized)
    assert hasattr(backbone, "final_layer") == (not cfg.agent.factorized)


def test_recipe_composition():
    cfg = check_recipe_composition("flowmatch")
    check_flow_recipe_defaults("flowmatch", cfg)
    assert cfg.agent.factorized is False
    check_observation_identity_constraints("flowmatch", ['camera_extrinsic', 'camera_intrinsic', 'point_cloud'])
