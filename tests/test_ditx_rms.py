import torch

from dexmani_policy.agents.action_decoders.backbone.ditx_rms import DiTXRMS


def _make_model():
    return DiTXRMS(
        horizon=4,
        action_dim=3,
        obs_token_dim=6,
        timestep_embed_dim=16,
        n_layers=2,
        hidden_dim=32,
        n_head=4,
        ffn_dim=48,
        p_drop_attn=0.0,
        qkv_bias=True,
        qk_norm=True,
    )


def test_ditx_rms_supports_dynamic_observation_length():
    model = _make_model()
    x = torch.randn(2, 4, 3)
    t = torch.rand(2)

    for token_count in (5, 9):
        context = torch.randn(2, token_count, 6)
        out = model(x, t, context)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_ditx_rms_context_mask_and_backward():
    model = _make_model()
    x = torch.randn(2, 4, 3)
    t = torch.rand(2)
    context = torch.randn(2, 7, 6)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )

    out = model(x, t, context, context_mask=mask)
    loss = (out - torch.ones_like(out)).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_ditx_rms_rejects_wrong_action_shape():
    model = _make_model()
    context = torch.randn(2, 5, 6)

    try:
        model(torch.randn(2, 5, 3), torch.rand(2), context)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("wrong horizon must fail")


def test_zero_initialization_starts_learning_and_optimizer_covers_parameters():
    torch.manual_seed(7)
    model = _make_model()
    groups = model.get_optim_groups(weight_decay=0.0)
    grouped = [p for group in groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    x, t, context = torch.randn(2, 4, 3), torch.rand(2), torch.randn(2, 7, 6)
    target = torch.randn_like(x)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    reached = set()
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, t, context) - target).square().mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
            if p.grad.abs().max() > 0:
                reached.add(name)
            if step == 0 and not name.startswith("final_proj."):
                assert torch.count_nonzero(p.grad) == 0, name
        optimizer.step()
    assert reached == set(original), set(original) - reached
    for name, p in model.named_parameters():
        assert not torch.equal(p, original[name]), name


def test_context_permutation_invariance_after_learning():
    model = _make_model().eval()
    # Activate all residual branches and final output so this is not vacuous.
    with torch.no_grad():
        torch.nn.init.normal_(model.final_proj.weight, std=0.1)
        for block in model.blocks:
            block.modulation[-1].bias.fill_(0.2)
    x, context = torch.randn(2, 4, 3), torch.randn(2, 9, 6)
    permutation = torch.randperm(9)
    mask = torch.rand(2, 9) > 0.3
    a = model(x, 0.5, context, mask)
    b = model(x, torch.tensor(0.5), context[:, permutation], mask[:, permutation])
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
