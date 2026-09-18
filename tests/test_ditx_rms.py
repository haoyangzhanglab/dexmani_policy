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
    loss = out.square().mean()
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
