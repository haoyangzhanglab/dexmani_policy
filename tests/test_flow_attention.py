"""Mask semantics independent of the backbone's zero-initialized output."""

import copy

import pytest
import torch
from timm.models.vision_transformer import RmsNorm

from dexmani_policy.agents.action_decoders.backbone.attention import CrossAttention


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("all_masked", [False, True])
def test_cross_attention_matches_reference(device, all_masked):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    model = (
        CrossAttention(16, 4, qkv_bias=True, qk_norm=True, norm_layer=RmsNorm)
        .to(device)
        .eval()
    )
    x = torch.randn(2, 3, 16, device=device)
    context = torch.randn(2, 5, 16, device=device)
    mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, False, not all_masked, False, False],
        ],
        device=device,
    )
    q = model.q_norm(model.q(x).reshape(2, 3, 4, 4).transpose(1, 2))
    kv = model.kv(context).reshape(2, 5, 2, 4, 4).permute(2, 0, 3, 1, 4)
    k, v = model.k_norm(kv[0]), kv[1]
    # Select valid keys explicitly; no SDPA/masked_fill in this oracle.
    expected = []
    for i in range(2):
        valid = mask[i]
        if valid.any():
            logits = q[i] @ k[i, :, valid].transpose(-2, -1) / 2
            row = logits.softmax(-1) @ v[i, :, valid]
        else:
            row = torch.zeros_like(q[i])
        expected.append(row.transpose(0, 1).reshape(3, 16))
    expected = model.proj(torch.stack(expected))
    outputs, grads = [], []
    for fused in [False, True]:
        candidate = copy.deepcopy(model)
        candidate.fused_attn = fused
        result = candidate(x, context, mask)
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
        perturbed = context.clone()
        perturbed[~mask] = 100 * torch.randn_like(perturbed[~mask])
        torch.testing.assert_close(candidate(x, perturbed, mask), result)
        result.square().sum().backward()
        gradients = [p.grad for p in candidate.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        outputs.append(result)
        grads.append(gradients)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
