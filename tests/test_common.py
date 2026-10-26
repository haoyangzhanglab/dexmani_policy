import copy

import pytest
import torch
from torch import nn

from dexmani_policy.common.checkpoint import load_policy_state
from dexmani_policy.common.image_util import resize_image_transform
from dexmani_policy.common.normalizer import LinearNormalizer


@pytest.mark.parametrize("key,shape", [("action", (19,)), ("finger_tactile_force", (5, 3))])
@pytest.mark.parametrize("mode", ["limits", "gaussian"])
def test_streaming_normalization(key, shape, mode):
    values = torch.randn(31, *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    values.reshape(31, -1)[:, 0] = 3
    original = values.clone()
    normalizer = LinearNormalizer()
    normalizer.fit_field(key, [values[:1], values[1:7], values[7:]], mode=mode)
    field = normalizer[key]

    for name, expected in {
        "min": values.amin(0), "max": values.amax(0),
        "mean": values.mean(0), "std": values.std(0, correction=1),
    }.items():
        torch.testing.assert_close(field.input_stats[name], expected.float())
    normalized = field.normalize(values)
    torch.testing.assert_close(field.unnormalize(normalized), values.float())
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert normalized.dtype == torch.float32 and field.scale.shape == shape
    torch.testing.assert_close(normalized.reshape(31, -1)[:, 0], torch.zeros(31))
    if mode == "limits":
        torch.testing.assert_close(normalized.reshape(31, -1)[:, 1:].amin(0), -torch.ones(values[0].numel() - 1))
        torch.testing.assert_close(normalized.reshape(31, -1)[:, 1:].amax(0), torch.ones(values[0].numel() - 1))
    else:
        torch.testing.assert_close(normalized.reshape(31, -1)[:, 1:].std(0), torch.ones(values[0].numel() - 1))


def test_normalizer_buffers_and_strict_loading():
    source = LinearNormalizer()
    source.fit_field("action", [torch.arange(60).reshape(20, 3)])
    source.fit_field("fingertip_xyz", [torch.arange(120).reshape(8, 5, 3)])
    restored = LinearNormalizer({"action": 3, "fingertip_xyz": 15})
    restored.load_state_dict(source.state_dict())
    torch.testing.assert_close(restored.state_dict(), source.state_dict())
    assert list(restored.parameters()) == []
    assert restored["action"] is restored["action"]
    assert all(buffer.device.type == "meta" for buffer in copy.deepcopy(restored).to("meta").buffers())
    restored.double()
    assert restored["action"].normalize(torch.ones(3)).dtype == torch.float64

    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict({})
    with pytest.raises(RuntimeError, match="size mismatch"):
        LinearNormalizer({"action": 4, "fingertip_xyz": 15}).load_state_dict(source.state_dict())


def test_policy_state_preserves_precision_and_rejects_invalid_affine():
    source = nn.Module()
    source.model = nn.Linear(3, 3).bfloat16()
    source.normalizer = LinearNormalizer({"action": 3})
    target = copy.deepcopy(source)
    target.model.float()
    load_policy_state(target, source.state_dict())
    assert target.model.weight.dtype == torch.bfloat16
    assert target.normalizer["action"].scale.dtype == torch.float32

    source.model.float()
    load_policy_state(target, source.state_dict())
    assert target.model.weight.dtype == torch.float32
    state = copy.deepcopy(source.state_dict())
    state["normalizer.fields.action.scale"][0] = 0
    with pytest.raises(ValueError, match="Invalid affine"):
        load_policy_state(target, state)


@pytest.mark.parametrize("values,mode", [
    (torch.empty(0, 3), "limits"),
    (torch.ones(1, 3), "gaussian"),
    (torch.full((2, 3), float("nan")), "limits"),
])
def test_normalizer_rejects_unusable_statistics(values, mode):
    with pytest.raises(ValueError, match="Insufficient|Nonfinite"):
        LinearNormalizer().fit_field("action", [values], mode=mode)


def test_resize_transform_maps_pixel_centers():
    transform = resize_image_transform((4, 8), (8, 4))
    corners = torch.tensor([[0., 0., 1.], [7., 3., 1.]])
    torch.testing.assert_close(corners @ transform.T, torch.tensor([[-0.25, 0.5, 1.], [3.25, 6.5, 1.]]))
