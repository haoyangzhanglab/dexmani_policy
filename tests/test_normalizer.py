import numpy as np
import pytest
import torch

from dexmani_policy.common.normalizer import LinearNormalizer


def _rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_fit_field_limits_matches_one_shot_fit():
    data = _rand((500, 6))
    ref = LinearNormalizer()
    ref.fit({"k": data}, last_n_dims=1, mode="limits")

    norm = LinearNormalizer()
    norm.fit_field("k", data, mode="limits")

    torch.testing.assert_close(
        norm.params_dict["k"]["scale"], ref.params_dict["k"]["scale"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        norm.params_dict["k"]["offset"], ref.params_dict["k"]["offset"], rtol=0, atol=0
    )


def test_fit_field_gaussian_matches_one_shot_fit():
    data = _rand((500, 6), seed=1)
    ref = LinearNormalizer()
    ref.fit({"k": data}, last_n_dims=1, mode="gaussian")

    norm = LinearNormalizer()
    norm.fit_field("k", data, mode="gaussian")

    torch.testing.assert_close(
        norm.params_dict["k"]["scale"], ref.params_dict["k"]["scale"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        norm.params_dict["k"]["offset"], ref.params_dict["k"]["offset"], rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", ["limits", "gaussian"])
def test_fit_field_chunks_matches_concatenated_fit(mode):
    data = _rand((1000, 6), seed=2)
    chunks = [data[:300], data[300:750], data[750:]]

    ref = LinearNormalizer()
    ref.fit({"k": data}, last_n_dims=1, mode=mode)

    norm = LinearNormalizer()
    norm.fit_field_chunks("k", chunks, last_n_dims=1, mode=mode)

    torch.testing.assert_close(
        norm.params_dict["k"]["scale"], ref.params_dict["k"]["scale"], rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        norm.params_dict["k"]["offset"], ref.params_dict["k"]["offset"], rtol=1e-4, atol=1e-6
    )


def test_near_constant_dim_identical_to_fit():
    data = _rand((200, 4), seed=3)
    data[:, 2] = 5.0  # constant dim
    ref = LinearNormalizer()
    ref.fit({"k": data}, mode="limits")
    norm = LinearNormalizer()
    norm.fit_field("k", data, mode="limits")

    torch.testing.assert_close(
        norm.params_dict["k"]["scale"], ref.params_dict["k"]["scale"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        norm.params_dict["k"]["offset"], ref.params_dict["k"]["offset"], rtol=0, atol=0
    )
    # near-constant dim -> identity scale, zero-centered offset
    assert norm.params_dict["k"]["scale"][2].item() == 1.0
    assert norm.params_dict["k"]["offset"][2].item() == pytest.approx(-5.0, abs=1e-5)


def test_identity_field_not_registered_and_zero_op_passthrough():
    norm = LinearNormalizer()
    norm.fit_field("action", _rand((100, 3)), mode="limits")

    rgb = torch.randint(0, 255, (2, 5, 224, 224, 3), dtype=torch.uint8)
    out = norm.normalize({"rgb": rgb, "action": torch.randn(2, 5, 3)})

    assert "rgb" not in norm.params_dict
    assert out["rgb"] is rgb  # passthrough must be a true zero-op (same object)


def test_fit_field_chunks_rejects_last_dim_mismatch():
    norm = LinearNormalizer()
    with pytest.raises(ValueError):
        norm.fit_field_chunks("k", [torch.randn(10, 3), torch.randn(10, 4)], mode="limits")


def test_fit_field_chunks_rejects_empty():
    norm = LinearNormalizer()
    with pytest.raises(ValueError):
        norm.fit_field_chunks("k", [], mode="limits")


def test_fit_field_chunks_single_sample_limits_ok():
    # limits tolerates a single sample: min == max -> near-constant identity scale.
    norm = LinearNormalizer()
    data = _rand((1, 6), seed=10)
    norm.fit_field_chunks("k", [data], mode="limits")
    scale = norm.params_dict["k"]["scale"]
    offset = norm.params_dict["k"]["offset"]
    assert scale.shape == (6,)
    assert (scale == 1).all()  # identity scale for near-constant dims
    assert bool(torch.isfinite(offset).all())


def test_fit_field_chunks_single_sample_gaussian_raises():
    # gaussian needs >= 2 samples to define unbiased std; fail fast, never NaN.
    norm = LinearNormalizer()
    data = _rand((1, 6), seed=11)
    with pytest.raises(ValueError, match="at least 2 samples"):
        norm.fit_field_chunks("k", [data], mode="gaussian")


@pytest.mark.parametrize("mode", ["limits", "gaussian"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fit_field_chunks_output_dtype_matches_one_shot(mode, dtype):
    # float64 is used ONLY for the streaming accumulator; the stored scale/offset
    # and input_stats must match one-shot fit_params(dtype=...) exactly.
    data = _rand((1000, 6), seed=7)
    chunks = [data[:400], data[400:]]

    ref = LinearNormalizer()
    ref.fit_field("k", data, mode=mode, dtype=dtype)

    norm = LinearNormalizer()
    norm.fit_field_chunks("k", chunks, mode=mode, dtype=dtype)

    for name in ("scale", "offset"):
        assert norm.params_dict["k"][name].dtype == dtype
        assert norm.params_dict["k"][name].dtype == ref.params_dict["k"][name].dtype
    for name in ("min", "max", "mean", "std"):
        assert norm.input_stats["k"][name].dtype == dtype
        assert norm.input_stats["k"][name].dtype == ref.input_stats["k"][name].dtype


def test_fit_field_chunks_std_is_unbiased():
    # Explicitly lock the unbiased (sample) std semantics against torch.std default.
    data = _rand((5000, 4), seed=8)
    chunks = [data[:2000], data[2000:]]

    norm = LinearNormalizer()
    norm.fit_field_chunks("k", chunks, mode="gaussian")

    ref_std = torch.from_numpy(data).float().std(dim=0, unbiased=True)
    torch.testing.assert_close(
        norm.input_stats["k"]["std"], ref_std, rtol=1e-4, atol=1e-6
    )


def test_fit_field_chunks_dtype_none_does_not_leak_float64():
    # dtype=None must collapse to float32 (never leak float64 into stored params),
    # matching one-shot fit_params(dtype=None) on float32 input.
    data = _rand((500, 6), seed=9)
    chunks = [data[:250], data[250:]]

    norm = LinearNormalizer()
    norm.fit_field_chunks("k", chunks, mode="limits", dtype=None)

    ref = LinearNormalizer()
    ref.fit_field("k", data, mode="limits", dtype=None)

    for name in ("scale", "offset"):
        assert norm.params_dict["k"][name].dtype == torch.float32
        assert norm.params_dict["k"][name].dtype == ref.params_dict["k"][name].dtype
    for name in ("min", "max", "mean", "std"):
        assert norm.input_stats["k"][name].dtype == torch.float32
        assert norm.input_stats["k"][name].dtype == ref.input_stats["k"][name].dtype
