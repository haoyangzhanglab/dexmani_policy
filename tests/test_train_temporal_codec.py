from argparse import Namespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.temporal_codec import TemporalHandCodec, load_codec
from dexmani_policy.training import train_temporal_codec as training
from test_temporal_codec_dataset import make_dataset, write_data


def test_evaluation_selects_original_action_units(tmp_path):
    path = tmp_path / "data.zarr"
    write_data(path)
    dataset = make_dataset(path)
    codec = TemporalHandCodec(horizon=4, latent_dim=2, hidden_dim=8, n_layers=1, n_heads=2)
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.zero_()
        codec.action_mean.fill_(0.5)
        codec.action_std.copy_(torch.arange(1, 13, dtype=torch.float32))
    metrics = training.evaluate(codec, dataset, batch_size=3)
    actions, q0 = dataset.raw_hand_batch(np.arange(len(dataset)))
    expected = np.abs(q0[:, None] + 0.5 - actions).mean(axis=(0, 1))
    np.testing.assert_allclose(metrics["physical_joint_mae"], expected, atol=1e-6)
    assert metrics["physical_mae"] == pytest.approx(float(expected.mean()))
    assert codec.training  # Evaluation preserves the caller's mode.


@pytest.mark.parametrize("reference_noise_std", [0.0, 0.03])
def test_small_train_fallback_is_explicit_and_artifact_can_be_reloaded(
        tmp_path, monkeypatch, capsys, reference_noise_std):
    path, output = tmp_path / "data.zarr", tmp_path / "codec.pt"
    write_data(path)
    cfg = OmegaConf.create({
        "action_key": "action", "action_dim": 19, "horizon": 4, "n_obs_steps": 2,
        "codec": {"path": str(output), "latent_dim": 2, "hidden_dim": 8, "n_layers": 1,
                  "n_heads": 2, "mlp_ratio": 2, "joint_weights": [1] * 8 + [0.5] * 4},
        "agent": {"codec_path": str(output)},
        "dataset": {"_target_": "dexmani_policy.datasets.temporal_codec_dataset.TemporalCodecDataset",
                    "zarr_path": str(path), "codec_path": str(output), "horizon": 4,
                    "obs_horizon": 2, "pad_before": 1, "pad_after": 1,
                    "sensor_modalities": ["joint_state"]},
    })
    monkeypatch.setattr(training, "load_policy_config", lambda args: cfg)
    args = Namespace(steps=2, batch_size=2, eval_every=1, print_every=1,
                     kl_warmup=0, kl_ramp=2, beta=3e-4, weight_decay=1e-4,
                     lr=3e-4, clip_grad=1.0, device="cpu", seed=42, overwrite=False,
                     reference_noise_std=reference_noise_std)
    original_batch = training.raw_batch
    original_encode = TemporalHandCodec.encode
    original_decode = TemporalHandCodec.decode
    original_normalize = TemporalHandCodec.normalize_actions
    reference_rng = torch.Generator().manual_seed(args.seed + 1)
    current, modes = {}, []

    def traced_batch(*a, **kw):
        actions, q0 = original_batch(*a, **kw)
        current.update(actions=actions.clone(), raw_q0=q0, q0=q0.clone(), rng=torch.get_rng_state())
        return actions, q0

    def traced_encode(codec, actions, q0):
        # Reference augmentation must not modify commands or the raw batch, nor
        # consume the global RNG used by posterior samples in either mode.
        torch.testing.assert_close(actions, current["actions"], rtol=0, atol=0)
        torch.testing.assert_close(current["raw_q0"], current["q0"], rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), current["rng"])
        expected = current["q0"]
        if codec.training and reference_noise_std:
            expected = expected + reference_noise_std * torch.randn(expected.shape, generator=reference_rng)
            assert not torch.equal(q0, current["q0"])
        torch.testing.assert_close(q0, expected, rtol=0, atol=0)
        current["reference"] = q0.clone()
        modes.append(codec.training)
        return original_encode(codec, actions, q0)

    def traced_decode(codec, z, q0):
        torch.testing.assert_close(q0, current["reference"], rtol=0, atol=0)
        return original_decode(codec, z, q0)

    def traced_normalize(codec, actions, q0):
        torch.testing.assert_close(actions, current["actions"], rtol=0, atol=0)
        torch.testing.assert_close(q0, current["reference"], rtol=0, atol=0)
        return original_normalize(codec, actions, q0)

    with monkeypatch.context() as spies:
        spies.setattr(training, "raw_batch", traced_batch)
        spies.setattr(TemporalHandCodec, "encode", traced_encode)
        spies.setattr(TemporalHandCodec, "decode", traced_decode)
        spies.setattr(TemporalHandCodec, "normalize_actions", traced_normalize)
        training.train(args)
    assert modes.count(True) == args.steps
    assert False in modes  # Validation really exercised the unaugmented path.
    codec, payload = load_codec(output)
    dataset = make_dataset(path, codec_path=output, require_codec=True)
    assert dataset.codec_sha256 == payload["_file_sha256"]
    assert payload["metrics"]["selection_split"] == "train (no held-out windows)"
    assert payload["metadata"]["relative_statistics"]["frames"] == len(dataset) * 4
    expected_mean, expected_std, _ = training.fit_relative_statistics(dataset, args.batch_size)
    torch.testing.assert_close(codec.action_mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(codec.action_std, expected_std, rtol=0, atol=0)
    assert payload["train_args"]["reference_noise_std"] == reference_noise_std
    assert payload["metadata"]["reference_augmentation"] == {
        "mode": "independent_gaussian_per_window_joint", "units": "radians",
        "std": reference_noise_std, "generator_seed": 43,
        "physical_action_target": "unchanged", "statistics": "unaugmented_training_windows",
    }
    assert payload["metadata"]["loss_reduction"] == {
        "kl": "mean_over_batch_time_latent",
        "reconstruction": "weighted_huber_mean_over_batch_valid_frames_joints",
        "joint_weights": "multiply_numerator_only_do_not_change_denominator",
    }
    torch.testing.assert_close(codec.joint_weights, torch.tensor([1.] * 8 + [.5] * 4))
    assert "no held-out windows" in capsys.readouterr().out
    assert torch.isfinite(codec.decode_actions(*_encoded(codec, dataset))).all()
    with pytest.raises(FileExistsError, match="--overwrite"):
        training.train(args)


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_reference_noise_is_rejected_before_loading_data(value):
    args = Namespace(steps=1, batch_size=2, eval_every=1, print_every=1,
                     kl_warmup=0, kl_ramp=2, beta=3e-4, weight_decay=1e-4,
                     reference_noise_std=value)
    with pytest.raises(ValueError, match="reference_noise_std must be nonnegative and finite"):
        training.train(args)


def test_cli_reference_noise_defaults_to_zero(monkeypatch):
    observed = {}
    monkeypatch.setattr("sys.argv", ["train_temporal_codec"])
    monkeypatch.setattr(training, "train", lambda args: observed.update(vars(args)))
    training.main()
    assert observed["reference_noise_std"] == 0.0


def _encoded(codec, dataset):
    actions, q0 = dataset.raw_hand_batch(np.array([0, 1]))
    mu, _ = codec.encode(torch.from_numpy(actions), torch.from_numpy(q0))
    return mu, torch.from_numpy(q0)
