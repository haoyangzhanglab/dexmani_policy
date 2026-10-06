"""Native synthetic Policy Dataset -> VQ -> exported codebook -> DQRISE loading."""

import argparse

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.normalization import LinearNormalizer
from dexmani_policy.training.build_utils import build_normalizer
from scripts.training import train_vq_hand as vq


@pytest.fixture
def policy_config(tmp_path):
    path = tmp_path / "data.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["domain"] = "real"
    root.create_dataset("meta/episode_ends", data=np.arange(6, 25, 6))
    root.create_dataset(
        "data/action",
        data=np.repeat(np.arange(24, dtype="f4")[:, None], 19, axis=1),
        chunks=(3, 19),
    )
    root.create_dataset(
        "data/joint_state", data=np.zeros((24, 19), "f4"), chunks=(3, 19)
    )
    root.create_dataset("row_info/dispatch_status", data=np.ones((24, 2), "u1"))
    return OmegaConf.create(
        {
            "action_key": "action",
            "normalization": {"action": "auto", "joint_state": "limits"},
            "agent": {
                "_target_": "dexmani_policy.agents.core.dqrise.DQRISEAgent",
                "action_dim": 19,
                "tcp_dim": 7,
            },
            "dataset": {
                "_target_": "dexmani_policy.datasets.base_dataset.BaseDataset",
                "zarr_path": str(path),
                "action_key": "action",
                "sensor_modalities": ["joint_state"],
                "horizon": 3,
                "obs_horizon": 1,
                "val_ratio": 0.25,
                "seed": 42,
            },
        }
    )


def test_policy_vq_counterexample_and_no_agent_dependency(policy_config, monkeypatch):
    from dexmani_policy.agents.core.dqrise import DQRISEAgent

    monkeypatch.setattr(
        DQRISEAgent, "__init__", lambda *a, **k: pytest.fail("prepared an agent")
    )
    old = LinearNormalizer()
    old.fit(
        {"hand": np.repeat(np.arange(24, dtype="f4")[:, None], 12, axis=1)},
        range_eps=1e-4,
    )
    assert old["hand"].params_dict["scale"][0] == pytest.approx(2 / 23)
    train, val, normalizer, metadata = vq.prepare_policy_data(policy_config)
    assert train.shape == (18, 12) and val.shape == (6, 12)
    assert normalizer["hand"].params_dict["scale"][0] == pytest.approx(2 / 17)
    assert metadata["train_source_rows"] == list(range(6, 24))
    assert metadata["val_source_rows"] == list(range(6))
    assert float(val.min()) < -1  # validation never contributes to fit


@pytest.mark.parametrize("defect", ["observation", "dispatch", "constant"])
def test_policy_vq_reuses_window_qualification(policy_config, defect):
    root = zarr.open_group(policy_config.dataset.zarr_path, mode="a")
    if defect == "observation":
        root["data/joint_state"][15:18] = np.nan
    elif defect == "dispatch":
        root["row_info/dispatch_status"][16:18] = 0
    else:
        root["data/action"][:, 7] = 3
    dataset = hydra.utils.instantiate(policy_config.dataset)
    expected_field = build_normalizer(dataset, {"action": "auto"}, "action")["action"]
    expected = expected_field.params_dict
    train, val, normalizer, metadata = vq.prepare_policy_data(policy_config)
    actual = normalizer["hand"].params_dict
    for name in ("scale", "offset"):
        torch.testing.assert_close(actual[name], expected[name][7:19])
    for name, stats in expected_field.input_stats.items():
        torch.testing.assert_close(normalizer["hand"].input_stats[name], stats[7:19])
    assert metadata["train_source_rows"] == dataset.sampler.action_source_rows.tolist()
    assert len(train) == len(dataset.sampler.action_source_rows)
    assert len(val) == 6
    if defect != "constant":
        assert len(train) < 18


def test_policy_vq_rejects_aux_layout(policy_config):
    policy_config.dataset.use_aux_ee = True
    with pytest.raises(ValueError, match="aux"):
        vq.prepare_policy_data(policy_config)


@pytest.mark.parametrize("action_key", ["action", "action_ee"])
def test_vq_checkpoint_export_and_actual_policy_load(
    policy_config, tmp_path, action_key
):
    from dexmani_policy.agents.core.dqrise import DQRISEAgent
    from dexmani_policy.agents.vq_hand import VQVAEHand
    from scripts.training.extract_vq_codebook import extract_codebook

    if action_key == "action_ee":
        root = zarr.open_group(policy_config.dataset.zarr_path, mode="a")
        root.create_dataset(
            "data/action_ee",
            data=np.repeat(np.arange(24, dtype="f4")[:, None], 21, axis=1),
        )
        policy_config.action_key = policy_config.dataset.action_key = action_key
        policy_config.agent.tcp_dim, policy_config.agent.action_dim = 9, 21
    torch.manual_seed(123)
    torch.set_num_threads(1)
    _train, val, normalizer, metadata = vq.prepare_policy_data(policy_config)
    args = argparse.Namespace(
        vq_decay=0.8,
        threshold_ema_dead_code=0,
        kmeans_iters=2,
        action_key=action_key,
        tcp_dim=policy_config.agent.tcp_dim,
    )
    model = VQVAEHand(
        12,
        [1.0] * 12,
        latent_dim=4,
        hidden_dim=8,
        num_groups=1,
        codebook_size=2,
        num_layers=1,
        kmeans_init=True,
        kmeans_iters=2,
    )
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model.train()
    enc, commitment, _, mse = model(torch.from_numpy(_train))
    (enc + commitment).backward()
    optimizer.step()
    scheduler.step()
    assert torch.isfinite(mse)
    checkpoint = tmp_path / "synthetic.pt"
    vq._save_checkpoint(
        checkpoint,
        epoch=0,
        vqvae=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer=normalizer,
        args=args,
        train_history=[],
        split_metadata=metadata,
        metrics={},
    )
    saved = torch.load(checkpoint, weights_only=False)
    restored = LinearNormalizer()
    restored.load_state_dict(saved["normalizer_state_dict"])
    torch.testing.assert_close(
        restored["hand"].normalize(np.ones((1, 12), "f4")),
        normalizer["hand"].normalize(np.ones((1, 12), "f4")),
    )
    codebook = tmp_path / "synthetic.npz"
    extract_codebook(checkpoint, str(codebook), device="cpu")
    dataset = hydra.utils.instantiate(policy_config.dataset)
    policy_normalizer = build_normalizer(
        dataset, dict(policy_config.normalization), action_key
    )
    from dexmani_policy.agents.normalization import SingleFieldLinearNormalizer
    policy_normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
    agent = DQRISEAgent(
        horizon=4,
        n_obs_steps=1,
        n_action_steps=1,
        action_dim=policy_config.agent.action_dim,
        tcp_dim=policy_config.agent.tcp_dim,
        codebook_path=str(codebook),
        codebook_num_groups=1,
        codebook_size=2,
        pc_out_dim=16,
        state_out_dim=8,
        num_points=8,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
    )
    agent.load_normalizer_from_dataset(policy_normalizer)
    agent.initialize_training()
    assert agent._normalizer_checked
    agent.action_key = action_key
    batch = {"obs": {"point_cloud": torch.rand(2, 1, 8, 6),
                     "joint_state": torch.zeros(2, 1, 19)},
             "action": torch.ones(2, 4, policy_config.agent.action_dim)}
    opt = torch.optim.AdamW(agent.parameters(), lr=1e-4)
    loss, _ = agent(batch)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    agent.eval()
    prediction = agent.predict_action(batch["obs"], inference_steps=2)
    assert torch.isfinite(prediction["pred_action"]).all()
    import copy
    restored_agent = copy.deepcopy(agent)
    restored_agent.load_state_dict(agent.state_dict(), strict=True)
    torch.manual_seed(42)
    expected = agent.predict_action(batch["obs"], inference_steps=2)
    torch.manual_seed(42)
    actual = restored_agent.predict_action(batch["obs"], inference_steps=2)
    torch.testing.assert_close(actual["pred_action"], expected["pred_action"])
    with torch.no_grad():
        policy_normalizer["action"].params_dict["scale"][-1] *= 1.1
    with pytest.raises(ValueError, match="do not match"):
        agent.load_normalizer_from_dataset(policy_normalizer)
    from scripts.training.measure_vq_usage import measure

    usage = measure(
        str(checkpoint),
        policy_config.dataset.zarr_path,
        codebook_path=str(codebook),
        split="validation",
    )
    assert sum(usage["nn_counts"]) == len(val)


def test_policy_cli_uses_policy_data_recipe_and_independent_optimizer_seed(
    policy_config, tmp_path, monkeypatch
):
    policy_config.vq_vae = {
        "seed": 1234,
        "num_groups": 1,
        "codebook_size": 2,
        "device": "cpu",
    }
    policy_config.agent.codebook_num_groups = 1
    policy_config.agent.codebook_size = 2
    path = tmp_path / "policy.yaml"
    OmegaConf.save(policy_config, path)
    captured = []
    monkeypatch.setattr(
        vq, "train", lambda args, *, policy_cfg: captured.append((args, policy_cfg))
    )
    vq.main(
        [
            "--policy-config",
            str(path),
            "--output_dir",
            str(tmp_path / "output"),
            "--zarr_path",
            "ignored",
            "--val_ratio",
            "0.9",
            "--tcp_dim",
            "99",
        ]
    )
    args, cfg = captured[0]
    assert args.zarr_path == policy_config.dataset.zarr_path
    assert args.val_ratio == 0.25 and args.tcp_dim == 7 and args.seed == 1234
    assert cfg.dataset.seed == 42


@pytest.mark.parametrize("mode", ["limits", "gaussian"])
def test_policy_vq_action_spec_without_observation_fit(
    policy_config, monkeypatch, mode
):
    from dexmani_policy.datasets.base_dataset import BaseDataset

    calls = []
    original = BaseDataset.iter_normalization_data

    def iterate(self, key):
        calls.append(key)
        return original(self, key)

    monkeypatch.setattr(BaseDataset, "iter_normalization_data", iterate)
    policy_config.normalization.action = mode
    _, _, normalizer, metadata = vq.prepare_policy_data(policy_config)
    assert calls == ["action"]
    assert metadata["normalization_spec"]["action"] == mode
    expected = build_normalizer(
        hydra.utils.instantiate(policy_config.dataset), {"action": mode}, "action"
    )
    for key in ("scale", "offset"):
        torch.testing.assert_close(
            normalizer["hand"].params_dict[key],
            expected["action"].params_dict[key][7:19],
        )


def test_standalone_recipe_keeps_full_data_statistics(policy_config):
    args = argparse.Namespace(
        zarr_path=policy_config.dataset.zarr_path,
        action_key="action",
        tcp_dim=7,
        hand_dim=12,
        seed=42,
        val_ratio=0.25,
        max_train_episodes=None,
    )
    _, _, normalizer, metadata = vq.prepare_standalone_data(args)
    assert normalizer["hand"].params_dict["scale"][0] == pytest.approx(2 / 23)
    assert metadata["train_frame_count"] == 18
