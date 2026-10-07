"""Native synthetic Policy Dataset -> VQ -> exported codebook -> DQRISE loading."""

import argparse
import json

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.normalization import LinearNormalizer
from dexmani_policy.training.build_utils import build_normalizer
from scripts.training import train_vq_hand as vq


def usage_checkpoint(cfg, tmp_path, *, manifest=False, capped=False):
    from dexmani_policy.agents.vq_hand import VQVAEHand
    root = zarr.open_group(cfg.dataset.zarr_path, mode='a')
    root.attrs.update(data_revision='saved-revision', episode_ids=['a','b','c','d'])
    if manifest:
        path = tmp_path/'split.json'
        path.write_text(json.dumps({'data_revision':'saved-revision', 'episode_ids':['a','b','c','d'],
            'train_ids':['a','b'], 'val_ids':['c'], 'exclusions':['d'],
            'trial_ids':{k:k for k in 'abcd'}, 'seed':42, 'group_unit':'trial'}))
        cfg.dataset.split_manifest = str(path); cfg.dataset.val_ratio = 0
    _, _, normalizer, metadata = vq.prepare_policy_data(cfg)
    if capped:
        # Saved evidence from an old manifest+cap run: only b, not all manifest train IDs.
        saved = metadata['data_recipe']['split_manifest']
        saved['actual_train_ids'] = ['b']; saved['train_mask'] = [False,True,False,False]
        metadata['train_episode_ids'] = [1]; metadata['train_source_rows'] = list(range(6,12))
        metadata['resolved_dataset'].update(max_train_episodes=1, val_ratio=.8)
    model = VQVAEHand(12, [1.] * 12, latent_dim=4, hidden_dim=8, num_groups=1,
                      codebook_size=2, num_layers=1, kmeans_init=False)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = tmp_path/'vq.pt'
    vq._save_checkpoint(checkpoint, epoch=0, vqvae=model, optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1), normalizer=normalizer,
        args=argparse.Namespace(vq_decay=.8,threshold_ema_dead_code=0,kmeans_iters=2,action_key='action',tcp_dim=7),
        train_history=[], split_metadata=metadata, metrics={})
    return checkpoint, metadata


@pytest.mark.parametrize('capped', [False,True])
@pytest.mark.parametrize('external', ['deleted','modified'])
def test_usage_restores_saved_split_without_external_manifest(policy_config,tmp_path,capped,external):
    from scripts.training.measure_vq_usage import measure
    checkpoint,metadata = usage_checkpoint(policy_config,tmp_path,manifest=True,capped=capped)
    path = tmp_path/'split.json'
    if external == 'deleted': path.unlink()
    else: path.write_text('{}')
    before = checkpoint.read_bytes()
    for split in ('train','validation'):
        usage = measure(str(checkpoint),policy_config.dataset.zarr_path,split=split)
        rows = metadata['train_source_rows' if split=='train' else 'val_source_rows']
        assert sum(usage['nn_counts']) == len(rows)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize('change', ['revision','missing_revision','observation','dispatch','rows', 'val_rows',
                                   'digest','mask','missing_saved'])
def test_usage_rejects_changed_identity_or_split(policy_config,tmp_path,change):
    from scripts.training.measure_vq_usage import measure
    explicit = change in {'digest','mask','missing_saved'}
    checkpoint,_ = usage_checkpoint(policy_config,tmp_path,manifest=explicit)
    root = zarr.open_group(policy_config.dataset.zarr_path, mode='a')
    if change == 'revision': root.attrs['data_revision'] = 'changed'
    elif change == 'missing_revision': del root.attrs['data_revision']
    elif change == 'observation': root['data/joint_state'][8:12] = np.nan
    elif change == 'dispatch': root['row_info/dispatch_status'][8:12] = 0
    else:
        payload = torch.load(checkpoint,weights_only=False)
        metadata = payload['split_metadata']
        if change == 'rows': metadata['train_source_rows'] = []
        elif change == 'val_rows': metadata['val_source_rows'] = []
        elif change == 'missing_saved': del metadata['data_recipe']['split_manifest']
        elif change == 'digest': metadata['data_recipe']['split_manifest']['sha256'] = 'bad'
        elif change == 'mask': metadata['data_recipe']['split_manifest']['train_mask'][0] = False
        torch.save(payload,checkpoint)
    with pytest.raises((ValueError,hydra.errors.InstantiationException)):
        measure(str(checkpoint),policy_config.dataset.zarr_path)


def test_usage_no_manifest_historical_identity_is_unverified(policy_config,tmp_path):
    from scripts.training.measure_vq_usage import measure
    checkpoint,metadata = usage_checkpoint(policy_config,tmp_path)
    payload = torch.load(checkpoint,weights_only=False)
    del payload['split_metadata']['data_revision']; torch.save(payload,checkpoint)
    with pytest.warns(UserWarning, match='身份未验证'):
        result = measure(str(checkpoint),policy_config.dataset.zarr_path)
    assert sum(result['nn_counts']) == len(metadata['train_source_rows'])


def test_codebook_export_keeps_exclusive_atomic_publication(policy_config,tmp_path,monkeypatch):
    from scripts.training.extract_vq_codebook import extract_codebook
    from dexmani_policy.agents.vq_hand import CodebookManager
    checkpoint,_ = usage_checkpoint(policy_config,tmp_path)
    output = tmp_path/'codebook.npz'
    extract_codebook(checkpoint,output,device='cpu')
    before = output.read_bytes()
    with pytest.raises(FileExistsError): extract_codebook(checkpoint,output,device='cpu')
    def fail_save(self,path):
        assert path.name.startswith('.dexmani-publish-') and path.name.endswith('.tmp.npz')
        path.write_bytes(b'partial export')
        raise OSError('export interrupted')
    monkeypatch.setattr(CodebookManager,'save',fail_save)
    with pytest.raises(OSError): extract_codebook(checkpoint,output,device='cpu',overwrite=True)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob('.dexmani-publish-*'))


@pytest.fixture
def policy_config(tmp_path):
    path = tmp_path / "data.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["domain"] = "real"
    root.attrs["dt"] = 0.1
    root.create_dataset("row_info/observation_timestamp_ns", data=(np.arange(24, dtype="i8") + 1) * 100_000_000)
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
    from unittest.mock import patch
    from dexmani_policy.training.logging import to_log_scalars
    with patch.object(torch.Tensor, 'item', side_effect=AssertionError('eager scalar conversion')):
        loss, metrics = agent(batch)
    for key in ('batch_nn_code_entropy', 'batch_nn_code_used_1pct'):
        assert torch.is_tensor(metrics[key]) and not metrics[key].requires_grad
    logged = to_log_scalars(metrics)
    # Identical hand targets select one prototype for the entire batch.
    assert logged['batch_nn_code_entropy'] == 0.0
    assert logged['batch_nn_code_used_1pct'] == 1
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



@pytest.mark.parametrize('legacy', [False, True])
def test_usage_restores_time_rule_and_old_source_rows(policy_config,tmp_path,legacy):
    from scripts.training.measure_vq_usage import measure
    root=zarr.open_group(policy_config.dataset.zarr_path,mode='a')
    stamps=root['row_info/observation_timestamp_ns'][:]
    stamps[9:] += 1_000_000_000
    root['row_info/observation_timestamp_ns'][:]=stamps
    policy_config.dataset.max_time_gap_ratio=None if legacy else 1.5
    checkpoint, metadata=usage_checkpoint(policy_config,tmp_path)
    if legacy:
        payload=torch.load(checkpoint,weights_only=False)
        del payload['split_metadata']['data_recipe']['time_filter']
        del payload['split_metadata']['resolved_dataset']['max_time_gap_ratio']
        torch.save(payload,checkpoint)
    result=measure(str(checkpoint),policy_config.dataset.zarr_path)
    assert sum(result['nn_counts'])==len(metadata['train_source_rows'])
