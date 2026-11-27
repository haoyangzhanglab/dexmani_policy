import copy
import json
import shutil
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, load_codebook
from dexmani_policy.agents.obs_encoder import rgb_point
from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.training.config import load_config, load_policy_config
from dexmani_policy.training.data import fit_action_normalizer, record_prior_data, use_policy_split
from dexmani_policy.training.initialization import build_policy
from dexmani_policy.training.pretrain.hand_prior.vqvae import train as train_vq
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer

from tests.helpers import (
    SmallRGBWeights, build_small_rgb, check_flow_recipe_defaults, check_observation_identity_constraints,
    check_recipe_composition,
)


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf])
def test_inherited_hand_affine_validates_full_zarr_action(make_demo, bad):
    path = make_demo()
    actions = zarr.open_group(str(path), mode="r+")["data/action"]
    if bad is not None:
        actions[6, 8] = bad
    dataset = DemonstrationDataset(path, sensor_modalities=["joint_state"], obs_horizon=1)
    normalizer = LinearNormalizer({"action": (19,)})
    hand = {"scale": torch.linspace(0.5, 1.5, 12), "offset": torch.linspace(-0.2, 0.3, 12)}
    if bad is not None:
        with pytest.raises(ValueError, match=r"field=action, episode=0, source=0, row=6, channel=\(8,\)") as error:
            fit_action_normalizer(dataset, normalizer, hand_affine=hand)
        assert str(path) in str(error.value)
        return
    fit_action_normalizer(dataset, normalizer, hand_affine=hand)
    field = normalizer["action"]
    torch.testing.assert_close(field.scale[7:], hand["scale"])
    torch.testing.assert_close(field.offset[7:], hand["offset"])
    raw = torch.from_numpy(actions[:])
    coded = field.normalize(raw)
    prior = HandStateCodebook(hand_dim=12, num_codes=len(raw))
    prior.sorted_hand_poses.copy_(raw[:, 7:] * hand["scale"] + hand["offset"])
    reconstructed = torch.cat((coded[:, :7], prior.decode(prior.encode(coded[:, 7:]).squeeze(-1))), -1)
    torch.testing.assert_close(field.unnormalize(reconstructed), raw)


def test_same_source_pretraining_carries_real_group_budget(tmp_path, make_demo):
    path = make_demo()
    data = zarr.open_group(str(path), mode="a")
    data["meta/episode_ends"].resize(6)
    data["meta/episode_ends"][:] = [2, 4, 6, 8, 10, 12]
    sources = ["a", "a", "b", "c", "d", "e"]
    data.attrs.update(format="dexmani.real.canonical", source_episodes=sources)
    policy = load_policy_config("dqrise", [
        f"zarr_path={path}", "task_name=test", "dataset.seed=9",
        "dataset.val_episodes=1", "dataset.max_train_episodes=2",
        f"workspace.output_dir={tmp_path / 'policy'}",
    ])
    policy_path = tmp_path / "policy.yaml"
    # Hydra --cfg job prints the recipe before runtime assembly fills dataset.zarr_path.
    policy_recipe = OmegaConf.to_container(policy, resolve=True)
    del policy_recipe["dataset"]["zarr_path"]
    OmegaConf.save(OmegaConf.create(policy_recipe), policy_path)
    kwargs = {"seed": 9, "val_episodes": 1, "max_train_episodes": 2}
    dataset = DemonstrationDataset(path, sensor_modalities=["joint_state"], obs_horizon=1, **kwargs)
    chosen = set(np.asarray(sources)[dataset.train_mask])
    held_out = set(np.asarray(sources)[dataset.val_mask])
    assert len(chosen) == 2 and len(held_out) == 1 and chosen.isdisjoint(held_out)
    assert dataset.train_mask[0] == dataset.train_mask[1]

    run = tmp_path / "vq"
    cfg = load_config("pretrain/hand_prior/vqvae", [
        f"workspace.output_dir={run}", "vq_vae.device=cpu", "vq_vae.batch_size=2",
        "vq_vae.codebook_size=2", "vq_vae.num_groups=2", "vq_vae.num_epochs=1",
    ])
    use_policy_split(cfg, policy_path)
    train_vq(cfg)
    _, affine, record = load_codebook(run / "codebook.npz")
    assert record["source"] == str(path)
    assert set(record["train_source_episodes"]) == chosen
    assert set(record["val_source_episodes"]) == held_out
    rows = np.repeat(dataset.train_mask, 2)
    hand = torch.from_numpy(data["data/action"][:][rows, 7:])
    torch.testing.assert_close(affine["scale"], 2 / (hand.amax(0) - hand.amin(0)))
    record_prior_data(policy, dataset, "codebook.npz", record)
    assert policy.prior_data[0].budget_verified
    assert "training_data" not in policy
    outside_budget = set(sources) - chosen - held_out
    for group, key in [(next(iter(outside_budget)), "train_source_episodes"),
                       (next(iter(held_out)), "train_source_episodes"),
                       (next(iter(outside_budget)), "val_source_episodes")]:
        with pytest.raises(ValueError, match="same-source prior exceeds"):
            record_prior_data(policy, dataset, "leaky.pt", dict(record, **{key: [group]}))
    copied = tmp_path / "copy" / path.name
    shutil.copytree(path, copied)
    record_prior_data(policy, dataset, "external.pt", dict(record, source=str(copied)))
    record_prior_data(policy, dataset, "unknown.pt", {"source": None})
    assert not policy.prior_data[-2].budget_verified
    assert policy.prior_data[-1] == {"source": None, "artifact": "unknown.pt", "budget_verified": False}


def test_vq_artifact_paths(output_path, make_demo):
    value, expected = output_path
    data = make_demo()
    cfg = load_config("pretrain/hand_prior/vqvae", [
        f"zarr_path={data.name}", f"workspace.output_dir='{value}'",
        "vq_vae.device=cpu", "vq_vae.num_epochs=1", "vq_vae.save_epochs=1",
        "vq_vae.batch_size=12", "vq_vae.warmup_steps=0", "vq_vae.hidden_dim=16",
        "vq_vae.latent_dim=8", "vq_vae.num_layers=1",
    ])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    actions = zarr.open_group(str(data), mode="r+")["data/action"]
    valid = actions[0, 7]
    actions[0, 7] = np.inf
    with pytest.raises(ValueError, match="Nonfinite data"):
        train_vq(cfg)
    assert not expected.exists()
    actions[0, 7] = valid
    train_vq(cfg)
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    assert torch.get_float32_matmul_precision() == "medium"
    assert cfg.workspace.output_dir == str(expected)
    saved_cfg = OmegaConf.load(expected / "config.yaml")
    assert saved_cfg.workspace.output_dir == str(expected)
    assert saved_cfg.zarr_path == saved_cfg.dataset.zarr_path == str(data)
    for name in ("metrics.jsonl", "codebook.npz", "checkpoints/last.pt"):
        assert (expected / name).exists()
    with pytest.raises(FileExistsError):
        train_vq(cfg)


def test_hand_affine_and_latent_order(tmp_path, make_demo, monkeypatch):
    def encode_joint_state(self, obs):
        features = self.rgb_encoder(obs["joint_state"][:, :2]).repeat(1, self.out_dim // 2)
        return features.unflatten(0, (-1, self.n_obs_steps))

    monkeypatch.setattr(rgb_point, "build_rgb_encoder", build_small_rgb)
    monkeypatch.setattr(rgb_point.RGBPointObservationEncoder, "forward", encode_joint_state)
    data_path = make_demo()
    dataset = DemonstrationDataset(data_path, sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    directory = tmp_path / "run"
    cfg = load_policy_config('dqrise', [
        f"zarr_path={data_path}", "task_name=test", f"workspace.output_dir={directory}",
        "workspace.wandb_cfg=null", "dataset.sensor_modalities=[joint_state]", "dataset.augmentation_cfg=null",
        "training.device=cpu", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.hidden_dim=16", "agent.n_layers=1", "agent.n_head=2", "agent.pc_encoder_config.channels=24",
        "+agent.rgb_backbone_config.architecture={model_type:dinov2}",
        "training.use_compile=false", "training.loop.total_train_steps=3",
        "dataloader.num_workers=0", "dataloader.batch_size=2",
    ])
    cfg.agent.action_dim = 19
    cfg.policy_name = "renamed_experiment"
    rgb = SmallRGBWeights()
    with torch.no_grad():
        rgb.weight.fill_(0.125)
    rgb_path = tmp_path / "rgb.pt"
    torch.save(rgb.state_dict(), rgb_path)
    cfg.initialization.rgb = str(rgb_path)
    path = tmp_path / "codebook.npz"
    poses = np.linspace(-1, 1, 36, dtype=np.float32).reshape(3, 12)
    scale, offset = torch.full((12,), 2.0), torch.full((12,), -0.5)
    training_data = json.dumps({"source": str(make_demo("codebook_source")), "train_source_episodes": [0]})
    np.savez(
        path, sorted_hand_poses=poses, hand_normalizer_scale=scale.numpy(), hand_normalizer_offset=offset.numpy(),
        training_data=training_data,
    )
    cfg.initialization.codebook = str(path)
    invalid_cfg = copy.deepcopy(cfg)
    invalid_path = tmp_path / "invalid_codebook.npz"
    np.savez(
        invalid_path, sorted_hand_poses=poses, hand_normalizer_scale=np.zeros(12, dtype=np.float32),
        hand_normalizer_offset=offset.numpy(), training_data=training_data,
    )
    invalid_cfg.initialization.codebook = str(invalid_path)
    with pytest.raises(ValueError, match="Invalid affine parameters"):
        build_policy(invalid_cfg, dataset, {"action": 19, "joint_state": 19})
    with patch.object(SmallRGBWeights, "initialize_weights", autospec=True,
                      side_effect=SmallRGBWeights.initialize_weights) as initialize:
        model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    assert isinstance(model.obs_encoder, rgb_point.RGBPointObservationEncoder)
    initialize.assert_called_once_with(model.obs_encoder.rgb_encoder, str(rgb_path))
    torch.testing.assert_close(model.obs_encoder.rgb_encoder.state_dict(), rgb.state_dict())
    torch.testing.assert_close(model.normalizer["action"].scale[7:], scale)
    torch.testing.assert_close(model.normalizer["action"].offset[7:], offset)
    assert cfg.agent.num_codes == model.hand_codebook.num_codes == 3
    torch.testing.assert_close(model.hand_codebook.sorted_hand_poses, torch.from_numpy(poses))
    assert cfg.prior_data[0].artifact == str(path)
    assert not cfg.prior_data[0].budget_verified
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    saved = load_checkpoint(latest)
    assert saved.effective_config["prior_data"] == OmegaConf.to_container(cfg.prior_data)
    path.unlink()
    rgb_path.unlink()
    cfg.agent.num_codes = None
    cfg.initialization.policy = str(latest)
    cfg.workspace.output_dir = str(tmp_path / "warm")
    for restore in (lambda: prepare_resume(directory), lambda: build_trainer(cfg)):
        with closing(restore()) as trainer:
            torch.testing.assert_close(trainer.model.state_dict(), saved.model_state)
    cfg.workspace.output_dir = str(tmp_path / "warm_bad")
    warm_data = make_demo("warm_bad")
    cfg.zarr_path = cfg.dataset.zarr_path = str(warm_data)
    zarr.open_group(str(warm_data), mode="r+")["data/action"][6, 8] = np.inf
    with pytest.raises(ValueError, match=r"field=action,.*row=6, channel=\(8,\)"):
        build_trainer(cfg)
    assert load_checkpoint(latest).global_step == 1
    assert not Path(cfg.workspace.output_dir).exists()


def test_recipe_composition():
    cfg = check_recipe_composition("dqrise")
    check_flow_recipe_defaults("dqrise", cfg)
    check_observation_identity_constraints("dqrise", ['camera_extrinsic', 'camera_intrinsic', 'point_cloud'])


@pytest.mark.parametrize("num_codes", [None, 7])
def test_warm_start_inherits_only_missing_num_codes(monkeypatch, num_codes):
    from dexmani_policy.agents.policies.dqrise import DQRISE
    from dexmani_policy.agents.policies.base import Policy
    from dexmani_policy.training import data

    cfg = OmegaConf.create({"agent": {"num_codes": num_codes}})
    checkpoint = SimpleNamespace(effective_config={"agent": {"num_codes": 5}})
    expected = object()
    monkeypatch.setattr(data, "iter_validated_actions", lambda dataset: iter(()))
    monkeypatch.setattr(Policy, "build_for_training", classmethod(lambda cls, *args, **kwargs: expected))
    assert DQRISE.build_for_training(cfg, None, {}, checkpoint=checkpoint) is expected
    assert cfg.agent.num_codes == (5 if num_codes is None else 7)
