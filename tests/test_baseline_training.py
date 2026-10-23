"""Synthetic CPU integration fixtures, not real-data/codebook experiment validation."""

import copy
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
from dexmani_policy.common.checkpoint import load_training_checkpoint, restore_policy_agent
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.deployment.metadata import inspect_policy
from dexmani_policy.deployment.runtime import PolicyRuntime, load_policy
from dexmani_policy.smoke_test import (
    close_data_loader,
    validate_checkpoint,
    validate_parameter_updates,
    validate_prediction_roundtrip,
)
from dexmani_policy.training.builder import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_trainer,
    build_training_components,
    validate_hand_pretraining,
)
from dexmani_policy.training.common.hand_representation import describe_hand_data
from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_policy_data
from dexmani_policy.training.validation import validate_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("name", ["sat", "dqrise"])
def test_synthetic_cpu_training_checkpoint_and_action_contracts(name, tmp_path):
    rng = np.random.default_rng(42)
    path = tmp_path / "synthetic.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["dt"] = 0.05
    root.create_dataset("meta/episode_ends", data=np.array([24, 48], dtype=np.int64))
    for key in ("joint_state", "action"):
        values = rng.uniform(-1, 2, (48, 19)).astype("float32")
        values[24:] += 3  # Distinct episode extrema expose the full-buffer statistics.
        root.create_dataset("data/" + key, data=values)
    root.create_dataset("data/point_cloud", data=rng.uniform(0, 1, (48, 64, 6)).astype("float32"))
    cfg = load_policy_config(
        name,
        [
            f"zarr_path={path}",
            "dataset.val_ratio=0.5",
            "dataset.max_train_episodes=null",
            "dataloader.batch_size=2",
            "dataloader.num_workers=0",
            "dataloader.persistent_workers=false",
            "dataloader.prefetch_factor=null",
            "dataloader.pin_memory=false",
            "training.device=cpu",
            "training.use_bfloat16=false",
            "training.use_compile=false",
            "training.lr_warmup_steps=0",
            "training.loop.total_train_steps=10",
            "training.loop.log_interval_steps=1",
            f"workspace.output_dir={tmp_path / 'run'}",
            "workspace.wandb_cfg=null",
            "agent.num_points=32",
            "agent.pc_encoder_config.channels=48",
            "agent.pc_encoder_config.num_patches=8",
            "agent.pc_encoder_config.group_size=8",
            "agent.pc_encoder_config.depth=1",
            "agent.num_inference_steps=2",
            *(["agent.n_layers=2"] if name == "sat" else ["agent.down_dims=[32,64]"]),
        ],
    )
    validate_config(cfg)
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    for statistic in ("min", "max"):
        expected = getattr(root["data/action"][:], statistic)(axis=0)
        torch.testing.assert_close(
            normalizer["action"].input_stats[statistic], torch.from_numpy(expected)
        )
    assert cfg.data_recipe[0].normalization.population == "full_replay_buffer"
    if name == "dqrise":
        empty = hydra.utils.instantiate(cfg.agent)
        with pytest.raises(ValueError, match="codebook_path"):
            empty.initialize_training()
        with pytest.raises(RuntimeError, match="no hand codebook"):
            empty.predict_action_from_condition(torch.zeros(1, 256))
        # Exercise VQ's production data/normalizer preparation without training a VQ model.
        _, _, vq_normalizer, vq_metadata = prepare_policy_data(cfg)
        for key in ("scale", "offset"):
            torch.testing.assert_close(
                vq_normalizer["hand"].params_dict[key], normalizer["action"].params_dict[key][7:]
            )
        assert vq_metadata["hand_data"] == describe_hand_data(dataset)
        codebook = HandStateCodebook(12)
        codebook.set_buffers(
            {
                "sorted_hand_poses": torch.linspace(-1, 1, 16).unsqueeze(1).repeat(1, 12),
                "pca_permutation": torch.arange(16),
                "layer_weights": torch.tensor([0.5, 0.5]),
                "hand_normalizer_scale": normalizer["action"].params_dict["scale"][7:],
                "hand_normalizer_offset": normalizer["action"].params_dict["offset"][7:],
            }
        )
        codebook.artifact_metadata = {"hand_data": describe_hand_data(dataset)}
        cfg.codebook_path = str(tmp_path / "synthetic_codebook.npz")
        codebook.save(cfg.codebook_path)
        continuous, indices = codebook.encode(codebook.sorted_hand_poses)
        decoded, decoded_indices = codebook.decode(continuous.squeeze(-1))
        torch.testing.assert_close(decoded, codebook.sorted_hand_poses, rtol=0, atol=0)
        assert torch.equal(indices, torch.arange(16)) and torch.equal(indices, decoded_indices)
        # An EEF artifact is rejected even if all hand affine parameters happen to match.
        codebook.artifact_metadata["hand_data"]["action_key"] = "action_ee"
        wrong_path = tmp_path / "synthetic_eef.npz"
        codebook.save(wrong_path)
        empty.codebook_path = str(wrong_path)
        with pytest.raises(ValueError, match="action_key=action"):
            empty.initialize_training()

    components = None
    try:
        components = build_training_components(cfg, torch.device("cpu"), dataset, normalizer)
        model = components.model
        assert model.obs_encoder.out_dim == 256
        assert cfg.agent.pc_encoder_config.fps_random_config == cfg.agent.fps_random_config
        assert cfg.agent.pc_encoder_config.depth == 1
        trainer = build_trainer(cfg, components)
        components.workspace.save_hydra_config(cfg)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        trainer.train(max_updates=4)
        validate_parameter_updates(model, before)
        for prefix in (
            "obs_encoder.pc_encoder.",
            "obs_encoder.state_mlp.",
            "obs_encoder.temporal_projection.",
        ):
            assert any(
                not torch.equal(parameter, before[name])
                for name, parameter in model.named_parameters()
                if name.startswith(prefix)
            )
        checkpoint_path = components.workspace.checkpoint_dir / "latest.pt"
        validate_checkpoint(cfg, components, trainer, checkpoint_path)
        if name == "dqrise":
            Path(cfg.codebook_path).unlink()  # Only this test's temporary synthetic NPZ.
        validate_prediction_roundtrip(cfg, components, trainer, checkpoint_path)
        checkpoint = load_training_checkpoint(checkpoint_path)
        resumed, resumed_ema, _ = build_model_and_ema(
            cfg, torch.device("cpu"), None, dataset=dataset, checkpoint=checkpoint
        )
        for actual, state in (
            (resumed, checkpoint.model_state),
            (resumed_ema, checkpoint.ema_model_state),
        ):
            for key, value in actual.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        restored = restore_policy_agent(None, checkpoint_path, use_ema=False, device="cpu")
        obs = {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}
        torch.manual_seed(123)
        output = restored.predict_action(obs)
        torch.manual_seed(123)
        repeated = restored.predict_action(obs)
        assert output["pred_action"].shape == (1, 16, 19)
        assert output["control_action"].shape == (1, 8, 19)
        torch.testing.assert_close(output["pred_action"], repeated["pred_action"], rtol=0, atol=0)
        torch.testing.assert_close(
            output["control_action"], output["pred_action"][:, 1:9], rtol=0, atol=0
        )
        info = inspect_policy(tmp_path / "run", checkpoint="latest", weights="raw")
        runtime = load_policy(info, device="cpu", seed=123)
        try:
            observation = {key: value.squeeze(0).numpy() for key, value in obs.items()}
            future = runtime.predict(observation)
            np.testing.assert_allclose(future, output["pred_action"][0, 1:].detach().numpy())
            assert future.shape == (15, 19)
            durations = runtime.warmup(samples=1)
            assert len(durations["steady"]) == 1
            assert durations["steady"][0] > 0
        finally:
            runtime.close()
        wrong_config = OmegaConf.create(OmegaConf.to_container(info.saved_config, resolve=True))
        wrong_config.dataset.pc_dim = 3
        wrong_info = type(info)(info.experiment_dir, info.checkpoint_path, wrong_config, "raw", 2)
        with pytest.raises(ValueError, match="point-cloud channels disagree"):
            PolicyRuntime(restored, wrong_info, device="cpu", seed=123)

        normalized = restored.normalizer["action"].normalize(output["pred_action"])
        condition = restored.encode_observation(obs)
        assert condition.shape == (1, 256)
        torch.manual_seed(123)
        if name == "sat":
            sampled = restored.action_decoder.predict_action(
                condition.unsqueeze(1), torch.zeros(1, 19, 16)
            ).transpose(1, 2)
            torch.testing.assert_close(normalized, sampled)
        else:
            assert restored.diffusion_action_dim == 8
            sampled = restored.action_decoder.predict_action(condition, torch.zeros(1, 16, 8))
            torch.testing.assert_close(normalized[..., :7], sampled[..., :7])
            decoded, indices = restored.hand_codebook.decode(sampled[..., -1])
            torch.testing.assert_close(normalized[..., 7:], decoded)
            assert torch.equal(output["pred_code_index"], indices)
            assert output["pred_code_continuous"].shape == (1, 16)
            with torch.no_grad():
                restored.hand_codebook.hand_normalizer_scale.mul_(2)
            with pytest.raises(ValueError, match="do not match"):
                restored.validate_codebook_normalizer()
            broken = copy.deepcopy(checkpoint.model_state)
            broken["hand_codebook.hand_normalizer_offset"] += 1
            with pytest.raises(ValueError, match="do not match"):
                resumed.load_state_dict(broken)
            model.hand_data = {**model.hand_data, "train_episodes": []}
            with pytest.raises(ValueError, match="pretraining data mismatch"):
                validate_hand_pretraining(cfg, model, dataset)
    finally:
        if components is not None:
            close_data_loader(components.train_loader)
            components.workspace.close()
