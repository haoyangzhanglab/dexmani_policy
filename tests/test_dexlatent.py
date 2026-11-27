from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.obs_encoder import rgb_point
from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.training.config import load_policy_config
from dexmani_policy.training.initialization import build_policy
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer

from tests.helpers import (
    SmallRGBWeights, build_small_rgb, check_flow_recipe_defaults, check_observation_identity_constraints,
    check_recipe_composition,
)


def test_hand_affine_and_latent_order(tmp_path, make_demo, monkeypatch):
    def encode_joint_state(self, obs):
        features = self.rgb_encoder(obs["joint_state"][:, :2]).repeat(1, self.out_dim // 2)
        return features.unflatten(0, (-1, self.n_obs_steps))

    monkeypatch.setattr(rgb_point, "build_rgb_encoder", build_small_rgb)
    monkeypatch.setattr(rgb_point.RGBPointObservationEncoder, "forward", encode_joint_state)
    data_path = make_demo()
    dataset = DemonstrationDataset(data_path, sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    directory = tmp_path / "run"
    cfg = load_policy_config('dexlatent', [
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
    from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior

    prior = DexLatentHandPrior()
    path = tmp_path / "prior.pt"
    torch.save({"autoencoders": {"xarm7_xhand_right." + key: value for key, value in prior.state_dict().items()}}, path)
    cfg.initialization.dexlatent = str(path)
    affine = prior.initialize_weights(path)
    scale, offset = affine["scale"], affine["offset"]
    with patch.object(SmallRGBWeights, "initialize_weights", autospec=True,
                      side_effect=SmallRGBWeights.initialize_weights) as initialize:
        model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    assert isinstance(model.obs_encoder, rgb_point.RGBPointObservationEncoder)
    initialize.assert_called_once_with(model.obs_encoder.rgb_encoder, str(rgb_path))
    torch.testing.assert_close(model.obs_encoder.rgb_encoder.state_dict(), rgb.state_dict())
    torch.testing.assert_close(model.normalizer["action"].scale[7:], scale)
    torch.testing.assert_close(model.normalizer["action"].offset[7:], offset)
    reference = LinearNormalizer()
    chunks = [torch.from_numpy(chunk) for chunk in dataset.iter_normalization_data("action")]
    latents = [prior.encode(chunk[:, 7:] * scale + offset) for chunk in chunks]
    reference.fit_field("hand_latent", latents)
    torch.testing.assert_close(model.normalizer["hand_latent"].state_dict(), reference["hand_latent"].state_dict())
    normalized = model._encode_actions(torch.cat(chunks))
    assert normalized.shape[-1] == 39
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    saved = load_checkpoint(latest)
    assert saved.effective_config["prior_data"] == OmegaConf.to_container(cfg.prior_data)
    path.unlink()
    rgb_path.unlink()
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
    cfg = check_recipe_composition("dexlatent")
    check_flow_recipe_defaults("dexlatent", cfg)
    check_observation_identity_constraints("dexlatent", ['camera_extrinsic', 'camera_intrinsic', 'point_cloud'])
