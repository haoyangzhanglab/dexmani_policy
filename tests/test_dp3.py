from contextlib import closing

import hydra
import numpy as np
import pytest
import torch

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.deployment import load_policy
from dexmani_policy.training.config import load_policy_config
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer

from tests.helpers import check_recipe_composition


def test_recipe_composition():
    check_recipe_composition("dp3")


def test_external_recipe_composition(tmp_path):
    (tmp_path / "external.yaml").write_text("defaults: [dp3, _self_]\npolicy_name: external\n")
    cfg = load_policy_config("external.yaml", ["horizon=8"], config_dir=tmp_path)
    assert cfg.policy_name == "external"
    assert cfg.agent.horizon == cfg.dataset.horizon == 8


def test_ddim_recipe_rejects_one_step():
    cfg = load_policy_config("dp3", ["agent.action_dim=19", "agent.state_dim=19", "agent.num_inference_steps=1"])
    with pytest.raises(hydra.errors.InstantiationException) as error:
        hydra.utils.instantiate(cfg.agent)
    assert isinstance(error.value.__cause__, ValueError)
    assert "leading DDIM" in str(error.value.__cause__)


def test_training_restore_and_prediction(tmp_path, make_demo):
    directory = tmp_path / "dp3"
    cfg = load_policy_config("dp3", [
        f"zarr_path={make_demo()}", "task_name=test", f"workspace.output_dir={directory}",
        "workspace.wandb_cfg=null", "training.device=cpu", "training.use_compile=false",
        "training.loop.total_train_steps=3", "dataloader.num_workers=0", "dataloader.batch_size=2",
        "dataset.augmentation_cfg=null", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.down_dims=[16,32]", "agent.num_points=8",
    ])
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
        obs = {key: value.numpy() for key, value in trainer.train_loader.dataset[0]["obs"].items()}
    saved = load_checkpoint(directory / "checkpoints/latest.pt")
    with closing(load_policy(directory, device="cpu", inference_steps=2)) as runtime:
        torch.testing.assert_close(runtime.agent.state_dict(), saved.ema_model_state)
        prediction = runtime.predict(obs)
        assert prediction.shape == (4, 19) and np.isfinite(prediction).all()
    with closing(prepare_resume(directory)) as resumed:
        torch.testing.assert_close(resumed.model.state_dict(), saved.model_state)
        resumed.train(max_updates=1)
        assert resumed.global_step == 2
