"""Training logs use scalar metrics and a fresh W&B run per workspace."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import hydra
import torch

from dexmani_policy.training.logging import to_log_scalars
from dexmani_policy.utils.config import load_policy_config


def test_workspace_logs_scalars_to_distinct_runs(tmp_path, monkeypatch):
    runs = [SimpleNamespace(log=Mock(), config={}, finish=Mock()) for _ in range(2)]
    initialize = Mock(side_effect=runs)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    cfg = load_policy_config("ddp/dqrise")
    cfg.workspace.wandb_cfg.name = "repeated_name"
    metrics = to_log_scalars({"train/loss": torch.tensor(.5), "train/video_loss": torch.tensor(.25)})
    for index, run in enumerate(runs):
        cfg.workspace.output_dir = str(tmp_path / str(index) / "same_basename")
        workspace = hydra.utils.instantiate(cfg.workspace)
        try:
            workspace.save_hydra_config(cfg)
            workspace.log(metrics, step=7)
            run.log.assert_called_once_with(metrics, step=7)
            assert run.config["output_dir"] == str(workspace.output_dir)
            record = json.loads((workspace.output_dir / "metrics.jsonl").read_text())
            assert record == {"step": 7, "train/loss": .5, "train/video_loss": .25}
        finally:
            workspace.close()
        run.finish.assert_called_once()
    settings = [call.kwargs for call in initialize.call_args_list]
    assert len({run["id"] for run in settings}) == 2
    assert all(run["id"] and "/" not in run["id"] for run in settings)
    assert all(run["resume"] == "never" and run["group"] == "ddp/dqrise" for run in settings)
