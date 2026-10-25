import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from safetensors import SafetensorError
from safetensors.torch import save_file

from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DEncoder
from dexmani_policy.common.checkpoint import (
    TrainingCheckpoint, load_policy_state, restore_policy_agent, save_training_checkpoint,
)
from dexmani_policy.common.config import load_policy_config, resolve_normalization_spec
from dexmani_policy.datasets.common.episode_split import build_split_manifest
from dexmani_policy.training.common.logging import JsonlLogger, accumulate_metrics, to_log_scalars
from dexmani_policy.training.workspace import WandbConfig, Workspace


class FailureBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_wandb_import_and_init_failures_close_local_logger(self):
        config = WandbConfig("test", "group", "run", "offline")
        for stage, module, error in (
            ("import", None, ModuleNotFoundError),
            ("init", SimpleNamespace(init=Mock(side_effect=RuntimeError("init failed"))), RuntimeError),
        ):
            with (self.subTest(stage=stage), patch.dict(sys.modules, wandb=module),
                  patch.object(JsonlLogger, "close", autospec=True, side_effect=JsonlLogger.close) as close,
                  self.assertRaises(error)):
                Workspace(self.root / stage, config)
            close.assert_called_once()
            self.assertTrue(close.call_args.args[0].file.closed)

    def test_wandb_failures_preserve_local_records_and_primary_error(self):
        config = WandbConfig("test", "group", "run", "offline")
        manifest = build_split_manifest(str(self.root / "data.zarr"), [3], {}, val_ratio=0)
        hydra_config = OmegaConf.create({"seed": 42, "data_recipe": [{"split_manifest": manifest}]})
        for stage in ("config", "log", "finish"):
            run = Mock()
            failure = RuntimeError(f"{stage} failed")
            {"config": run.config.update, "log": run.log, "finish": run.finish}[stage].side_effect = failure
            if stage != "finish":
                run.finish.side_effect = OSError("secondary close failure")
            with (self.subTest(stage=stage), redirect_stderr(io.StringIO()) as stderr,
                  patch.dict(sys.modules, wandb=SimpleNamespace(init=Mock(return_value=run))),
                  self.assertRaises(RuntimeError) as raised):
                with closing(Workspace(self.root / stage, config)) as workspace:
                    workspace.save_hydra_config(hydra_config)
                    workspace.log({"loss": 1.5}, step=3)
            self.assertIs(raised.exception, failure)
            self.assertTrue(workspace.json_logger.file.closed)
            run.finish.assert_called_once()
            self.assertEqual(OmegaConf.load(self.root / stage / "config.yaml").seed, 42)
            if stage != "config":
                record = json.loads((self.root / stage / "metrics.jsonl").read_text())
                self.assertEqual(record, {"step": 3, "loss": 1.5})
            if stage != "finish":
                self.assertIn("secondary close failure", stderr.getvalue())
        with (patch.dict(sys.modules, wandb=None), closing(Workspace(self.root / "disabled")) as workspace):
            workspace.log({"loss": 2.0}, step=1)

    def test_saved_config_requires_manifest_and_never_overwrites_existing_config(self):
        for name, config in (("missing_recipe", {}), ("missing_split", {"data_recipe": [{}]})):
            with self.subTest(name=name), closing(Workspace(self.root / name)) as workspace:
                with self.assertRaises(KeyError):
                    workspace.save_hydra_config(OmegaConf.create(config))
                self.assertFalse((workspace.output_dir / "config.yaml").exists())
                self.assertFalse((workspace.output_dir / "split_manifest_0.json").exists())

        manifest = build_split_manifest(str(self.root / "data.zarr"), [3], {}, val_ratio=0)
        config = OmegaConf.create({"data_recipe": [{"split_manifest": manifest}]})
        with closing(Workspace(self.root / "export")) as workspace:
            workspace.save_hydra_config(config)
            exported = json.loads((workspace.output_dir / "split_manifest_0.json").read_text())
            self.assertEqual(exported, manifest)
        with closing(Workspace(self.root / "collision")) as workspace:
            path = workspace.output_dir / "config.yaml"
            path.write_text("existing: true\n")
            with self.assertRaises(FileExistsError):
                workspace.save_hydra_config(config)
            self.assertEqual(path.read_text(), "existing: true\n")

    def test_checkpoint_requires_parameters_and_restores_saved_precision(self):
        source = torch.nn.Linear(2, 1).double()
        target = torch.nn.Linear(2, 1).float()
        state = source.state_dict()
        load_policy_state(target, state)
        torch.testing.assert_close(target.state_dict(), state, rtol=0, atol=0)
        self.assertEqual(target.weight.dtype, torch.float64)
        del state["bias"]
        with self.assertRaisesRegex(KeyError, "bias"):
            load_policy_state(target, state)

    def test_metrics_keep_scalars_and_reject_invalid_values(self):
        metrics = {"loss": torch.tensor([1.25], requires_grad=True), "count": 2, "rate": 0.5}
        totals = {}
        accumulate_metrics(totals, metrics)
        accumulate_metrics(totals, metrics)
        self.assertEqual(totals["loss"].dtype, torch.float64)
        self.assertFalse(totals["loss"].requires_grad)
        self.assertEqual(to_log_scalars(totals), {"loss": 2.5, "count": 4.0, "rate": 1.0})
        for value, error in ((torch.ones(2), RuntimeError), ("bad", ValueError), (None, TypeError)):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    accumulate_metrics({}, {"loss": value})
                with self.assertRaises(error):
                    to_log_scalars({"loss": value})

    def test_uni3d_local_weights_fail_without_downloading_or_random_fallback(self):
        directory = self.root / "pretrained"
        encoder = Uni3DEncoder(
            embed_dim=8, num_group=4, group_size=2,
            use_pretrained_weights=True, pretrained_weights_path=directory,
        )
        with patch("huggingface_hub.hf_hub_download") as download:
            with self.assertRaisesRegex(ValueError, "pretrained_weights_path is None"):
                encoder.load_pretrained_weights(None)
            with self.assertRaises(FileNotFoundError):
                encoder.initialize_training()
            self.assertFalse(directory.exists())
            directory.mkdir()
            path = directory / "model.safetensors"
            path.write_bytes(b"invalid weights")
            with self.assertRaises(SafetensorError):
                encoder.initialize_training()
            save_file({"pc_encoder.unrelated": torch.zeros(1)}, path)
            with self.assertRaisesRegex(ValueError, "different model"):
                encoder.initialize_training()
            weights = {"pc_encoder." + key: value.clone() for key, value in encoder.state_dict().items()}
            weights.pop("pc_encoder.out_proj.bias")
            weights["pc_encoder.out_proj.weight"].fill_(0.125)
            save_file(weights, path)
            encoder.initialize_training()
            torch.testing.assert_close(encoder.out_proj.weight, weights["pc_encoder.out_proj.weight"])
            download.assert_not_called()
        encoder.use_pretrained_weights = False
        path.unlink()
        with patch.object(encoder, "load_pretrained_weights") as load:
            encoder.initialize_training()
            load.assert_not_called()

    def test_r3d_raw_and_ema_restore_without_initialization_weights(self):
        cfg = load_policy_config("r3d")
        with open_dict(cfg):
            cfg.workspace.output_dir = str(self.root / "run")
            cfg.horizon, cfg.n_action_steps = 4, 2
            cfg.agent.state_out_dim = 8
            cfg.agent.embedding_dim = cfg.agent.timestep_embed_dim = 8
            cfg.agent.num_heads, cfg.agent.depth, cfg.agent.mlp_dim = 2, 1, 16
            cfg.agent.pc_encoder_config.embed_dim = 8
            cfg.agent.pc_encoder_config.num_group = 4
            cfg.agent.pc_encoder_config.group_size = 2
            cfg.agent.pc_encoder_config.pretrained_weights_path = str(self.root)
            cfg.normalization_dims = {"action": 19, "joint_state": 19, "point_cloud": 6}
        agent = hydra.utils.instantiate(cfg.agent)
        encoder = agent.obs_encoder.pc_encoder
        weights_path = self.root / "model.safetensors"
        save_file({"pc_encoder." + key: value.clone() for key, value in encoder.state_dict().items()}, weights_path)
        agent.initialize_training()
        agent.set_normalization_spec(resolve_normalization_spec(cfg), dict(cfg.normalization_dims))
        for key, width in cfg.normalization_dims.items():
            agent.normalizer.fit_field(key, [torch.arange(2 * width).reshape(2, width).float()])
        with open_dict(cfg):
            cfg.policy_metadata = agent.inference_metadata
        ema = copy.deepcopy(agent)
        with torch.no_grad():
            ema.action_decoder.model.output_proj.bias.add_(0.25)
        saved = OmegaConf.to_container(cfg, resolve=True)
        path = save_training_checkpoint(self.root / "policy.pt", TrainingCheckpoint(
            epoch=0, global_step=0, next_micro_step=0,
            model_state=agent.state_dict(), ema_model_state=ema.state_dict(),
            optimizer_state={}, scheduler_state={}, resume_contract={},
            ema_updater_step=0, rng_states=[], effective_config=saved,
        ))
        weights_path.unlink()
        observation = {"joint_state": torch.randn(1, 2, 19), "point_cloud": torch.randn(1, 2, 8, 6)}
        for use_ema, original in ((False, agent), (True, ema)):
            original.eval()
            torch.manual_seed(17)
            expected = original.predict_action(observation, inference_steps=2)
            restored = restore_policy_agent(saved, path, use_ema=use_ema, device="cpu")
            torch.testing.assert_close(restored.state_dict(), original.state_dict(), rtol=0, atol=0)
            torch.manual_seed(17)
            torch.testing.assert_close(restored.predict_action(observation, inference_steps=2), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
