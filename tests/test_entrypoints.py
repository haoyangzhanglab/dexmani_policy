import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from dexmani_policy import eval_pipeline, evaluate, select_checkpoint, smoke_test, train, train_ddp
from dexmani_policy.common.config import PROJECT_ROOT
from dexmani_policy.training.builder import TrainingComponents


def training_components():
    return TrainingComponents(
        device=torch.device("cpu"), model=Mock(), ema_model=Mock(), ema_updater=None,
        optimizer=Mock(), scheduler=Mock(), workspace=Mock(),
        train_loader=SimpleNamespace(num_workers=1, _iterator=Mock()),
    )


class EvaluationEntrypointsTest(unittest.TestCase):
    arguments = ["--policy-name", "ddp/dp3", "--task-name", "task", "--exp-name", "run"]

    def test_saved_config_and_explicit_overrides_reach_evaluation(self):
        saved = {"eval": {"seed": 19, "offline": {"episodes": 7}}}
        for module, operation in (
            (evaluate, "evaluate_checkpoint"),
            (select_checkpoint, "select_best_checkpoint"),
            (eval_pipeline, "run_pipeline"),
        ):
            for explicit in (False, True):
                with self.subTest(module=module.__name__, explicit=explicit):
                    arguments = self.arguments + (
                        ["--seed", "42", "--no-ema", "--inference-steps", "4",
                         "--action-steps", "2", "eval.video.fps=12"] if explicit else []
                    )
                    with patch.object(module, "load_experiment_config", return_value=saved) as load:
                        with patch.object(module, operation) as execute:
                            module.main(arguments)

                    directory = PROJECT_ROOT / "experiments/ddp/dp3/task/run"
                    load.assert_called_once_with(directory)
                    args, kwargs = execute.call_args
                    self.assertEqual(args[0], directory)
                    self.assertEqual(args[1]["eval"]["offline"]["episodes"], 7)
                    self.assertEqual(kwargs["eval_seed"], 42 if explicit else None)
                    self.assertEqual(kwargs["use_ema"], False if explicit else None)
                    self.assertEqual(kwargs["action_steps"], 2 if explicit else None)
                    key = "inference_steps_list" if module is evaluate else "inference_steps"
                    steps = [4] if module is evaluate else 4
                    self.assertEqual(kwargs[key], steps if explicit else None)
                    if explicit:
                        overrides = args[1] if module is select_checkpoint else kwargs["overrides"]
                        self.assertEqual(overrides["eval"]["video"]["fps"], 12)

    def test_selection_video_defaults(self):
        for flags, expected in (([], False), (["--videos"], True), (["--no-videos"], False)):
            with self.subTest(flags=flags):
                with patch.object(select_checkpoint, "load_experiment_config", return_value={}):
                    with patch.object(select_checkpoint, "select_best_checkpoint") as select:
                        select_checkpoint.main(self.arguments + flags)
                self.assertIs(select.call_args.kwargs["record_video"], expected)


class DistributedEntrypointsTest(unittest.TestCase):
    def setUp(self):
        resources = ExitStack()
        self.addCleanup(resources.close)
        self.cfg = OmegaConf.create({
            "training": {"seed": 42, "use_compile": True, "compile_mode": "reduce-overhead"},
            "max_updates": 2,
        })
        self.components = training_components()
        self.model = self.components.model
        self.trainer = Mock()
        resources.enter_context(patch.object(train_ddp.torch.cuda, "set_device"))
        resources.enter_context(patch.object(train_ddp, "set_seed"))
        self.initialize = resources.enter_context(patch.object(train_ddp.dist, "init_process_group"))
        self.destroy = resources.enter_context(patch.object(train_ddp.dist, "destroy_process_group"))
        self.construct = resources.enter_context(patch.object(
            train_ddp, "build_training_components", return_value=self.components,
        ))
        self.wrap = resources.enter_context(patch.object(
            train_ddp, "DDP", return_value=SimpleNamespace(module=self.model),
        ))
        resources.enter_context(patch.object(train_ddp, "build_trainer", return_value=self.trainer))
        self.load = resources.enter_context(patch.object(train_ddp, "load_training_checkpoint"))
        self.restore = resources.enter_context(patch.object(
            train_ddp, "restore_training_state", return_value=(3, 1, 0),
        ))

    def test_fresh_run_synchronizes_ema_and_uses_selected_devices(self):
        iterator = self.components.train_loader._iterator
        train_ddp.train_distributed(0, self.cfg, [2, 3], [], None)
        self.initialize.assert_called_once_with(
            backend="nccl", init_method="env://", world_size=2, rank=0,
        )
        self.assertEqual(self.wrap.call_args.kwargs["device_ids"], [2])
        self.model.compile_backbone.assert_called_once_with(mode="reduce-overhead")
        self.components.ema_model.compile_backbone.assert_called_once_with(mode="reduce-overhead")
        self.components.ema_model.load_state_dict.assert_called_once_with(
            self.model.state_dict.return_value, strict=True,
        )
        self.restore.assert_not_called()
        self.trainer.train.assert_called_once_with(resume_state=None, max_updates=2)
        self.components.workspace.close.assert_called_once()
        iterator._shutdown_workers.assert_called_once()
        self.destroy.assert_called_once()

    def test_resume_preserves_ema_and_skips_disabled_compilation(self):
        self.cfg.resume_from = "source.pt"
        self.cfg.training.use_compile = False
        train_ddp.train_distributed(0, self.cfg, [0, 1], [], None)
        self.load.assert_called_once_with("source.pt")
        self.restore.assert_called_once_with(
            self.load.return_value, self.components, effective_config=self.cfg,
        )
        self.components.ema_model.load_state_dict.assert_not_called()
        self.model.compile_backbone.assert_not_called()
        self.components.ema_model.compile_backbone.assert_not_called()
        self.trainer.train.assert_called_once_with(resume_state=(3, 1, 0), max_updates=2)

    def test_non_primary_rank_needs_no_workspace_or_local_ema(self):
        self.components.workspace = None
        self.components.ema_model = None
        iterator = self.components.train_loader._iterator
        train_ddp.train_distributed(1, self.cfg, [2, 3], [], None)
        self.assertEqual(self.construct.call_args.kwargs["rank"], 1)
        self.assertEqual(self.wrap.call_args.kwargs["device_ids"], [3])
        self.model.compile_backbone.assert_called_once_with(mode="reduce-overhead")
        iterator._shutdown_workers.assert_called_once()
        self.destroy.assert_called_once()

    def test_process_group_closes_on_construction_training_and_cleanup_errors(self):
        for operation in (self.construct, self.trainer.train, self.components.workspace.close):
            with self.subTest(operation=operation):
                self.components.model = self.model
                self.destroy.reset_mock()
                operation.side_effect = RuntimeError("failure")
                with self.assertRaisesRegex(RuntimeError, "failure"):
                    train_ddp.train_distributed(0, self.cfg, [0, 1], [], None)
                self.destroy.assert_called_once()
                operation.side_effect = None

    def test_device_assignment_rejects_invalid_or_overlapping_ranks(self):
        self.assertEqual(train_ddp.resolve_gpu_ids(2, None, 4), [0, 1])
        self.assertEqual(train_ddp.resolve_gpu_ids(2, [3, 1], 4), [3, 1])
        for count, devices in ((1, None), (2.0, None), (2, [0]), (2, [0, 0]),
                               (2, [0, 4]), (2, [-1, 0]), (2, [True, 0])):
            with self.subTest(count=count, devices=devices), self.assertRaises(ValueError):
                train_ddp.resolve_gpu_ids(count, devices, 4)
        with self.assertRaises(ValueError):
            train_ddp.resolve_gpu_ids(2, None, 0)


class TrainingEntrypointsTest(unittest.TestCase):
    def test_single_device_workers_and_workspace_close_on_training_error(self):
        cfg = OmegaConf.create({"training": {"seed": 42, "device": "cuda:0"}})
        components = training_components()
        iterator = components.train_loader._iterator
        trainer = Mock()
        trainer.train.side_effect = RuntimeError("training failed")
        with ExitStack() as resources:
            resources.enter_context(patch.object(train.torch.cuda, "is_available", return_value=True))
            resources.enter_context(patch.object(train, "set_seed"))
            resources.enter_context(patch.object(train, "prepare_training", return_value=(cfg, None)))
            resources.enter_context(patch.object(
                train, "build_dataset_and_normalizer", return_value=([], None),
            ))
            resources.enter_context(patch.object(
                train, "build_training_components", return_value=components,
            ))
            resources.enter_context(patch.object(train, "build_trainer", return_value=trainer))
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                train.main.__wrapped__(cfg)
        components.workspace.close.assert_called_once()
        iterator._shutdown_workers.assert_called_once()

    def test_smoke_closes_workspace_and_tempdir_even_if_workers_fail_to_close(self):
        cfg = OmegaConf.create({
            "training": {"seed": 42, "device": "cuda:0"}, "workspace": {},
        })
        iterator = Mock()
        iterator._shutdown_workers.side_effect = RuntimeError("worker cleanup failed")
        loader = SimpleNamespace(num_workers=1, _iterator=iterator)
        components = training_components()
        components.train_loader = loader
        with ExitStack() as resources:
            resources.enter_context(patch.object(
                smoke_test.torch.cuda, "is_available", return_value=True,
            ))
            resources.enter_context(patch.object(smoke_test, "set_seed"))
            resources.enter_context(patch.object(
                smoke_test, "build_dataset_and_normalizer", return_value=([], None),
            ))
            resources.enter_context(patch.object(
                smoke_test, "build_training_components", return_value=components,
            ))
            resources.enter_context(patch.object(
                smoke_test, "build_trainer", side_effect=RuntimeError("trainer failed"),
            ))
            with self.assertRaisesRegex(RuntimeError, "worker cleanup failed"):
                smoke_test.validate_training(cfg)
        iterator._shutdown_workers.assert_called_once()
        self.assertIsNone(loader._iterator)
        components.workspace.close.assert_called_once()
        self.assertFalse(Path(cfg.workspace.output_dir).exists())


if __name__ == "__main__":
    unittest.main()
