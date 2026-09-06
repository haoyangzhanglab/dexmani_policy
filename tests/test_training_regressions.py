import contextlib
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.common import checkpoint_io
from dexmani_policy.common.checkpoint_io import TrainCheckpoint
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.training.build_utils import (
    validate_config,
    validate_gradient_accumulation,
)
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig


class _DelegatingAgent(BaseAgent):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.call = None

    def compute_loss(self, batch, **kwargs):
        self.call = (batch, kwargs)
        return "delegated"


class _TinyLoader:
    def __init__(self, values):
        self._batches = [{"x": torch.tensor(float(value))} for value in values]
        self.dataset = object()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _ForwardOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.forward_values = []
        self.no_sync_values = []
        self._inside_no_sync = False

    def forward(self, batch, **kwargs):
        value = float(batch["x"])
        self.forward_values.append(value)
        if self._inside_no_sync:
            self.no_sync_values.append(value)
        loss = self.weight * batch["x"]
        return loss, {"loss": loss.detach()}

    def compute_loss(self, batch, **kwargs):
        raise AssertionError("training must call model(...), not compute_loss(...)")

    @contextlib.contextmanager
    def no_sync(self):
        self._inside_no_sync = True
        try:
            yield
        finally:
            self._inside_no_sync = False


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=1.0)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def get_last_lr(self):
        return [1.0]


class _ResumeWorkspace:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def load_checkpoint(self, tag_or_path):
        return self.checkpoint


class _EMAUpdater:
    def __init__(self):
        self.optimization_step = 99
        self.decay = 0.1


def _resume_checkpoint(*, ema_model_state, ema_updater_step):
    return TrainCheckpoint(
        epoch=4,
        global_step=12,
        model_state={"weight": torch.tensor(2.0)},
        ema_model_state=ema_model_state,
        optimizer_state={"optimizer": "restored"},
        scheduler_state={"scheduler": "restored"},
        monitor={},
        train_params={"num_training_steps": 20},
        ema_updater_step=ema_updater_step,
        ema_decay=0.95,
        rng_state=get_rng_state(),
    )


def _build_resume_trainer(checkpoint):
    model = _ForwardOnlyModel()
    ema_model = _ForwardOnlyModel()
    ema_updater = _EMAUpdater()
    optimizer = mock.Mock()
    scheduler = mock.Mock()
    trainer = Trainer(
        device=torch.device("cpu"),
        model=model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=_TinyLoader([1]),
        workspace=_ResumeWorkspace(checkpoint),
        train_loop_cfg=TrainLoopConfig(total_train_steps=20),
        use_ema_teacher_for_consistency=False,
        num_training_steps=20,
        is_main_process=False,
    )
    return trainer, model, ema_model, ema_updater, optimizer, scheduler


def _minimal_config(*, use_ema, use_ema_teacher_for_consistency):
    return OmegaConf.create(
        {
            "horizon": 16,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "action_key": "action",
            "optimizer": {},
            "agent": {},
            "dataset": {},
            "env_runner": {"env_kwargs": {"control_mode": "joint"}},
            "training": {
                "use_ema": use_ema,
                "use_ema_teacher_for_consistency": (
                    use_ema_teacher_for_consistency
                ),
            },
        }
    )


def _build_trainer(values, accumulation_steps, total_steps, *, distributed=False):
    model = _ForwardOnlyModel()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()
    trainer = Trainer(
        device=torch.device("cpu"),
        model=model,
        ema_model=None,
        ema_updater=None,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=_TinyLoader(values),
        workspace=None,
        train_loop_cfg=TrainLoopConfig(
            total_train_steps=total_steps,
            log_interval_steps=100,
            gradient_accumulation_steps=accumulation_steps,
        ),
        use_ema_teacher_for_consistency=False,
        num_training_steps=total_steps,
        max_grad_norm=0,
        is_main_process=False,
        distributed=distributed,
    )
    return trainer, model, optimizer, scheduler


class TrainingRegressionTests(unittest.TestCase):
    def test_base_agent_forward_delegates_to_compute_loss(self):
        agent = _DelegatingAgent()
        batch = object()

        result = agent(batch, marker=3)

        self.assertEqual(result, "delegated")
        self.assertEqual(agent.call, (batch, {"marker": 3}))

    def test_tail_group_uses_its_actual_size_and_syncs_at_boundary(self):
        trainer, model, optimizer, scheduler = _build_trainer(
            range(1, 11), 4, 3, distributed=True
        )

        def all_gather(outputs, value):
            outputs[0].copy_(value)

        with mock.patch.object(
            torch.distributed, "get_world_size", return_value=1
        ), mock.patch.object(torch.distributed, "all_gather", side_effect=all_gather):
            trainer.train()

        self.assertEqual(optimizer.step_count, 3)
        self.assertEqual(scheduler.step_count, 3)
        self.assertEqual(trainer.global_step, 3)
        self.assertEqual(model.forward_values, list(range(1, 11)))
        self.assertEqual(model.no_sync_values, [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0])
        self.assertAlmostEqual(model.weight.item(), -18.5)

    def test_loader_shorter_than_accumulation_window_updates_once(self):
        trainer, model, optimizer, scheduler = _build_trainer(range(1, 4), 4, 1)

        trainer.train()

        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(scheduler.step_count, 1)
        self.assertEqual(trainer.global_step, 1)
        self.assertAlmostEqual(model.weight.item(), -2.0)

    def test_accumulation_one_preserves_one_update_per_batch(self):
        trainer, model, optimizer, scheduler = _build_trainer(range(1, 4), 1, 3)

        trainer.train()

        self.assertEqual(optimizer.step_count, 3)
        self.assertEqual(scheduler.step_count, 3)
        self.assertEqual(trainer.global_step, 3)
        self.assertAlmostEqual(model.weight.item(), -6.0)

    def test_invalid_accumulation_and_empty_loader_fail_immediately(self):
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            validate_gradient_accumulation(0, 1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            validate_gradient_accumulation(3, 0)
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            _build_trainer([], 1, 1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            _build_trainer([1], 0, 1)

    def test_missing_ema_weights_fail_before_any_resume_state_is_restored(self):
        checkpoint = _resume_checkpoint(ema_model_state=None, ema_updater_step=7)
        trainer, model, ema_model, updater, optimizer, scheduler = (
            _build_resume_trainer(checkpoint)
        )

        with self.assertRaisesRegex(RuntimeError, "ema_model_state"):
            trainer.load_for_resume("latest")

        self.assertEqual(model.weight.item(), 0.0)
        self.assertEqual(ema_model.weight.item(), 0.0)
        self.assertEqual(updater.optimization_step, 99)
        optimizer.load_state_dict.assert_not_called()
        scheduler.load_state_dict.assert_not_called()

    def test_ema_resume_state_rejects_missing_or_invalid_updater_step(self):
        invalid_steps = (None, True, -1, 1.0, "1")
        for step in invalid_steps:
            with self.subTest(step=step):
                checkpoint = _resume_checkpoint(
                    ema_model_state={"weight": torch.tensor(3.0)},
                    ema_updater_step=step,
                )
                with self.assertRaisesRegex(RuntimeError, "ema_updater_step"):
                    checkpoint_io.validate_ema_resume_state(
                        checkpoint, require_ema=True
                    )

    def test_ema_resume_state_allows_non_ema_and_complete_checkpoints(self):
        without_ema = _resume_checkpoint(
            ema_model_state=None, ema_updater_step=None
        )
        checkpoint_io.validate_ema_resume_state(without_ema, require_ema=False)

        complete = _resume_checkpoint(
            ema_model_state={"weight": torch.tensor(3.0)}, ema_updater_step=7
        )
        checkpoint_io.validate_ema_resume_state(complete, require_ema=True)

        trainer, model, ema_model, updater, optimizer, scheduler = (
            _build_resume_trainer(complete)
        )
        self.assertEqual(trainer.load_for_resume("latest"), (12, 4))
        self.assertEqual(model.weight.item(), 2.0)
        self.assertEqual(ema_model.weight.item(), 3.0)
        self.assertEqual(updater.optimization_step, 7)
        self.assertEqual(updater.decay, 0.95)
        optimizer.load_state_dict.assert_called_once_with(complete.optimizer_state)
        scheduler.load_state_dict.assert_called_once_with(complete.scheduler_state)

    def test_consistency_teacher_requires_ema(self):
        for use_ema, use_teacher in (
            (False, False),
            (True, False),
            (True, True),
        ):
            with self.subTest(use_ema=use_ema, use_teacher=use_teacher):
                validate_config(
                    _minimal_config(
                        use_ema=use_ema,
                        use_ema_teacher_for_consistency=use_teacher,
                    )
                )

        with self.assertRaisesRegex(
            ValueError, "use_ema_teacher_for_consistency=true requires.*use_ema=true"
        ):
            validate_config(
                _minimal_config(
                    use_ema=False, use_ema_teacher_for_consistency=True
                )
            )


if __name__ == "__main__":
    unittest.main()
