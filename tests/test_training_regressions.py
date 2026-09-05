import contextlib
import unittest
from unittest import mock

import torch

from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.training.build_utils import validate_gradient_accumulation
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


if __name__ == "__main__":
    unittest.main()
