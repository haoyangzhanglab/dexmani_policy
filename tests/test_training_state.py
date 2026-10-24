import copy
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import load_training_checkpoint
from dexmani_policy.smoke_test import validate_checkpoint_state
from dexmani_policy.training.builder import TrainingComponents, build_trainer
from dexmani_policy.training.common.ema import EMA
from dexmani_policy.training.data import build_train_loader
from dexmani_policy.training.resume import restore_model_weights, restore_training_state
from dexmani_policy.training.workspace import Workspace


class RegressionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.seen = []

    def get_training_loss_kwargs(self, ema_model):
        return {}

    def validate_normalizer(self):
        pass

    def forward(self, batch, *, log_metrics=True):
        x = batch["obs"]["state"]
        self.seen.extend(x.flatten().tolist())
        prediction = self.linear(x + torch.rand_like(x) * 0.01)
        loss = (prediction - batch["action"]).square().mean()
        return loss, {"loss": loss.detach()} if log_metrics else {}


def regression_run(output_dir, *, total_steps=7, length=10, rank=0, world_size=1):
    torch.manual_seed(42)
    cfg = OmegaConf.create({
        "training": {
            "seed": 42,
            "loop": {
                "total_train_steps": total_steps,
                "log_interval_steps": 1,
                "gradient_accumulation_steps": 2,
            },
        },
        "dataloader": {"batch_size": 2, "shuffle": False, "num_workers": 0, "drop_last": False},
    })
    dataset = [
        {"obs": {"state": torch.tensor([index / 10])}, "action": torch.tensor([index / 20])}
        for index in range(length)
    ]
    model = RegressionPolicy()
    ema_model = copy.deepcopy(model)
    updater = EMA(ema_model)
    optimizer = torch.optim.AdamW([{
        "params": list(model.parameters()),
        "param_names": [name for name, _ in model.named_parameters()],
    }], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step))
    components = TrainingComponents(
        device=torch.device("cpu"), model=model, ema_model=ema_model, ema_updater=updater,
        optimizer=optimizer, scheduler=scheduler,
        train_loader=build_train_loader(cfg, dataset, rank=rank, world_size=world_size),
        workspace=Workspace(str(output_dir)) if rank == 0 else None,
    )
    return cfg, components


class TrainingStateTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_components(self, name, **kwargs):
        cfg, components = regression_run(self.root / name, **kwargs)
        self.addCleanup(components.close)
        return cfg, components, build_trainer(cfg, components)

    def assert_model_equal(self, actual, expected):
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)

    def test_resume_matches_continuous_across_accumulation_and_epoch(self):
        cfg, expected, trainer = self.run_components("continuous")
        trainer.train()
        expected_rng = torch.rand(4)
        for stop in (1, 2, 3, 4):
            with self.subTest(stop=stop):
                cfg, first, first_trainer = self.run_components(f"first-{stop}")
                first_trainer.train(max_updates=stop)
                checkpoint = load_training_checkpoint(first.workspace.checkpoint_dir / "latest.pt")
                cfg, resumed, resumed_trainer = self.run_components(f"resume-{stop}")
                restore_model_weights(checkpoint, resumed.model, resumed.ema_model, resumed.device)
                state = restore_training_state(checkpoint, resumed, effective_config=cfg)
                resumed_trainer.train(resume_state=state)
                self.assert_model_equal(resumed.model, expected.model)
                self.assert_model_equal(resumed.ema_model, expected.ema_model)
                self.assertEqual(first.model.seen + resumed.model.seen, expected.model.seen)
                self.assertEqual(resumed.ema_updater.optimization_step, 7)
                self.assertEqual(resumed.scheduler.state_dict(), expected.scheduler.state_dict())
                self.assertEqual(
                    (resumed_trainer.current_epoch, resumed_trainer.next_micro_step), (2, 2),
                )
                torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)

    def test_tail_accumulation_matches_effective_batches(self):
        cfg, actual, trainer = self.run_components("accumulated", total_steps=3, length=9)
        initial_rng = torch.get_rng_state()
        reference = copy.deepcopy(actual.model)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step))
        trainer.train()
        torch.set_rng_state(initial_rng)
        for indices in (range(0, 4), range(4, 8), range(8, 9)):
            x = torch.tensor([[index / 10] for index in indices])
            batch = {"obs": {"state": x}, "action": x / 2}
            loss, _ = reference(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        for actual_parameter, expected_parameter in zip(
            actual.model.parameters(), reference.parameters(),
        ):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-6, atol=1e-7)
        self.assertEqual((trainer.current_epoch, trainer.next_micro_step), (1, 0))

    def test_checkpoint_collisions_and_invalid_resume(self):
        cfg, components, trainer = self.run_components("milestones", total_steps=2)
        trainer.train(max_updates=1)
        paths = list(components.workspace.checkpoint_dir.glob("epoch=*.pt"))
        self.assertEqual(len(paths), 1)
        self.assertIn("milestone=40pct", paths[0].name)
        checkpoint = load_training_checkpoint(paths[0])
        cfg, resumed, _ = self.run_components("invalid", total_steps=2)
        checkpoint.next_micro_step = 1
        with self.assertRaisesRegex(ValueError, "accumulation boundary"):
            restore_training_state(checkpoint, resumed, effective_config=cfg)
        checkpoint.next_micro_step = 2
        checkpoint.resume_contract["dataset_length"] += 1
        with self.assertRaisesRegex(ValueError, "dataset_length"):
            restore_training_state(checkpoint, resumed, effective_config=cfg)

    def test_workspace_refuses_existing_training_artifacts(self):
        for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
            with self.subTest(name=name):
                directory = self.root / name
                directory.mkdir()
                artifact = directory / name
                artifact.write_text("existing artifact")
                with self.assertRaises(FileExistsError):
                    Workspace(str(directory))
                self.assertEqual(artifact.read_text(), "existing artifact")

    def test_nonfinite_loss_does_not_update_parameters(self):
        cfg, components, trainer = self.run_components("nonfinite")
        before = copy.deepcopy(components.model)
        batch = {"obs": {"state": torch.zeros(2, 1)}, "action": torch.full((2, 1), float("nan"))}
        with self.assertRaisesRegex(RuntimeError, "Non-finite loss"):
            trainer.train_microbatch(batch)
        self.assert_model_equal(components.model, before)
        self.assertEqual(components.ema_updater.optimization_step, 0)

    def test_smoke_checkpoint_comparison_covers_raw_and_ema(self):
        cfg, components, trainer = self.run_components("smoke", total_steps=2)
        trainer.train(max_updates=1)
        path = components.workspace.checkpoint_dir / "latest.pt"
        validate_checkpoint_state(cfg, components, trainer, path)
        for model in (components.model, components.ema_model):
            with self.subTest(model=model):
                parameter = next(model.parameters())
                before = parameter.detach().clone()
                with torch.no_grad():
                    parameter.add_(1)
                with self.assertRaises(AssertionError):
                    validate_checkpoint_state(cfg, components, trainer, path)
                with torch.no_grad():
                    parameter.copy_(before)


if __name__ == "__main__":
    unittest.main()
