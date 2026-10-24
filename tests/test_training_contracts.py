import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
from dexmani_policy.common.checkpoint import (
    TrainingCheckpoint, load_training_checkpoint, restore_policy_agent, save_training_checkpoint,
)
from dexmani_policy.common.config import load_policy_config, resolve_input_recipe
from dexmani_policy.common.runtime import get_rng_state
from dexmani_policy.datasets.common.episode_split import (
    build_index_manifest, get_val_mask, validate_split_manifest,
)
from dexmani_policy.datasets.common.distributed_sampler import (
    ResumableDistributedSampler, multitask_indices,
)
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset
from dexmani_policy.training.common.scheduler import get_scheduler
from dexmani_policy.training.common.hand_representation import describe_hand_data
from dexmani_policy.training.builder import build_model_and_ema, build_optimizer_and_scheduler
from dexmani_policy.training.data import build_dataset_and_normalizer, build_normalizer, build_train_loader
from dexmani_policy.training.pretrain.hand_prior.lamp import build_prior_dataset, kl_weight
from dexmani_policy.training.pretrain.hand_prior import lamp as lamp_pretrain
from dexmani_policy.training.pretrain.hand_prior.vqvae import build_policy_dataset
from dexmani_policy.training.trainer import Trainer
from dexmani_policy.training.resume import build_resume_contract, restore_training_state


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.zarr"
        root = zarr.open_group(str(self.path), mode="w")
        root.attrs["dt"] = 0.0625
        self.ends = np.array([10, 20, 30])
        root.create_dataset("meta/episode_ends", data=self.ends)
        self.action = np.arange(30 * 19, dtype=np.float32).reshape(30, 19) / 10
        self.state = -5 + self.action / 10
        self.action[10:20] += 10000
        self.action[20:] -= 100000
        self.state[10:20] += 30000
        self.state[20:] -= 200000
        for name, values in (("action", self.action), ("joint_state", self.state),
                             ("action_ee", np.arange(30 * 21, dtype=np.float32).reshape(30, 21)),
                             ("point_cloud", np.random.default_rng(4).normal(size=(30, 8, 6)).astype(np.float32))):
            root.create_dataset("data/" + name, data=values, chunks=(4, *values.shape[1:]))
        self.manifest = build_index_manifest(self.path, self.ends, dict(root.attrs), val_ratio=0)
        self.manifest.update(train_indices=[0], val_indices=[1], exclusions=[2])
        self.split_path = Path(self.directory.name) / "split.json"
        self.split_path.write_text(json.dumps(self.manifest))

    def dataset(self, cls=DemonstrationDataset, **kwargs):
        return cls(str(self.path), sensor_modalities=["joint_state", "point_cloud"],
                   split_manifest=str(self.split_path), val_ratio=0, pc_dim=6, **kwargs)

    def test_train_only_streaming_stats_and_validation(self):
        dataset = self.dataset()
        with patch.object(dataset.replay_buffer, "read", wraps=dataset.replay_buffer.read) as read:
            normalizer = build_normalizer(dataset, {"joint_state": "gaussian", "action": "limits"}, "action")
        for call in read.call_args_list:
            self.assertLessEqual(call.args[1].stop, 10)
            self.assertFalse(call.kwargs["cache"])
        torch.testing.assert_close(normalizer["joint_state"].input_stats["mean"], torch.from_numpy(self.state[:10].mean(0)))
        torch.testing.assert_close(normalizer["joint_state"].input_stats["std"], torch.from_numpy(self.state[:10].std(0, ddof=1)))
        torch.testing.assert_close(normalizer["action"].input_stats["max"], torch.from_numpy(self.action[:10].max(0)))
        val = dataset.get_validation_dataset()
        self.assertIsNone(val.augmentation_cfg)
        with self.assertRaisesRegex(ValueError, "reuse"):
            list(val.iter_normalization_data("action"))
        self.assertEqual([len(b["action"]) for b in DataLoader(val, batch_size=4)], [4, 4, 2])

    def test_multitask_and_auxiliary_statistics(self):
        first, second = self.dataset(), self.dataset()
        multi = MultiTaskDataset([first, second], ["one", "two"])
        normalizer = build_normalizer(multi, {"joint_state": "gaussian"}, "action")
        expected = torch.from_numpy(np.concatenate([self.state[:10]] * 2))
        torch.testing.assert_close(normalizer["joint_state"].input_stats["std"], expected.std(0))
        aux = self.dataset(use_aux_ee=True)
        aux_normalizer = build_normalizer(aux, {"action": "limits"}, "action")
        self.assertEqual(aux_normalizer["action"].params_dict["scale"].shape, (28,))
        ee = self.dataset(action_key="action_ee")
        ee_normalizer = build_normalizer(ee, {"action": "auto"}, "action_ee")
        torch.testing.assert_close(ee_normalizer["action"].params_dict["scale"][3:9], torch.ones(6))
        torch.testing.assert_close(ee_normalizer["action"].params_dict["offset"][3:9], torch.zeros(6))

    def test_manifest_pretraining_policy_identity(self):
        for name, factory in (("dqrise", build_policy_dataset), ("lamp", build_prior_dataset)):
            cfg = load_policy_config(name, [f"zarr_path={self.path}",
                f"dataset.split_manifest={self.split_path}", "dataset.val_ratio=0",
                "dataset.max_train_episodes=null", "dataset.augmentation_cfg=null"])
            pretrain = factory(cfg)
            cls = LAMPDataset if name == "lamp" else DemonstrationDataset
            policy = self.dataset(cls, horizon=cfg.horizon, obs_horizon=cfg.n_obs_steps,
                                  pad_before=cfg.dataset.pad_before, pad_after=cfg.dataset.pad_after)
            self.assertEqual(pretrain.data_recipe["split_manifest"], policy.data_recipe["split_manifest"])
            np.testing.assert_array_equal(pretrain.train_mask, policy.train_mask)
            np.testing.assert_array_equal(pretrain.val_mask, policy.val_mask)
        for key, value in (("zarr_path", "/wrong"), ("episode_ends", [9, 20, 30]),
                           ("train_indices", [True]), ("val_indices", [0])):
            invalid = {**self.manifest, key: value}
            with self.assertRaises(ValueError):
                validate_split_manifest(invalid, {}, 3, episode_ends=self.ends, zarr_path=self.path)
        with self.assertRaises(ValueError):
            build_index_manifest(self.path, [10], {}, val_ratio=0.1)
        for ends in ([], [0], [10, 10], [10.5], np.array([10, 9], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                build_index_manifest(self.path, ends, {}, val_ratio=0)

    def test_history_and_dq_source_alignment(self):
        dataset = self.dataset(LAMPDataset, history_steps=8)
        for row in (0, 1, 7, 8, 9, 10, 11, 20):
            start = (row // 10) * 10
            expected = np.stack([self.action[i, 7:] if i >= start else self.state[start, 7:]
                                 for i in range(row - 8, row)])
            np.testing.assert_array_equal(dataset.history_for_frame(row), expected)
        sample = dataset[0]
        torch.testing.assert_close(sample["action_context"]["hand_history"], torch.from_numpy(np.repeat(self.state[0:1, 7:], 8, axis=0)))
        torch.testing.assert_close(sample["action"][0], torch.from_numpy(self.action[0]))
        dq = self.dataset(horizon=20, obs_horizon=1, pad_before=0, pad_after=15)
        for i in range(len(dq)):
            rows = dq.sampler.sample_source_rows(i)
            batch = dq[i]
            self.assertEqual(rows[0], i)
            torch.testing.assert_close(batch["obs"]["joint_state"][0], torch.from_numpy(self.state[i]))
            torch.testing.assert_close(batch["action"][0], torch.from_numpy(self.action[i]))
            np.testing.assert_array_equal(batch["action"].numpy(), self.action[rows])

    def test_explicit_trial_format_and_holdout(self):
        ids = ["e0", "e1", "e2"]
        manifest = dict(
            format="dexmani.episode_trials.v1", episode_ids=ids,
            trial_ids={"e0": "t0", "e1": "t1", "e2": "t2"},
            train_ids=["e2", "e0"], val_ids=["e1"], exclusions=[],
        )
        kwargs = dict(episode_ends=self.ends, zarr_path=self.path)
        train, val, normalized = validate_split_manifest(manifest, {"episode_ids": ids}, 3, **kwargs)
        np.testing.assert_array_equal(train, [True, False, True])
        np.testing.assert_array_equal(val, [False, True, False])
        self.assertEqual(normalized["train_ids"], ["e0", "e2"])
        root = zarr.open_group(str(self.path), mode="a")
        root.attrs["episode_ids"] = ids
        self.split_path.write_text(json.dumps(manifest))
        dataset = self.dataset()
        np.testing.assert_array_equal(dataset.train_mask, train)
        np.testing.assert_array_equal(dataset.val_mask, val)
        self.assertEqual(dataset.data_recipe["split_manifest"], normalized)
        invalid = copy.deepcopy(manifest)
        invalid["trial_ids"]["e1"] = "t0"
        with self.assertRaisesRegex(ValueError, "trial cannot cross"):
            validate_split_manifest(invalid, {"episode_ids": ids}, 3, **kwargs)
        for source in (manifest, self.manifest):
            invalid = {key: value for key, value in source.items() if key != "format"}
            with self.assertRaisesRegex(ValueError, "format"):
                validate_split_manifest(invalid, {"episode_ids": ids}, 3, **kwargs)
        with self.assertRaisesRegex(ValueError, "Holdout"):
            get_val_mask(1, 0.1)
        np.testing.assert_array_equal(get_val_mask(1, 0), [False])
        self.split_path.write_text(json.dumps({"train_ids": ["e0"]}))
        with patch.object(DemonstrationDataset, "finite_rows") as scan:
            with self.assertRaisesRegex(ValueError, "format"):
                self.dataset()
            scan.assert_not_called()

    def test_multitask_sampling_cursor_and_recipe(self):
        dataset = MultiTaskDataset([self.dataset(), self.dataset()], ["first", "second"])
        for rank in range(2):
            sampler = ResumableDistributedSampler(
                dataset, batch_size=2, num_replicas=2, rank=rank, seed=42,
            )
            sampler.set_epoch(3)
            complete = list(sampler)
            sampler.set_epoch(3, next_micro_step=1)
            self.assertEqual(list(sampler), complete[2:])
            sampler.set_epoch(3)
            self.assertEqual(list(sampler), complete)
            sampler.set_epoch(4)
            self.assertNotEqual(list(sampler), complete)
        validation = dataset.get_validation_dataset()
        args = (validation.task_lengths, validation.sample_probs, validation.seed)
        first = multitask_indices(*args, epoch=0, deterministic=True, sampling_strategy="proportional")
        later = multitask_indices(*args, epoch=5, deterministic=True, sampling_strategy="proportional")
        self.assertEqual(first, later)
        self.assertEqual(sorted(first), list(range(len(validation))))
        weighted = multitask_indices([10, 10], [0, 1], seed=42, epoch=0, sampling_strategy="weighted")
        self.assertEqual(len(weighted), 20)
        self.assertTrue(all(10 <= index < 20 for index in weighted))

        loader = DataLoader(dataset, sampler=sampler, batch_size=2)
        contract = build_resume_contract(loader)
        self.assertEqual(contract["multitask_sampling"], "numpy_seed_epoch")
        previous = {key: value for key, value in contract.items() if key != "multitask_sampling"}
        checkpoint = SimpleNamespace(effective_config={}, resume_contract=previous)
        components = SimpleNamespace(
            train_loader=loader, optimizer=None, scheduler=None, ema_updater=None,
        )
        with self.assertRaisesRegex(ValueError, "sampling recipe changed"):
            restore_training_state(checkpoint, components, effective_config={})

    def test_dq_checkpoint_resume_next_update(self):
        directory = Path(self.directory.name)
        cfg = load_policy_config("dqrise", [f"zarr_path={self.path}",
            f"dataset.split_manifest={self.split_path}", "dataset.val_ratio=0",
            "dataset.augmentation_cfg=null", f"codebook_path={directory / 'codebook.npz'}"])
        with open_dict(cfg):
            cfg.workspace.output_dir = str(directory / "run")
            cfg.workspace.wandb_cfg = None
            cfg.agent.num_points = 8
            cfg.agent.state_out_dim = 8
            cfg.agent.pc_encoder_config = dict(
                channels=24, num_patches=2, group_size=4, depth=1, num_heads=4,
            )
            cfg.agent.fps_random_config = {"use_random": False}
            cfg.agent.down_dims = [16, 32]
            cfg.agent.diffusion_step_embed_dim = 32
            cfg.dataloader.batch_size = 2
            cfg.dataloader.num_workers = 0
            cfg.dataloader.persistent_workers = False
            cfg.dataloader.prefetch_factor = None
            cfg.training.device = "cpu"
            cfg.training.use_compile = False
            cfg.training.use_bfloat16 = False
            cfg.training.lr_warmup_steps = 2
            cfg.training.loop.total_train_steps = 6
        OmegaConf.resolve(cfg)
        dataset, normalizer = build_dataset_and_normalizer(cfg, verbose=False)
        codebook = HandStateCodebook(12)
        params = normalizer["action"].params_dict
        codebook.set_buffers(dict(
            sorted_hand_poses=torch.linspace(-1, 1, 16)[:, None].expand(16, 12).clone(),
            pca_permutation=torch.arange(16), layer_weights=torch.tensor([0.2, 0.8]),
            hand_normalizer_scale=params["scale"][7:], hand_normalizer_offset=params["offset"][7:],
        ))
        codebook.artifact_metadata = dict(
            hand_data=describe_hand_data(dataset), split_manifest=self.manifest,
        )
        codebook.save(cfg.codebook_path)
        model, ema, updater = build_model_and_ema(cfg, "cpu", normalizer, dataset=dataset)
        self.assertIsNone(ema)
        self.assertIsNone(updater)
        optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=False)
        loader = build_train_loader(cfg, dataset)
        batch = next(iter(loader))
        model.train().compute_loss(batch)[0].backward()
        optimizer.step()
        scheduler.step()
        saved = OmegaConf.to_container(cfg, resolve=True)
        checkpoint = TrainingCheckpoint(
            epoch=0, global_step=1, next_micro_step=1, model_state=model.state_dict(),
            ema_model_state=None, optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(), resume_contract=build_resume_contract(loader),
            ema_updater_step=None, rng_states=[get_rng_state()], effective_config=saved,
        )
        checkpoint_path = save_training_checkpoint(directory / "policy.pt", checkpoint)
        checkpoint = load_training_checkpoint(checkpoint_path)
        Path(cfg.codebook_path).unlink()
        invocation = copy.deepcopy(cfg)
        with open_dict(invocation):
            invocation.resume_from = str(checkpoint_path)
        restored_cfg = resolve_input_recipe(invocation, overrides=[], resume_checkpoint=checkpoint)
        restored, _, _ = build_model_and_ema(
            restored_cfg, "cpu", None, dataset=dataset, checkpoint=checkpoint,
        )
        restored_optimizer, restored_scheduler = build_optimizer_and_scheduler(
            restored_cfg, restored, verbose=False,
        )
        components = SimpleNamespace(
            train_loader=loader, optimizer=restored_optimizer, scheduler=restored_scheduler,
            ema_updater=None, device="cpu",
        )
        self.assertEqual(restore_training_state(checkpoint, components, effective_config=restored_cfg), (1, 0, 1))
        rng = torch.get_rng_state()
        losses = []
        for policy, opt, sched in ((model, optimizer, scheduler),
                                   (restored, restored_optimizer, restored_scheduler)):
            torch.set_rng_state(rng)
            opt.zero_grad(set_to_none=True)
            loss, _ = policy.train().compute_loss(batch, log_metrics=False)
            losses.append(loss.detach())
            loss.backward()
            opt.step()
            sched.step()
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
        self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())
        inference = restore_policy_agent(saved, checkpoint_path, use_ema=False, device="cpu")
        torch.testing.assert_close(inference.state_dict(), checkpoint.model_state, rtol=0, atol=0)
        self.assertEqual(inference.predict_action(batch["obs"])["control_action"].shape, (2, 16, 19))


class LAMPPriorBatchTests(unittest.TestCase):
    def test_source_row_batch_sizes_and_update_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                steps=5, batch_size=256, eval_every=2, print_every=1,
                beta=0.001, beta_warmup=2, warmup_steps=2, noise_std=0.01,
                weight_decay=1e-4, lr=2e-3, min_lr=1e-5, clip_grad=1.0,
                policy_config=None, config="lamp", overrides=[],
                output=str(Path(directory) / "prior.pt"), overwrite=False,
                device="cpu", seed=42,
            )
            cfg = SimpleNamespace(agent=SimpleNamespace(
                history_steps=3, latent_dim=2, prior_hidden_dim=8,
            ))
            for count in (257, 3, 0):
                with self.subTest(count=count):
                    dataset = SimpleNamespace(
                        sampler=SimpleNamespace(action_source_rows=np.arange(count)),
                        get_validation_dataset=lambda: None,
                        history_for_rows=lambda rows: np.zeros((len(rows), 3, 12), np.float32),
                        hand_actions=np.zeros((count, 12), np.float32),
                        data_recipe={"split_manifest": {}},
                    )
                    prior = lamp_pretrain.HandMotionPrior(12, 3, 2, 8)
                    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr)
                    with (
                        patch.object(lamp_pretrain, "load_policy_config", return_value=cfg),
                        patch.object(lamp_pretrain, "build_prior_dataset", return_value=dataset),
                        patch.object(lamp_pretrain, "describe_hand_data", return_value={}),
                        patch.object(lamp_pretrain, "HandMotionPrior", return_value=prior) as model,
                        patch.object(torch.optim, "AdamW", return_value=optimizer),
                        patch.object(lamp_pretrain, "prior_batch", wraps=lamp_pretrain.prior_batch) as batch,
                        patch.object(lamp_pretrain, "kl_weight", wraps=kl_weight) as beta,
                        patch.object(lamp_pretrain, "save_prior") as save,
                        patch("builtins.print") as log,
                    ):
                        if count == 0:
                            with self.assertRaisesRegex(ValueError, "nonempty training source rows"):
                                lamp_pretrain.train(args)
                            model.assert_not_called()
                            batch.assert_not_called()
                            continue
                        lamp_pretrain.train(args)
                    # The remaining calls belong to final evaluation, which keeps the tail.
                    sizes = [len(call.args[1]) for call in batch.call_args_list]
                    self.assertEqual(sizes[:args.steps], [min(count, args.batch_size)] * args.steps)
                    self.assertEqual(sizes[args.steps:], [256, 1] if count == 257 else [3])
                    self.assertTrue(all(int(state["step"]) == args.steps for state in optimizer.state.values()))
                    self.assertEqual([call.args[0] for call in beta.call_args_list], list(range(args.steps)))
                    self.assertEqual(save.call_args.args[5], args.steps)
                    if count < args.batch_size:
                        messages = "\n".join(str(call.args[0]) for call in log.call_args_list)
                        self.assertIn("small-data smoke exception: actual batch size=3", messages)


class ScheduleAndClipTests(unittest.TestCase):
    def test_schedule_boundaries_and_restore(self):
        for name, warmup, total, ratio in (("cosine_min_lr", 500, 20000, 0.005),
                                          ("cosine_min_lr", 1000, 20000, 0.01),
                                          ("cosine", 150, 3000, 0), ("cosine", 2000, 100000, 0)):
            p = torch.nn.Parameter(torch.tensor(1.0))
            opt = torch.optim.AdamW([p], lr=0.002)
            scheduler = get_scheduler(opt, name, warmup, total, lr_min_ratio=ratio)
            def expected(step):
                if step < warmup:
                    return step / (warmup - 1 if name == "cosine_min_lr" else warmup)
                return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
            for step in (0, 1, warmup - 1, warmup, warmup + 1, total - 1):
                self.assertAlmostEqual(scheduler.lr_lambdas[0](step), expected(step), places=14)
            for _ in range(3):
                opt.step()
                scheduler.step()
            state, schedule_state = copy.deepcopy(opt.state_dict()), copy.deepcopy(scheduler.state_dict())
            restored_opt = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=0.002)
            restored_schedule = get_scheduler(restored_opt, name, warmup, total, lr_min_ratio=ratio)
            restored_opt.load_state_dict(state)
            restored_schedule.load_state_dict(schedule_state)
            for _ in range(4):
                self.assertEqual(opt.param_groups[0]["lr"], restored_opt.param_groups[0]["lr"])
                opt.step()
                scheduler.step()
                restored_opt.step()
                restored_schedule.step()
        self.assertEqual(kl_weight(0, 1e-3, 2000), 0)
        self.assertEqual(kl_weight(1999, 1e-3, 2000), 1e-3)
        self.assertEqual(kl_weight(2000, 1e-3, 2000), 1e-3)
        self.assertEqual(kl_weight(0, 1e-3, 0), 1e-3)
        self.assertEqual(kl_weight(0, 1e-3, 1), 0)
        self.assertEqual(kl_weight(1, 1e-3, 1), 1e-3)

    def test_optimizer_clipping_groups_and_nonfinite(self):
        model = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1)) for _ in range(3)])
        opt = torch.optim.SGD([
            dict(params=[model[0]], clip_group="head"),
            dict(params=[model[1]], clip_group="head"),
            dict(params=[model[2]], clip_group="backbone"),
        ], lr=1)
        scheduler = get_scheduler(opt, "constant", 0, 10)
        trainer = Trainer.__new__(Trainer)
        trainer.model, trainer.optimizer, trainer.scheduler = model, opt, scheduler
        trainer.max_grad_norm, trainer.ema_updater = 1, None
        trainer.current_epoch = trainer.global_step = 0
        for p, value in zip(model, (3., 4., 10.)):
            p.grad = torch.tensor([value])
        trainer.optimizer_step()
        torch.testing.assert_close(torch.cat(list(model)), torch.tensor([-0.6, -0.8, -1.0]))
        trainer.max_grad_norm = 0
        model[0].grad = torch.tensor([float("nan")])
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            trainer.optimizer_step()
        torch.testing.assert_close(model.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
