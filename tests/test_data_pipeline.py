from contextlib import closing, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
import zarr

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs, validate_rgb_preprocessing
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.aux.r3d_dataset import R3DAuxDataset
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.deployment.runtime import PolicyRuntime
from dexmani_policy.training.common.data import build_dataset_and_normalizer
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.common.resume import resolve_input_recipe
from dexmani_policy.training.trainer import PolicyTrainer


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.zarr_path = self.root / "data.zarr"
        rng = np.random.default_rng(42)
        self.action = rng.normal(size=(24, 19)).astype(np.float32)
        self.action[8:16] += 100
        self.auxiliary = rng.normal(size=(24, 21)).astype(np.float32)
        self.auxiliary[:, 9:] = np.nan
        arrays = {
            "action": self.action,
            "action_ee": self.auxiliary,
            "joint_state": rng.normal(size=(24, 19)).astype(np.float32),
            "point_cloud": rng.normal(size=(24, 8, 6)).astype(np.float32),
            "rgb": rng.integers(0, 256, size=(24, 12, 16, 3), dtype=np.uint8),
        }
        group = zarr.open_group(str(self.zarr_path), mode="w")
        for key, values in arrays.items():
            group.create_dataset(f"data/{key}", data=values, chunks=(4, *values.shape[1:]))
        group.create_dataset("meta/episode_ends", data=np.array([8, 16, 24]))
        self.manifest = {
            "source_id": None, "episode_ids": None, "trial_ids": None,
            "episode_ends": [8, 16, 24], "train_indices": [0, 2], "val_indices": [1],
        }

    def make_config(self, name="dp3"):
        cfg = load_policy_config(name)
        with open_dict(cfg):
            cfg.zarr_path = str(self.zarr_path)
            cfg.horizon = 4
            cfg.n_action_steps = 2
            cfg.dataset.update(split_manifest=self.manifest, pad_after=0, augmentation_cfg=None)
            cfg.dataloader.update(batch_size=2, num_workers=0, pin_memory=False)
            cfg.training.update(device="cpu", use_compile=False, use_bfloat16=False)
            cfg.training.loop.update(total_train_steps=4, log_interval_steps=2)
            cfg.training.lr_warmup_steps = 0
            cfg.workspace.output_dir = str(self.root / "training")
            cfg.workspace.wandb_cfg = None
            if name == "dp3":
                cfg.agent.update(
                    pc_out_dim=8, state_out_dim=8, num_points=8, down_dims=[16, 32],
                    diffusion_step_embed_dim=16, num_training_steps=10, num_inference_steps=2,
                )
        return cfg

    def test_crop_dimensions_are_pairs(self):
        validate_rgb_preprocessing((10, 12), (8, 10))
        for resize, crop in (((10, 12), (8, 10, 3)), ((10, 12, 3), (8, 10)), ((10, 12), ())):
            with self.subTest(resize=resize, crop=crop), self.assertRaises(ValueError):
                validate_rgb_preprocessing(resize, crop)

    def test_rgb_validation_and_deployment_share_preprocessing(self):
        settings = dict(
            rgb_preprocess_size=(10, 12), rgb_random_crop_size=(8, 10),
            rgb_keep_uint8=True, rgb_color_aug=lambda value: value,
        )
        dataset = RobotDataset(
            str(self.zarr_path), sensor_modalities=["joint_state", "rgb"],
            horizon=4, obs_horizon=2, pad_before=1, split_manifest=self.manifest, **settings,
        )
        self.assertEqual(dataset[0]["obs"]["rgb"].shape, (2, 3, 8, 10))
        validation = dataset.get_validation_dataset()
        normalizer = dataset.get_normalizer({"action": "limits", "joint_state": "limits"})
        self.assertIs(validation.get_normalizer({}), normalizer)
        expected = torch.from_numpy(self.action[np.r_[0:8, 16:24]].min(axis=0))
        torch.testing.assert_close(normalizer["action"].input_stats["min"], expected)

        runtime = PolicyRuntime.__new__(PolicyRuntime)
        runtime.info = SimpleNamespace(observation_fields=dataset.sensor_modalities, n_obs_steps=2)
        runtime.device = "cpu"
        runtime.rgb_preprocessing = rgb_preprocessing_kwargs(settings)
        raw = validation.sampler.sample_sequence(0)
        observation = {key: raw[key] for key in dataset.sensor_modalities}
        prepared = runtime.prepare_observations(observation)
        for key, value in validation[0]["obs"].items():
            torch.testing.assert_close(prepared[key][0], value, rtol=0, atol=0)
        observation["rgb"] = observation["rgb"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            runtime.prepare_observations(observation)

    def test_auxiliary_target_and_normalizer_use_the_same_columns(self):
        cfg = self.make_config("r3d")
        for enabled in (False, True):
            cfg.use_aux_ee = enabled
            dataset, normalizer, description = build_dataset_and_normalizer(cfg, verbose=False)
            self.assertIs(type(dataset), R3DAuxDataset if enabled else RobotDataset)
            width = 28 if enabled else 19
            self.assertEqual(dataset[0]["action"].shape, (4, width))
            self.assertEqual(description["normalization_dims"]["action"], width)
            self.assertEqual(description["control_action_dim"], 19)
            expected = self.action[np.r_[0:8, 16:24]]
            if enabled:
                expected = np.concatenate((expected, self.auxiliary[np.r_[0:8, 16:24], :9]), axis=-1)
            torch.testing.assert_close(
                normalizer["action"].input_stats["min"], torch.from_numpy(expected.min(axis=0)),
            )

    def test_normalizer_ignores_empty_chunks_and_rejects_nonfinite_affine(self):
        normalizer = LinearNormalizer()
        normalizer.fit_field("action", [torch.tensor([[1., 2.], [3., 4.]]), torch.empty(0, 99)])
        torch.testing.assert_close(normalizer["action"].scale, torch.ones(2))
        for mode in ("limits", "gaussian"):
            for value in (float("nan"), float("inf")):
                with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                    normalizer.fit_field("action", [torch.tensor([[1., value], [3., 4.]])], mode=mode)

    def test_resume_keeps_the_embedded_split_and_saved_recipe(self):
        cfg = self.make_config()
        saved = OmegaConf.to_container(cfg, resolve=True)
        original = copy.deepcopy(saved)
        resumed = resolve_input_recipe(cfg, saved_config=saved, overrides=["dataloader.num_workers=1"])
        self.assertEqual(OmegaConf.to_container(resumed.dataset.split_manifest), self.manifest)
        self.assertEqual(resumed.dataloader.num_workers, 1)
        self.assertEqual(saved, original)
        with self.assertRaisesRegex(ValueError, "forbidden override"):
            resolve_input_recipe(cfg, saved_config=saved, overrides=["horizon=8"])
        saved["dataset"].pop("split_manifest")
        with self.assertRaisesRegex(ValueError, "split_manifest"):
            resolve_input_recipe(cfg, saved_config=saved, overrides=[])

    def test_rng_restore_failure_closes_the_training_log(self):
        cfg = self.make_config()
        with redirect_stdout(io.StringIO()):
            dataset, normalizer, description = build_dataset_and_normalizer(cfg, verbose=False)
            with closing(PolicyTrainer(cfg, dataset, normalizer, description)) as trainer:
                trainer.save_checkpoint("test")
                source = trainer.checkpoint_dir / "latest.pt"
            checkpoint = load_checkpoint(source)
            resumed = OmegaConf.create(checkpoint.effective_config)
            with open_dict(resumed):
                resumed.resume_from = str(source)
                resumed.workspace.output_dir = str(self.root / "resume")
            dataset, normalizer, description = build_dataset_and_normalizer(
                resumed, resume_checkpoint=checkpoint, verbose=False,
            )
            loggers = []

            def create_logger(**kwargs):
                logger = ExperimentLogger(**kwargs)
                loggers.append(logger)
                return logger

            with patch("dexmani_policy.training.trainer.ExperimentLogger", side_effect=create_logger), patch(
                "dexmani_policy.training.trainer.set_rng_state", side_effect=RuntimeError("RNG restore failed"),
            ), self.assertRaisesRegex(RuntimeError, "RNG restore failed"):
                PolicyTrainer(resumed, dataset, normalizer, description, checkpoint=checkpoint)
        self.assertTrue(loggers[0].metrics_file.closed)

    def test_cpu_training_and_checkpoint_restoration(self):
        from dexmani_policy.smoke_test import validate_training

        with redirect_stdout(io.StringIO()):
            validate_training(self.make_config(), max_updates=2)

    def test_cpu_resume_reuses_saved_normalizer_and_rng(self):
        cfg = self.make_config()
        with redirect_stdout(io.StringIO()):
            dataset, normalizer, description = build_dataset_and_normalizer(cfg, verbose=False)
            with closing(PolicyTrainer(cfg, dataset, normalizer, description)) as trainer:
                trainer.train(max_updates=1)
                source = trainer.checkpoint_dir / "latest.pt"
            checkpoint = load_checkpoint(source)
            resumed = OmegaConf.create(checkpoint.effective_config)
            with open_dict(resumed):
                resumed.resume_from = str(source)
                resumed.workspace.output_dir = str(self.root / "resume")
            with patch.object(RobotDataset, "get_normalizer", side_effect=AssertionError("unexpected refit")):
                dataset, normalizer, description = build_dataset_and_normalizer(
                    resumed, resume_checkpoint=checkpoint, verbose=False,
                )
            with closing(PolicyTrainer(resumed, dataset, normalizer, description, checkpoint=checkpoint)) as trainer:
                self.assertEqual(trainer.global_step, 1)
                torch.testing.assert_close(torch.get_rng_state(), checkpoint.rng_state["torch"])
                trainer.train(max_updates=1)
                self.assertEqual(trainer.global_step, 2)
                for key, value in normalizer.state_dict().items():
                    torch.testing.assert_close(value, checkpoint.model_state[f"normalizer.{key}"])


if __name__ == "__main__":
    unittest.main()
