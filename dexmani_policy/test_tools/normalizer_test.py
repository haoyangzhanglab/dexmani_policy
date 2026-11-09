"""Targeted normalization and strict raw/EMA restore regression checks."""

import copy
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import load_checkpoint, restore_policy
from dexmani_policy.common.normalizer import LinearNormalizer, normalization_feature_shapes
from dexmani_policy.test_tools.smoke_test import make_source
from dexmani_policy.train import build_trainer
from dexmani_policy.training.common.config import load_policy_config, validate_normalization
from dexmani_policy.training.common.data import build_dataset, fit_normalizer
from dexmani_policy.training.trainer import PolicyTrainer


class NormalizerTest(unittest.TestCase):
    def test_streaming_limits_and_constant_channels(self):
        values = torch.tensor([[1., 3., 7.], [5., 9., 7.], [3., 6., 7.]])
        normalizer = LinearNormalizer()
        normalizer.fit_field("action", (values[:1], values[1:1], values[1:]))
        field = normalizer["action"]
        result = field.normalize(values)
        torch.testing.assert_close(result.amin(0), torch.tensor([-1., -1., 0.]))
        torch.testing.assert_close(result.amax(0), torch.tensor([1., 1., 0.]))
        torch.testing.assert_close(field.unnormalize(result), values)
        probe = torch.randn(4, 3)
        torch.testing.assert_close(field.unnormalize(field.normalize(probe)), probe)
        self.assertTrue(torch.isfinite(field.scale).all())
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            normalizer.fit_field("action", [])
        # Non-tactile multi-axis fields must retain independent feature limits.
        tips = torch.randn(8, 5, 3)
        normalizer.fit_field("fingertip_xyz", tips.split(3))
        self.assertEqual(normalizer["fingertip_xyz"].scale.shape, (5, 3))
        torch.testing.assert_close(normalizer["fingertip_xyz"].normalize(tips).amin(0), -torch.ones(5, 3))

    def test_identity_structure_and_training_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zarr"
            source = make_source(path)
            cfg = load_policy_config("dp3", [f"zarr_path={path}", "horizon=4", "n_action_steps=2", "dataset.val_episodes=1"])
            dataset, description = build_dataset(cfg)
            normalizer = LinearNormalizer(normalization_feature_shapes(description["field_dims"], cfg.normalization))
            fit_normalizer(dataset, cfg.normalization, normalizer)
            episode = int(np.flatnonzero(dataset.train_mask)[0])
            train = torch.from_numpy(source["data/action"][episode * 6:(episode + 1) * 6])
            torch.testing.assert_close(normalizer["action"].normalize(train).amin(0), -torch.ones(19))
            torch.testing.assert_close(normalizer["action"].normalize(train).amax(0), torch.ones(19))
            spec = dict(action="identity", point_cloud="identity", finger_tactile_force="identity",
                        taxel_tactile_force="identity")
            dims = dict(action=19, point_cloud=6, finger_tactile_force=15, taxel_tactile_force=1800)
            identity = LinearNormalizer(normalization_feature_shapes(dims, spec))
            with patch.object(dataset, "iter_normalization_data", side_effect=AssertionError("identity must not fit")):
                fit_normalizer(dataset, spec, identity)
            self.assertEqual(set(identity.fields), {"action", "finger_tactile_force", "taxel_tactile_force"})
            for field in identity.fields.values():
                value = torch.randn(2, *field.scale.shape)
                self.assertTrue(torch.equal(field.normalize(value), value))
                self.assertTrue(torch.equal(field.scale, torch.ones_like(field.scale)))
                self.assertTrue(torch.equal(field.offset, torch.zeros_like(field.offset)))

    def test_new_training_rejects_old_modes(self):
        cfg = load_policy_config("adaptac")
        validate_normalization(cfg)
        for mode in ("gaussian", "rms", "max_abs", "unknown"):
            cfg.normalization.action = mode
            with self.assertRaisesRegex(ValueError, "normalization mode"):
                validate_normalization(cfg)
            with self.assertRaisesRegex(ValueError, "normalization mode"):
                fit_normalizer(None, {"action": mode}, LinearNormalizer({"action": (19,)}))
        cfg.normalization.action = "limits"
        cfg.normalization.finger_tactile_force = "limits"
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_normalization(cfg)

    def test_inherited_affine_and_hand_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zarr"
            make_source(path)
            cfg = load_policy_config("dp3", [f"zarr_path={path}", "horizon=4", "n_action_steps=2"])
            dataset, description = build_dataset(cfg)
            inherited = LinearNormalizer({"point_cloud": (6,)})
            inherited["point_cloud"].scale.fill_(2.5)
            inherited["point_cloud"].offset.fill_(-0.3)
            hand = {"scale": torch.linspace(1, 3, 12), "offset": torch.linspace(-2, 1, 12)}
            fitted = LinearNormalizer({
                **normalization_feature_shapes(description["field_dims"], cfg.normalization), "hand_latent": (32,),
            })
            fitted["hand_latent"].scale.fill_(3)
            fit_normalizer(dataset, cfg.normalization, fitted, initial_state=inherited.state_dict(),
                           inherited_fields=("point_cloud",), hand_affine=hand)
            torch.testing.assert_close(fitted["hand_latent"].scale, torch.full((32,), 3.))
            for part in ("scale", "offset"):
                torch.testing.assert_close(getattr(fitted["action"], part)[7:], hand[part], rtol=0, atol=0)
                torch.testing.assert_close(getattr(fitted["point_cloud"], part),
                                           getattr(inherited["point_cloud"], part), rtol=0, atol=0)
            # A missing pretrained affine cannot silently become a newly fitted field.
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                fit_normalizer(dataset, cfg.normalization, fitted,
                               initial_state={}, inherited_fields=("point_cloud",))

    def test_patch_zero_reference_with_saved_nonidentity_affine(self):
        from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandPatchEncoder
        from dexmani_policy.agents.policies.adaptac import AdapTac

        model = AdapTac(
            horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
            tactile_encoder=XHandPatchEncoder(), num_points=8, feature_dim=32,
            down_dims=(16, 32), diffusion_step_embed_dim=16,
            pc_encoder_config=dict(num_patches=4, group_size=4, depth=1),
            force_predictor_config=dict(depth=1, embed_dim=32, num_heads=4),
        ).eval()
        dims = dict(action=19, joint_state=19, point_cloud=6, taxel_tactile_force=1800)
        spec = dict(action="limits", joint_state="limits", point_cloud="identity", taxel_tactile_force="identity")
        model.configure_normalizer(dims, normalization_feature_shapes(dims, spec))
        field = model.normalizer["taxel_tactile_force"]
        field.scale.fill_(2.5)
        field.offset.fill_(-0.75)
        obs = dict(joint_state=torch.randn(1, 2, 19), point_cloud=torch.randn(1, 2, 8, 6),
                   taxel_tactile_force=torch.zeros(1, 2, 5, 120, 3))
        with torch.no_grad():
            pre = model._preprocess_obs(obs)["taxel_tactile_force"]
            self.assertEqual(torch.count_nonzero(pre), 0)
            self.assertEqual(torch.count_nonzero(model.obs_encoder.tactile_encoder._patch_descriptors(pre)), 0)
            _, current, _ = model._encode_force_inputs(obs)
            torch.testing.assert_close(current, torch.full_like(current, -0.75), rtol=0, atol=0)
            obs["taxel_tactile_force"].normal_()
            expected = obs["taxel_tactile_force"] * field.scale
            torch.testing.assert_close(model._preprocess_obs(obs)["taxel_tactile_force"], expected.flatten(0, 1))
            _, current, _ = model._encode_force_inputs(obs)
            torch.testing.assert_close(current, (expected + field.offset).reshape(1, 2, -1))


class RestoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        path = root / "source.zarr"
        source = make_source(path)
        rng = np.random.default_rng(42)
        for key, shape in (("eef_pose", (12, 9)), ("fingertip_xyz", (12, 5, 3)),
                           ("finger_tactile_force", (12, 5, 3))):
            source.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype("float32"))
        cfg = load_policy_config("interact_dex", [
            f"zarr_path={path}", "training.device=cpu", "horizon=4", "n_obs_steps=2", "n_action_steps=2",
            "agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=4",
            "agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_patches=4",
            "agent.pc_encoder_config.group_size=4", "agent.num_points=8",
            "dataloader.batch_size=2", "dataloader.num_workers=0", "dataset.augmentation_cfg=null",
            "training.loop.total_train_steps=3", "training.lr_warmup_steps=0",
        ])
        cfg.workspace.output_dir = str(root / "run")
        cfg.workspace.wandb_cfg = None
        with closing(build_trainer(cfg)) as trainer:
            trainer.save_checkpoint("normalizer regression fixture")
        cls.checkpoint = load_checkpoint(root / "run/checkpoints/latest.pt")
        cls.cfg = OmegaConf.create(cls.checkpoint.effective_config)
        cls.dataset, cls.description = build_dataset(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saved_values_preserved_without_refitting(self):
        saved = copy.deepcopy(self.checkpoint)
        # Legacy modes and non-unit tactile affine remain valid on restore.
        saved.effective_config["normalization"]["joint_state"] = "gaussian"
        for index, state in enumerate((saved.model_state, saved.ema_model_state)):
            state["normalizer.fields.finger_tactile_force.scale"].fill_(2 + index)
            state["normalizer.fields.finger_tactile_force.offset"].fill_(-0.75 - index)
            deployed = restore_policy(saved.effective_config, state, "cpu")
            legacy_config = copy.deepcopy(saved.effective_config)
            legacy_config["normalization"]["finger_tactile_force"] = "limits"
            legacy = restore_policy(legacy_config, state, "cpu")
            torch.testing.assert_close(legacy.normalizer.state_dict(), deployed.normalizer.state_dict(), rtol=0, atol=0)
            for name, value in deployed.normalizer.state_dict().items():
                torch.testing.assert_close(value, state[f"normalizer.{name}"], rtol=0, atol=0)
        with patch.object(self.dataset, "iter_normalization_data", side_effect=AssertionError("resume must not fit")):
            with closing(PolicyTrainer(OmegaConf.create(saved.effective_config), self.dataset,
                                       self.description, checkpoint=saved)) as resumed:
                for model, state in ((resumed.model, saved.model_state), (resumed.ema_model, saved.ema_model_state)):
                    for name, value in model.normalizer.state_dict().items():
                        torch.testing.assert_close(value, state[f"normalizer.{name}"], rtol=0, atol=0)

    def test_interact_zero_force_residual(self):
        model = restore_policy(self.checkpoint.effective_config, self.checkpoint.model_state, "cpu")
        field = model.normalizer["finger_tactile_force"]
        # Clone: assign=True intentionally shares CPU checkpoint storage.
        field.scale = torch.full_like(field.scale, 2.5)
        field.offset = torch.full_like(field.offset, -0.75)
        obs = {key: value.unsqueeze(0) for key, value in self.dataset[0]["obs"].items()}
        obs["finger_tactile_force"].zero_()
        geometry = []
        handle = model.obs_encoder.cross_attention.register_forward_hook(lambda module, args, output: geometry.append(output))
        with torch.no_grad():
            features = model._encode_features(obs)
        handle.remove()
        torch.testing.assert_close(features.interaction.flatten(0, 1), geometry[0], rtol=0, atol=0)

    def test_strict_structure_and_values_in_all_restore_paths(self):
        base = self.checkpoint.model_state
        mutations = []
        for field in ("action", "finger_tactile_force", "joint_state"):
            for parts in (("scale",), ("offset",), ("scale", "offset")):
                keys = [f"normalizer.fields.{field}.{part}" for part in parts]
                mutations.append((f"missing {keys}", {k: v for k, v in base.items() if k not in keys}, RuntimeError))
        key = "normalizer.fields.action.scale"
        mutations += [("shape", {**base, key: torch.ones(1)}, RuntimeError),
                      ("extra", {**base, "normalizer.fields.extra.scale": torch.ones(1)}, RuntimeError)]
        for part, value in (("scale", float("nan")), ("scale", float("inf")), ("scale", 0.),
                            ("offset", float("nan")), ("offset", float("inf"))):
            key = f"normalizer.fields.action.{part}"
            mutations.append((f"{part}={value}", {**base, key: torch.full_like(base[key], value)}, ValueError))
        for name, state, error in mutations:
            with self.subTest(path="deployment", defect=name), self.assertRaises(error):
                restore_policy(self.checkpoint.effective_config, state, "cpu")
            for slot in ("model_state", "ema_model_state"):
                saved = copy.copy(self.checkpoint)
                setattr(saved, slot, state)
                with self.subTest(path=slot, defect=name), self.assertRaises(error):
                    PolicyTrainer(copy.deepcopy(self.cfg), self.dataset, self.description, checkpoint=saved)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
