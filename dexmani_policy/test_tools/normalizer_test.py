"""Targeted normalization and strict raw/EMA restore regression checks."""

import copy
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import load_checkpoint, restore_policy
from dexmani_policy.common.normalizer import FieldNormalizer, LinearNormalizer, normalization_feature_shapes
from dexmani_policy.test_tools.smoke_test import make_source, run_smoke
from dexmani_policy.train import build_trainer
from dexmani_policy.training.common.config import load_policy_config, validate_normalization
from dexmani_policy.training.common.data import build_dataset, fit_normalizer
from dexmani_policy.training.trainer import PolicyTrainer


def make_policy_fixture(root, recipe, tactile="force"):
    path = root / "source.zarr"
    source = make_source(path)
    rng = np.random.default_rng(42)
    for key, shape in (("eef_pose", (12, 9)), ("fingertip_xyz", (12, 5, 3)),
                       ("finger_tactile_force", (12, 5, 3)), ("taxel_tactile_force", (12, 5, 120, 3)),
                       ("hand_pointcloud", (12, 8, 3)), ("object_pointcloud", (12, 1024, 3))):
        source.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype("float32"))
    source.create_dataset("data/rgb", data=rng.integers(0, 256, (12, 32, 32, 3), dtype=np.uint8))
    source.create_dataset("data/camera_intrinsic", data=np.tile(np.array([20, 0, 16, 0, 20, 16, 0, 0, 1], np.float32), (12, 1)))
    source.create_dataset("data/camera_extrinsic", data=np.tile(np.eye(3, 4, dtype=np.float32).reshape(12), (12, 1)))
    source.attrs["hoi_pcd"] = dict(task_name="pick_apple_messy", object_names=["fixture"], template_sha256="0" * 64)
    overrides = [f"zarr_path={path}", "training.device=cpu", "horizon=4", "n_obs_steps=2", "n_action_steps=2",
                 "dataloader.batch_size=2", "dataloader.num_workers=0", "dataset.augmentation_cfg=null",
                 "training.loop.total_train_steps=2", "training.lr_warmup_steps=0", "training.use_compile=false",
                 "agent.num_inference_steps=2"]
    if recipe in {"dp3", "adaptac", "afro", "cordvip"}:
        overrides += ["agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16"]
    else:
        overrides += ["agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=4"]
    if recipe in {"adaptac", "interact_dex", "dexlatent", "dqrise"}:
        overrides += ["agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_patches=4",
                      "agent.pc_encoder_config.group_size=4", "agent.num_points=8"]
    if recipe == "dp3":
        overrides += ["agent.num_points=8"]
    if recipe == "adaptac":
        overrides += [f"tactile={tactile}", "agent.feature_dim=32", "agent.force_num_inference_steps=2",
                      "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=32"]
    if recipe in {"dexlatent", "dqrise"}:
        overrides += ["initialization.rgb=null", "agent.rgb_backbone_name=siglip",
                      "agent.rgb_backbone_config.tune_mode=full", "agent.rgb_backbone_config.image_size=[32,32]",
                      "+agent.rgb_backbone_config.architecture={hidden_size:32,intermediate_size:64,num_hidden_layers:1,num_attention_heads:4,patch_size:8,image_size:32}"]
    cfg = load_policy_config(recipe, overrides)
    if recipe == "dqrise":
        asset = root / "codebook.npz"
        np.savez(asset, sorted_hand_poses=rng.uniform(-1, 1, (8, 12)).astype("float32"),
                 hand_normalizer_scale=np.linspace(1, 3, 12, dtype="float32"),
                 hand_normalizer_offset=np.linspace(-2, 1, 12, dtype="float32"))
        cfg.initialization.codebook = str(asset)
    elif recipe == "dexlatent":
        from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior
        asset = root / "prior.pt"
        torch.save({"autoencoders": {f"xarm7_xhand_right.{key}": value
                                    for key, value in DexLatentHandPrior().state_dict().items()}}, asset)
        cfg.initialization.dexlatent = str(asset)
    elif recipe in {"afro", "cordvip"}:
        from dexmani_policy.training.pretrain.obs.afro import AFRO
        from dexmani_policy.training.pretrain.obs.cordvip import CordViP
        from dexmani_policy.training.pretrain.obs.common import export_encoder
        spec = {key: mode for key, mode in cfg.normalization.items() if key != "action" and
                (recipe == "cordvip" or key == "point_cloud")}
        pretrain = AFRO(spec) if recipe == "afro" else CordViP(spec, n_obs_steps=2)
        for field in pretrain.normalizer.fields.values():
            field.scale.fill_(2.5)
            field.offset.fill_(-0.3)
        asset = root / "encoder.pt"
        export_encoder(pretrain.state_dict(), spec, asset)
        cfg.initialization.encoder = str(asset)
    cfg.workspace.output_dir = str(root / "run")
    cfg.workspace.wandb_cfg = None
    return cfg, source


class NormalizerTest(unittest.TestCase):
    def test_invalid_inputs_and_config(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Nonfinite"):
                LinearNormalizer().fit_field("action", [torch.tensor([[value]])])
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            LinearNormalizer().fit_field("action", [torch.empty(0, 3)])
        for scale, offset in (([0.], [0.]), ([1.], [float("inf")]), ([1., 2.], [0.])):
            with self.assertRaises(ValueError):
                FieldNormalizer(scale, offset)
        dims = dict(action=19, joint_state=19, finger_tactile_force=15)
        good = dict(action="limits", joint_state="identity", finger_tactile_force="identity")
        for spec in ({**good, "action": "identity"}, {**good, "finger_tactile_force": "limits"},
                     {**good, "extra": "identity"}, {k: v for k, v in good.items() if k != "action"},
                     *({**good, "joint_state": mode} for mode in ("gaussian", "rms", "max_abs"))):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                normalization_feature_shapes(dims, spec)
        with self.assertRaisesRegex(RuntimeError, "Missing affine field"):
            fit_normalizer(None, {"action": "limits"}, LinearNormalizer())
        cfg = load_policy_config("dp3", ["normalization.joint_state=identity"])
        validate_normalization(cfg)
        cfg.normalization.action = "identity"
        with self.assertRaisesRegex(ValueError, "action must use limits"):
            validate_normalization(cfg)

    def test_near_constant_fp32_and_action_ee(self):
        normalizer = LinearNormalizer()
        values = torch.tensor([[2., 1.], [2.00001, 3.]], dtype=torch.float64)
        normalizer.fit_field("action", values.split(1))
        self.assertEqual(normalizer["action"].scale.dtype, torch.float32)
        self.assertEqual(normalizer["action"].scale[0], 1)
        torch.testing.assert_close(normalizer["action"].unnormalize(normalizer["action"].normalize(values)), values.float())
        actions = torch.randn(10, 21)
        dataset = SimpleNamespace(action_key="action_ee", iter_normalization_data=lambda key: actions.split(3))
        normalizer = LinearNormalizer({"action": (21,)})
        fit_normalizer(dataset, {"action": "limits"}, normalizer)
        torch.testing.assert_close(normalizer["action"].scale[3:9], torch.ones(6))
        torch.testing.assert_close(normalizer["action"].offset[3:9], torch.zeros(6))
        torch.testing.assert_close(normalizer["action"].unnormalize(normalizer["action"].normalize(actions)), actions)

    def test_tactile_shape_contracts(self):
        from dexmani_policy.common.inference import prepare_observations

        for field, shape in (("finger_tactile_force", (5, 3)), ("taxel_tactile_force", (5, 120, 3))):
            good = np.zeros((2, *shape), dtype=np.float32)
            options = dict(normalization_dims={field: int(np.prod(shape)), "action": 19},
                           n_obs_steps=2, rgb_preprocessing={}, device="cpu")
            result = prepare_observations({field: good}, **options)
            self.assertEqual(result[field].shape, (1, 2, *shape))
            with self.assertRaisesRegex(ValueError, "trailing shape"):
                prepare_observations({field: good.reshape(2, -1)}, **options)
        with tempfile.TemporaryDirectory() as directory:
            cfg, source = make_policy_fixture(Path(directory), "adaptac", "taxel")
            del source["data/taxel_tactile_force"]
            source.create_dataset("data/taxel_tactile_force", data=np.zeros((12, 5, 120, 3), dtype=np.int32))
            with self.assertRaisesRegex(InstantiationException, "floating point") as caught:
                build_dataset(cfg)
            self.assertIsInstance(caught.exception.__cause__, ValueError)

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
            spec = dict(point_cloud="identity", joint_state="identity", finger_tactile_force="identity",
                        taxel_tactile_force="identity")
            dims = dict(joint_state=19, point_cloud=6, finger_tactile_force=15, taxel_tactile_force=1800)
            identity = LinearNormalizer(normalization_feature_shapes(dims, spec))
            with patch.object(dataset, "iter_normalization_data", side_effect=AssertionError("identity must not fit")):
                fit_normalizer(dataset, spec, identity)
            self.assertEqual(set(identity.fields), set())
            self.assertEqual(identity.state_dict(), {})
            self.assertEqual(list(identity.buffers()), [])
            from dexmani_policy.agents.policies.base import Policy
            model = Policy(torch.nn.Identity(), torch.nn.Identity(), 4, 2, 2, 19)
            spec["action"], dims["action"] = "limits", 19
            model.configure_normalizer(dims, normalization_feature_shapes(dims, spec))
            obs = dict(joint_state=torch.randn(2, 2, 19), finger_tactile_force=torch.randn(2, 2, 5, 3))
            with patch.object(FieldNormalizer, "normalize", side_effect=AssertionError("identity must bypass affine")):
                pre = model._preprocess_obs(obs)
            for key, value in obs.items():
                self.assertTrue(torch.equal(pre[key], value.flatten(0, 1)))
                self.assertEqual(pre[key].data_ptr(), value.data_ptr())

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
            with self.assertRaisesRegex(RuntimeError, "Missing affine field"):
                fit_normalizer(dataset, cfg.normalization, LinearNormalizer({"action": (19,)}),
                               initial_state={}, inherited_fields=("point_cloud",))
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                fit_normalizer(dataset, cfg.normalization, fitted,
                               initial_state=LinearNormalizer({"point_cloud": (3,)}).state_dict(),
                               inherited_fields=("point_cloud",))



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
        for index, state in enumerate((saved.model_state, saved.ema_model_state)):
            state["normalizer.fields.action.scale"].fill_(2 + index)
            state["normalizer.fields.action.offset"].fill_(-0.75 - index)
            deployed = restore_policy(saved.effective_config, state, "cpu")
            self.assertEqual(set(deployed.normalizer.fields), {"action", "joint_state"})
            for name, value in deployed.normalizer.state_dict().items():
                torch.testing.assert_close(value, state[f"normalizer.{name}"], rtol=0, atol=0)
        with patch.object(self.dataset, "iter_normalization_data", side_effect=AssertionError("resume must not fit")):
            with closing(PolicyTrainer(OmegaConf.create(saved.effective_config), self.dataset,
                                       self.description, checkpoint=saved)) as resumed:
                for model, state in ((resumed.model, saved.model_state), (resumed.ema_model, saved.ema_model_state)):
                    for name, value in model.normalizer.state_dict().items():
                        torch.testing.assert_close(value, state[f"normalizer.{name}"], rtol=0, atol=0)

    def test_restore_rejects_invalid_config(self):
        for key, mode in (("action", "identity"), ("finger_tactile_force", "limits"),
                          *(("joint_state", mode) for mode in ("gaussian", "rms", "max_abs"))):
            saved = copy.deepcopy(self.checkpoint)
            saved.effective_config["normalization"][key] = mode
            with self.subTest(key=key, mode=mode):
                with self.assertRaises(ValueError):
                    restore_policy(saved.effective_config, saved.model_state, "cpu")
                with self.assertRaises(ValueError):
                    PolicyTrainer(OmegaConf.create(saved.effective_config), self.dataset,
                                  self.description, checkpoint=saved)
        cfg = copy.deepcopy(self.checkpoint.effective_config)
        del cfg["normalization_dims"]["action"]
        del cfg["normalization"]["action"]
        with self.assertRaisesRegex(ValueError, "include action"):
            restore_policy(cfg, self.checkpoint.model_state, "cpu")

    def test_interact_zero_force_residual(self):
        model = restore_policy(self.checkpoint.effective_config, self.checkpoint.model_state, "cpu")
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
        for field in ("action", "joint_state"):
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


class PolicyRegressionTest(unittest.TestCase):
    def test_pretrained_identity_does_not_inherit_affine(self):
        for recipe, key in (("afro", "point_cloud"), ("cordvip", "joint_state")):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                cfg, _ = make_policy_fixture(Path(directory), recipe)
                source = cfg.initialization.encoder
                payload = torch.load(source, weights_only=True)
                payload["normalization"][key] = "identity"
                for part in ("scale", "offset"):
                    del payload["normalizer_state"][f"fields.{key}.{part}"]
                torch.save(payload, source)
                with closing(build_trainer(cfg)) as trainer:
                    self.assertEqual(trainer.cfg.normalization[key], "identity")
                    self.assertNotIn(key, trainer.model.normalizer.fields)

    def test_policy_training_resume_and_inference(self):
        for recipe, tactile in (("dp3", "force"), ("adaptac", "force"), ("adaptac", "taxel"),
                                ("adaptac", "patch"), ("interact_dex", "force"), ("dexlatent", "force"),
                                ("dqrise", "force"), ("afro", "force"), ("cordvip", "force")):
            with self.subTest(recipe=recipe, tactile=tactile), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg, _ = make_policy_fixture(root, recipe, tactile)
                run_smoke(cfg, root / "run", max_updates=1, resume_updates=1)
                saved = load_checkpoint(root / "run/checkpoints/latest.pt")
                dataset, _ = build_dataset(cfg)
                obs = {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}
                for state in (saved.model_state, saved.ema_model_state):
                    if state is None:
                        continue
                    with patch("dexmani_policy.common.normalizer.LinearNormalizer.fit_field",
                               side_effect=AssertionError("restore must not fit")):
                        model = restore_policy(saved.effective_config, state, "cpu")
                    expected = {key for key, mode in cfg.normalization.items() if mode == "limits"}
                    if recipe == "dexlatent":
                        expected.add("hand_latent")
                        self.assertEqual(model.normalizer["hand_latent"].scale.shape, (32,))
                        latents = torch.cat([model._encode_actions(chunk)[1] for chunk in dataset.iter_normalization_data("action")])
                        reference = LinearNormalizer()
                        reference.fit_field("hand_latent", [latents])
                        torch.testing.assert_close(model.normalizer["hand_latent"].state_dict(), reference["hand_latent"].state_dict())
                    self.assertEqual(set(model.normalizer.fields), expected)
                    for key, value in model.normalizer.state_dict().items():
                        torch.testing.assert_close(value, state[f"normalizer.{key}"], rtol=0, atol=0)
                    if recipe == "dqrise":
                        torch.testing.assert_close(model.normalizer["action"].scale[7:], torch.linspace(1, 3, 12))
                        torch.testing.assert_close(model.normalizer["action"].offset[7:], torch.linspace(-2, 1, 12))
                    if recipe in {"afro", "cordvip"}:
                        inherited = ("point_cloud",) if recipe == "afro" else ("hand_pointcloud", "object_pointcloud", "joint_state")
                        for key in inherited:
                            torch.testing.assert_close(model.normalizer[key].scale, torch.full_like(model.normalizer[key].scale, 2.5))
                            torch.testing.assert_close(model.normalizer[key].offset, torch.full_like(model.normalizer[key].offset, -0.3))
                    prediction = model.predict_action(obs)["future_actions"]
                    self.assertEqual(prediction.shape, (1, 3, 19))
                    self.assertTrue(torch.isfinite(prediction).all())
                    if model.supports_rtc:
                        rtc = model.predict_action(obs, rtc_prefix=prediction[:, :1], delay_steps=1,
                                                   rtc_guidance_cap=1.)["future_actions"]
                        self.assertTrue(torch.isfinite(rtc).all())

    def test_tactile_raw_inputs_targets_and_patch_units(self):
        for tactile in ("force", "taxel", "patch"):
            with self.subTest(tactile=tactile), tempfile.TemporaryDirectory() as directory:
                cfg, source = make_policy_fixture(Path(directory), "adaptac", tactile)
                with closing(build_trainer(cfg)) as trainer:
                    model = trainer.model.eval()
                    batch = next(iter(trainer.train_loader))
                    key = model.force_key
                    self.assertNotIn(key, model.normalizer.fields)
                    self.assertEqual(model.normalization_dims[key], 15 if tactile == "force" else 1800)
                    pre = model._preprocess_obs(batch["obs"])[key]
                    torch.testing.assert_close(pre, batch["obs"][key].flatten(0, 1), rtol=0, atol=0)
                    received = []
                    handle = model.obs_encoder.tactile_encoder.register_forward_pre_hook(lambda module, args: received.append(args[0]))
                    _, current, _ = model._encode_force_inputs(batch["obs"])
                    handle.remove()
                    torch.testing.assert_close(received[0], pre, rtol=0, atol=0)
                    torch.testing.assert_close(current, batch["obs"][key].flatten(2), rtol=0, atol=0)
                    with patch.object(model.force_predictor, "compute_loss_and_prediction",
                                      wraps=model.force_predictor.compute_loss_and_prediction) as predict:
                        model.compute_loss(batch)
                    torch.testing.assert_close(predict.call_args.args[1], batch["future_force"].flatten(2), rtol=0, atol=0)
                    dataset = trainer.train_loader.dataset
                    query = dataset.sampler.sample_source_rows(0)[dataset.action_start_index]
                    torch.testing.assert_close(dataset[0]["future_force"][0], torch.from_numpy(source[f"data/{key}"][query + 1]), rtol=0, atol=0)
                    if tactile == "patch":
                        encoder = model.obs_encoder.tactile_encoder
                        zero = encoder._patch_descriptors(torch.zeros_like(pre))
                        self.assertEqual(torch.count_nonzero(zero), 0)
                        force = torch.zeros_like(pre)
                        force[..., 0] = encoder.activity_scale
                        descriptors = encoder._patch_descriptors(force)
                        torch.testing.assert_close(descriptors[..., 6], torch.full_like(descriptors[..., 6], 0.5))

    def test_interact_raw_loss_mask_and_gradient_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, source = make_policy_fixture(Path(directory), "interact_dex")
            with closing(build_trainer(cfg)) as trainer:
                model = trainer.model
                batch = next(iter(trainer.train_loader))
                dataset = trainer.train_loader.dataset
                query = dataset.sampler.sample_source_rows(0)[dataset.action_start_index]
                torch.testing.assert_close(dataset[0]["future_force"],
                                           torch.from_numpy(source["data/finger_tactile_force"][query + cfg.future_force_horizon]),
                                           rtol=0, atol=0)
                received = []
                handle = model.obs_encoder.tactile_fusion.register_forward_pre_hook(lambda module, args: received.append(args[1]))
                features = model._encode_features(batch["obs"])
                handle.remove()
                torch.testing.assert_close(received[0], batch["obs"]["finger_tactile_force"].flatten(0, 1), rtol=0, atol=0)
                prediction = model.future_force_head(features.interaction)
                batch["future_force_valid"] = torch.tensor([True, False])
                loss, metrics = model._compute_future_force_loss(prediction, batch, log_metrics=True)
                expected = torch.nn.functional.smooth_l1_loss(prediction[0], batch["future_force"][0])
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(metrics["future_force_mae_raw"], (prediction[0] - batch["future_force"][0]).abs().mean())
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.obs_encoder.cross_attention.parameters()))
                for module in (model.action_decoder, model.obs_encoder.joint_encoder, model.obs_encoder.scene_pool):
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                batch["future_force_valid"].zero_()
                loss, _ = model._compute_future_force_loss(prediction, batch, log_metrics=False)
                self.assertEqual(loss, 0)
                with patch.object(model.future_force_head, "forward", side_effect=AssertionError("inference must skip future head")):
                    model.eval().predict_action(batch["obs"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
