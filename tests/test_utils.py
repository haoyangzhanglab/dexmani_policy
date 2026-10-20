"""CPU regression checks for utility boundaries and embedded hand representations."""

import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.core.normalizer import AffineNormalizer, FieldNormalizer
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior, load_motion_prior
from dexmani_policy.agents.hand_prior.temporal_vae import TemporalHandVAE, load_temporal_vae
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.hand_prior.temporal_codec_dataset import TemporalCodecDataset
from dexmani_policy.datasets.point_cloud_dataset import PointCloudDataset
from dexmani_policy.training.builder import validate_hand_pretraining
from dexmani_policy.training.common.checkpoint import (
    CheckpointStore,
    TrainCheckpoint,
    restore_policy_agent,
)
from dexmani_policy.training.pretrain.hand_prior.export_codebook import export_codebook
from dexmani_policy.training.pretrain.hand_prior.lamp import save_prior
from dexmani_policy.training.pretrain.hand_prior.temporal_vae import save_codec
from dexmani_policy.training.pretrain.hand_prior.vqvae import main as train_vq, save_checkpoint
from dexmani_policy.training.resume import restore_model_weights
from dexmani_policy.utils.config import (
    load_policy_config,
    resolve_dataset_paths,
    resolve_input_recipe,
    validate_action_key_consistency,
    validate_source_recipe,
)
from dexmani_policy.utils.hand_representation import (
    describe_hand_data,
    validate_kl_schedule,
)
from dexmani_policy.utils.runtime import (
    atomic_path,
    dict_apply,
    ensure_tensor,
    get_rng_state,
    set_rng_state,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"action": "limits", "joint_state": "identity", "point_cloud": "identity"}


def make_dataset(path, kind="lamp", **overrides):
    if not path.exists():
        group = zarr.open_group(str(path), mode="w")
        rng = np.random.default_rng(42)
        for key, shape in (
            ("action", (24, 19)),
            ("joint_state", (24, 19)),
            ("point_cloud", (24, 8, 6)),
        ):
            group.create_dataset("data/" + key, data=rng.uniform(-0.5, 0.5, shape).astype("f4"))
        group.create_dataset("meta/episode_ends", data=np.array([8, 16, 24], dtype="i8"))
        group.attrs.update(dt=0.05, data_revision="utils-test", episode_ids=["a", "b", "c"])
    cls = {"lamp": LAMPDataset, "temporal": TemporalCodecDataset, "dqrise": PointCloudDataset}[kind]
    options = dict(
        horizon=1 if kind == "lamp" else 4,
        obs_horizon=1 if kind == "lamp" else 2,
        val_ratio=0.34,
        pc_dim=6,
    )
    if kind == "lamp":
        options["history_steps"] = 2
    return cls(str(path), **(options | overrides))


def make_artifact(path, kind, dataset):
    cfg = OmegaConf.create({"normalization": SPEC})
    source = describe_hand_data(dataset)
    args = SimpleNamespace(steps=2, beta=0.1, beta_warmup=1, kl_warmup=0, kl_ramp=1)
    metrics = dict(
        beta=0.1,
        selection_split="validation" if dataset.val_mask.any() else "final_no_holdout",
        evaluation=dict(
            mse=0.2, physical_mae=0.3, reconstruction=0.2, kl=0.1, physical_joint_mae=[0.3] * 12
        ),
    )
    if kind == "lamp":
        module = HandMotionPrior(history_steps=2, hidden_dim=16)
        optimizer = torch.optim.Adam(module.parameters())
        save_prior(path, module, source, cfg, args, 2, metrics, optimizer)
        return load_motion_prior(path)[0]
    if kind == "temporal":
        module = TemporalHandVAE(horizon=4, latent_dim=3, hidden_dim=24, n_layers=1)
        save_codec(path, module, source, cfg, args, 2, metrics, {})
        return load_temporal_vae(path)[0]

    module = HandVQVAE(12, [1.0] * 12, latent_dim=4, hidden_dim=16, codebook_size=2)
    optimizer = torch.optim.Adam(module.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    normalizer = AffineNormalizer()
    normalizer["hand"] = FieldNormalizer.from_parameters(torch.ones(12), torch.zeros(12))
    metadata = dict(
        hand_data=source,
        normalization_spec=SPEC,
        selection_metric="val_mse" if dataset.val_mask.any() else "train_mse",
    )
    checkpoint = path.with_suffix(".pt")
    args = SimpleNamespace(vq_decay=0.8, threshold_ema_dead_code=0, kmeans_iters=10)
    save_checkpoint(
        checkpoint,
        epoch=2,
        vqvae=module,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer=normalizer,
        args=args,
        train_history=[0.2],
        split_metadata=metadata,
        metrics={"train_mse": 0.2, "val_mse": 0.3 if dataset.val_mask.any() else float("nan")},
    )
    exported = export_codebook(checkpoint, path)
    loaded = HandStateCodebook(12, codebook_size=2)
    loaded.load(path)
    for name, value in exported.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    return loaded


def policy_config(path, kind):
    agent = dict(
        horizon=1 if kind == "lamp" else 4,
        n_obs_steps=1 if kind == "lamp" else 2,
        n_action_steps=1,
        action_dim=19,
        state_dim=19,
        num_points=8,
        state_out_dim=8,
    )
    if kind == "dqrise":
        agent.update(
            _target_="dexmani_policy.agents.core.dqrise_policy.DQRISEPolicy",
            tcp_dim=7,
            codebook_path=str(path),
            codebook_size=2,
            encoder_type="dp3",
            pc_out_dim=16,
            down_dims=[16, 32],
            diffusion_step_embed_dim=16,
        )
    else:
        agent["pc_encoder_config"] = dict(
            channels=24, num_patches=2, group_size=2, depth=1, num_heads=4
        )
        if kind == "lamp":
            agent.update(
                _target_="dexmani_policy.agents.core.lamp_policy.LAMPPolicy",
                prior_path=str(path),
                history_steps=2,
                prior_hidden_dim=16,
            )
        else:
            agent.update(
                _target_="dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy",
                codec_path=str(path),
                latent_dim=3,
                codec_hidden_dim=24,
                codec_n_layers=1,
                n_layers=1,
                hidden_dim=24,
                n_head=4,
            )
    return OmegaConf.create({"action_key": "action", "agent": agent, "normalization": SPEC})


class UtilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dexmani-utils-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_atomic_publication(self):
        path = self.root / "state"
        path.write_text("old")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with atomic_path(path) as temporary:
                temporary.write_text("partial")
                raise RuntimeError("interrupted")
        self.assertEqual(path.read_text(), "old")
        with self.assertRaises(FileExistsError):
            with atomic_path(path, overwrite=False) as temporary:
                temporary.write_text("replacement")
        self.assertEqual(path.read_text(), "old")
        for target in (path, self.root / "new"):
            with atomic_path(target, overwrite=target == path) as temporary:
                temporary.write_text("complete")
            self.assertEqual(target.read_text(), "complete")
        with atomic_path(self.root / "latest") as temporary:
            temporary.unlink()
            temporary.symlink_to(path.name)
        self.assertEqual((self.root / "latest").read_text(), "complete")
        self.assertFalse(list(self.root.glob(".dexmani-publish-*")))

    def test_tensor_containers_and_rng(self):
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = ensure_tensor(array)
        array[0, 0] = 42
        self.assertEqual(tensor[0, 0], 42)
        self.assertIs(ensure_tensor(tensor), tensor)
        result = dict_apply({"obs": {"x": tensor}, "text": ["pick", tensor]}, lambda x: x + 1)
        self.assertEqual(result["text"][0], "pick")
        torch.testing.assert_close(result["obs"]["x"], tensor + 1)
        torch.testing.assert_close(result["text"][1], tensor + 1)
        set_seed(42)
        state = get_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        set_rng_state(state)
        self.assertEqual((random.random(), np.random.rand()), expected[:2])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))
        with self.assertRaisesRegex(ValueError, "CUDA RNG"):
            set_rng_state(state, device="cuda")
        with self.assertRaisesRegex(ValueError, "CUDA RNG"):
            set_rng_state(state | {"torch_cuda": torch.zeros(8, dtype=torch.uint8)})

    def test_config_cli_and_light_imports(self):
        command = [sys.executable, "-m", "dexmani_policy.utils.config", "--config-name", "dp"]
        path = self.root / "data with spaces"
        path.mkdir()
        result = subprocess.run(
            [*command, "--check", f"dataset.zarr_path='{path}'"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["dataset_paths"], [str(path)])
        result = subprocess.run(
            [*command, "dataset.zarr_path=robot_data/example.zarr"],
            cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(ROOT)),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout)["dataset_paths"], [str(ROOT / "robot_data/example.zarr")]
        )
        result = subprocess.run(
            [*command, "--check", f"dataset.zarr_path={path.parent / 'missing'}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing dataset directories", result.stderr)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from dexmani_policy.utils.config import resolve_dataset_paths; "
                "resolve_dataset_paths('dp', []); import dexmani_policy.deployment; "
                "assert 'torch' not in sys.modules",
            ],
            cwd=ROOT,
            check=True,
        )
        paths = resolve_dataset_paths("multitask_dit", [])
        self.assertEqual(len(paths), 2)
        self.assertEqual(resolve_dataset_paths("dp.yaml", []), resolve_dataset_paths("dp", []))
        self.assertEqual(
            resolve_dataset_paths(str(ROOT / "dexmani_policy/configs/dp.yaml"), []),
            resolve_dataset_paths("dp", []),
        )
        duplicate = [
            "dataset.datasets.0.zarr_path=/tmp/shared",
            "dataset.datasets.1.zarr_path=/tmp/shared",
        ]
        self.assertEqual(resolve_dataset_paths("multitask_dit", duplicate), [Path("/tmp/shared")])

    def test_saved_recipe_and_coordinate_boundaries(self):
        cfg = load_policy_config("dp")
        cfg.workspace.output_dir = str(self.root)
        cfg.dataset.zarr_path = str(self.root / "saved.zarr")
        saved = OmegaConf.to_container(cfg, resolve=True)
        checkpoint = self.root / "checkpoints/latest.pt"
        checkpoint.parent.mkdir()
        torch.save({"_format": "simple.v4", "effective_config": saved}, checkpoint)
        (self.root / "config.yaml").write_text("unrelated: true\n")
        overrides = [f"+resume_from={checkpoint}"]
        self.assertEqual(resolve_dataset_paths("dp3", overrides), [Path(cfg.dataset.zarr_path)])
        current = load_policy_config("dp", overrides)
        resolved = resolve_input_recipe(current, overrides=overrides + ["training.device=cpu"])
        self.assertEqual(resolved.training.device, "cpu")
        validate_source_recipe(saved, resolved)
        with self.assertRaisesRegex(ValueError, "forbidden override"):
            resolve_input_recipe(current, overrides=overrides + ["horizon=8"])
        with self.assertRaisesRegex(ValueError, "task identity mismatch"):
            resolve_input_recipe(current, overrides=overrides + ["task_name=different"])
        bad = copy.deepcopy(saved)
        bad["seed"] = 99
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            validate_source_recipe(saved, bad)
        for runner in (
            {"env_kwargs": {"control_mode": "joint"}},
            {"task_configs": [{"task_name": "pick", "env_kwargs": {"control_mode": "joint"}}]},
        ):
            with self.assertRaisesRegex(ValueError, "control_mode"):
                validate_action_key_consistency({"action_key": "action_ee", "env_runner": runner})

    def test_explicit_split_restore(self):
        path = self.root / "data.zarr"
        make_dataset(path)
        manifest = {
            "data_revision": "utils-test",
            "episode_ids": ["a", "b", "c"],
            "trial_ids": {"a": "trial1", "b": "trial2", "c": "trial3"},
            "train_ids": ["b", "a"],
            "val_ids": ["c"],
            "exclusions": [],
        }
        manifest_path = self.root / "split.json"
        manifest_path.write_text(json.dumps(manifest))
        dataset = make_dataset(path, val_ratio=0, split_manifest=str(manifest_path))
        saved = dataset.data_recipe["split_manifest"]
        self.assertEqual(set(saved), {"content"})
        self.assertEqual(saved["content"]["train_ids"], ["a", "b"])
        manifest_path.unlink()

        restored = make_dataset(path, val_ratio=0, saved_split=saved)
        np.testing.assert_array_equal(restored.train_mask, dataset.train_mask)
        np.testing.assert_array_equal(restored.val_mask, dataset.val_mask)
        self.assertEqual(restored.data_recipe["split_manifest"], saved)
        for changes in (
            {"data_revision": "changed"},
            {"episode_ids": ["b", "a", "c"]},
            {"trial_ids": {"a": "trial1", "b": "trial2", "c": "trial1"}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_dataset(path, val_ratio=0, saved_split={"content": manifest | changes})

    def test_hand_data_settings(self):
        path = self.root / "data.zarr"
        dataset = make_dataset(path)
        source = describe_hand_data(dataset)
        with patch.object(type(dataset.replay_buffer), "__getitem__", side_effect=AssertionError):
            self.assertEqual(describe_hand_data(dataset), source)
        artifact = self.root / "prior.pt"
        make_artifact(artifact, "lamp", dataset)
        cfg = policy_config(artifact, "lamp")
        model = hydra.utils.instantiate(cfg.agent)
        model.initialize_training()
        validate_hand_pretraining(cfg, model, dataset)
        for options in ({"val_ratio": 0}, {"history_steps": 3}, {"horizon": 2, "pad_after": 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "data mismatch"):
                validate_hand_pretraining(cfg, model, make_dataset(path, **options))
        group = zarr.open_group(str(path), mode="a")
        for key, value in (
            ("data_revision", "changed"),
            ("dt", 0.1),
            ("episode_ids", ["c", "b", "a"]),
        ):
            original = group.attrs[key]
            group.attrs[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "data mismatch"):
                validate_hand_pretraining(cfg, model, make_dataset(path))
            group.attrs[key] = original
        # Content edits are identified by the producer's data_revision.
        group["data"]["action"][0, 7] += 0.1
        validate_hand_pretraining(cfg, model, make_dataset(path))
        group["data"]["point_cloud"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "data mismatch"):
            validate_hand_pretraining(cfg, model, make_dataset(path))
        group.attrs["hand_units"] = "degrees"
        with self.assertRaisesRegex(ValueError, "radians"):
            describe_hand_data(make_dataset(path))

    def test_kl_schedule_and_no_holdout_export(self):
        validate_kl_schedule(2, 0.1, 2)
        for steps, beta in ((1, 0.1), (2, 0), (2, float("nan"))):
            with self.subTest(steps=steps, beta=beta), self.assertRaises(ValueError):
                validate_kl_schedule(steps, beta, 2)
        for kind in ("lamp", "temporal", "dqrise"):
            with self.subTest(kind=kind):
                dataset = make_dataset(self.root / f"{kind}.zarr", kind, val_ratio=0)
                make_artifact(
                    self.root / (kind + (".npz" if kind == "dqrise" else ".pt")), kind, dataset
                )

    def test_vq_training_without_holdout(self):
        path = self.root / "data.zarr"
        make_dataset(path, "dqrise", val_ratio=0)
        output = self.root / "vq"
        train_vq(
            [
                "--policy-config",
                "dqrise",
                "--policy-override",
                f"dataset.zarr_path={path}",
                "--policy-override",
                "action_key=action",
                "--policy-override",
                "dataset.max_train_episodes=null",
                "--policy-override",
                "horizon=4",
                "--policy-override",
                "n_action_steps=1",
                "--num_epochs",
                "1",
                "--batch_size",
                "8",
                "--latent_dim",
                "4",
                "--hidden_dim",
                "16",
                "--num_layers",
                "1",
                "--kmeans_init",
                "true",
                "--kmeans_iters",
                "2",
                "--threshold_ema_dead_code",
                "0",
                "--warmup_steps",
                "0",
                "--save_epochs",
                "1",
                "--codebook_report_epochs",
                "2",
                "--device",
                "cpu",
                "--num_workers",
                "0",
                "--output_dir",
                str(output),
            ]
        )
        codebook = export_codebook(output / "vqvae_hand_best.pt", output / "codebook.npz")
        self.assertEqual(codebook.artifact_metadata["selection"]["split"], "train_no_holdout")
        report = json.loads((output / "codebook_diagnostics.json").read_text())
        self.assertEqual(report["splits"]["held_out"]["count"], 0)

    def test_embedded_raw_ema_and_resume(self):
        for kind in ("lamp", "temporal", "dqrise"):
            with self.subTest(kind=kind):
                path = self.root / (kind + (".npz" if kind == "dqrise" else ".pt"))
                dataset = make_dataset(self.root / f"{kind}.zarr", kind)
                make_artifact(path, kind, dataset)
                cfg = policy_config(path, kind)
                cfg.dataset = {"zarr_path": dataset.zarr_path}
                model = hydra.utils.instantiate(cfg.agent)
                model.set_normalization_spec(SPEC)
                model.initialize_training()
                normalizer = AffineNormalizer()
                normalizer["action"] = FieldNormalizer.from_parameters(
                    torch.ones(19), torch.zeros(19)
                )
                model.load_normalizer_from_dataset(normalizer)
                validate_hand_pretraining(cfg, model, dataset)
                state = copy.deepcopy(model.state_dict())
                checkpoint = TrainCheckpoint(
                    epoch=0,
                    global_step=1,
                    next_micro_step=1,
                    model_state=state,
                    ema_model_state=copy.deepcopy(state),
                    optimizer_state={},
                    scheduler_state={},
                    resume_contract={},
                    ema_updater_step=1,
                    ema_decay=0.9,
                    rng_states=[get_rng_state()],
                    effective_config=OmegaConf.to_container(cfg, resolve=True),
                )
                store = CheckpointStore(self.root / kind / "checkpoints")
                checkpoint_path = store.save("latest.pt", checkpoint)
                checkpoint = store.load(checkpoint_path)
                validate_hand_pretraining(cfg, model, dataset, checkpoint=checkpoint)
                wrong_data = make_dataset(Path(dataset.zarr_path), kind, val_ratio=0)
                with self.assertRaisesRegex(ValueError, "data mismatch"):
                    validate_hand_pretraining(cfg, model, wrong_data, checkpoint=checkpoint)
                path.unlink()
                shutil.rmtree(dataset.zarr_path)
                checkpoint.effective_config.pop("hand_data")
                checkpoint_path = store.save("standalone.pt", checkpoint)
                (checkpoint_path.parent.parent / "config.yaml").write_text("unrelated: true\n")
                self.assertEqual(
                    store.load(checkpoint_path).effective_config, checkpoint.effective_config
                )
                supplied = copy.deepcopy(checkpoint.effective_config)
                supplied["action_key"] = "action_ee"
                with self.assertRaisesRegex(ValueError, "recipe mismatch"):
                    restore_policy_agent(supplied, checkpoint_path, use_ema=False, device="cpu")
                for use_ema in (False, True):
                    restored = restore_policy_agent(
                        None, checkpoint_path, use_ema=use_ema, device="cpu"
                    )
                    for name, value in restored.state_dict().items():
                        self.assertTrue(torch.equal(value, state[name]), name)
                raw, ema = (hydra.utils.instantiate(cfg.agent) for _ in range(2))
                for agent in (raw, ema):
                    agent.set_normalization_spec(SPEC)
                restore_model_weights(checkpoint, raw, ema, "cpu")
                for agent in (raw, ema):
                    for name, value in agent.state_dict().items():
                        self.assertTrue(torch.equal(value, state[name]), name)
                with torch.no_grad():
                    restored.normalizer["action"].params_dict["scale"].zero_()
                with self.assertRaises((ValueError, RuntimeError)):
                    raw.load_state_dict(restored.state_dict(), strict=True)
                    raw.validate_normalizer()

    def test_old_artifacts_and_normalizer_mismatch(self):
        for kind, loader, version in (
            ("lamp", load_motion_prior, "lamp_prior.v2"),
            ("temporal", load_temporal_vae, "temporal_hand_codec.v1"),
        ):
            path = self.root / f"{kind}.pt"
            dataset = make_dataset(self.root / f"{kind}.zarr", kind)
            make_artifact(path, kind, dataset)
            if kind == "lamp":
                model = hydra.utils.instantiate(policy_config(path, kind).agent)
                model.initialize_training()
                normalizer = AffineNormalizer()
                normalizer["action"] = FieldNormalizer.from_parameters(
                    torch.ones(19) * 2, torch.zeros(19)
                )
                with self.assertRaisesRegex(ValueError, "affine mismatch"):
                    model.load_normalizer_from_dataset(normalizer)
            payload = torch.load(path, weights_only=True)
            payload["_format"] = version
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "regenerate"):
                loader(path)
        with self.assertRaisesRegex(ValueError, "format_version=5"):
            HandVQVAE.from_checkpoint({"format_version": 4})
        path = self.root / "old.npz"
        np.savez(path, format_version=np.array(3), pose_space=np.array("affine_raw"))
        with self.assertRaisesRegex(ValueError, "v4"):
            HandStateCodebook(12).load(path)


if __name__ == "__main__":
    unittest.main()
