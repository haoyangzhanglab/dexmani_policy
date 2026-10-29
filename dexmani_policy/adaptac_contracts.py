"""AdapTac tactile numerical, data, model and deployment contracts."""

import tempfile
import time
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from dexmani_policy.common.checkpoint import load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state
from dexmani_policy.deployment.metadata import inspect_policy
from dexmani_policy.deployment.runtime import load_policy
from dexmani_policy.training.common.data import build_dataset_and_normalizer
from dexmani_policy.training.common.optimization import build_optimizer
from dexmani_policy.training.trainer import PolicyTrainer
from dexmani_policy.smoke_checks import (
    capture_policy_predictions, expect_value_error, validate_data_batch,
    validate_policy_restoration, validate_tensor_layout,
)


def validate_adaptac_batch(batch, dataset, cfg):
    validate_data_batch(batch, dataset, cfg)
    batch_size = len(batch["action"])
    obs = batch["obs"]
    validate_tensor_layout(
        batch["future_force"],
        (batch_size, cfg.horizon - cfg.n_obs_steps + 1, *obs["finger_tactile_force"].shape[-2:]),
        "future_force", torch.float32,
    )

    validate_tensor_layout(batch["future_force_valid"], batch["future_force"].shape[:2], "future_force_valid", torch.bool)
    assert batch["future_force_valid"].any(dim=1).all()


ADAPTAC_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def tactile_smoke_attrs(source, task_name="pick_apple_messy"):
    attrs = {"task_name": task_name, "dt": 0.0625}
    if source == "sim":
        attrs.update(physics_dt=0.004166666666666667, frame_skip=15, action_dim=19,
                     action_space="joint", obs_alignment="obs[t]_before_action[t]", robot_setup_json="{}")
    else:
        attrs.update(format="dexmani.real.canonical", pointcloud_config={"num_points": 32})
    return attrs


def validate_tactile_numerics():
    from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandPatchEncoder
    from dexmani_policy.datasets.common.validation import validate_tactile_source

    generator = torch.Generator().manual_seed(24)
    encoder = XHandPatchEncoder().eval()
    for shape in ((5, 3), (5, 120, 3)):
        x = torch.randn((7, *shape), generator=generator) * 7 - 2
        for mode in ("rms", "max_abs", "limits", "gaussian"):
            full, chunked = LinearNormalizer(), LinearNormalizer()
            full.fit_field("x", [x], mode=mode, last_n_dims=len(shape))
            chunked.fit_field("x", [x[:0], *x.split(2)], mode=mode, last_n_dims=len(shape))
            torch.testing.assert_close(full.state_dict(), chunked.state_dict())
            field = full["x"]
            assert field.scale.shape == shape and field.scale.dtype == torch.float32
            if mode == "gaussian":
                torch.testing.assert_close(field.scale, x.std(dim=0).reciprocal())
            elif mode == "limits":
                torch.testing.assert_close(field.scale, 2 / (x.amax(dim=0) - x.amin(dim=0)))
            z = field.normalize(x)
            torch.testing.assert_close(field.unnormalize(z), x, atol=3e-6, rtol=1e-5)
            if mode in {"rms", "max_abs"}:
                assert not field.offset.any()
                torch.testing.assert_close(field.normalize(2 * x), 2 * z)
                assert field.normalize(x * 100).abs().max() > 1
                rescaled = LinearNormalizer()
                rescaled.fit_field("x", [x * 100], mode=mode, last_n_dims=len(shape))
                z2 = rescaled["x"].normalize(x * 100)
                torch.testing.assert_close(z, z2)
                if mode == "rms":
                    torch.testing.assert_close(z.square().mean(), torch.tensor(1.0))
                if len(shape) == 3:
                    torch.testing.assert_close(encoder._patch_descriptors(z), encoder._patch_descriptors(z2))
        for mode in ("rms", "max_abs"):
            for value in (0.0, -3.0, 4.0):
                n = LinearNormalizer()
                x = torch.full((2, *shape), value)
                n.fit_field("x", [x], mode=mode, last_n_dims=len(shape))
                torch.testing.assert_close(n["x"].normalize(x), torch.full_like(x, 0 if value == 0 else value / abs(value)))
            for invalid in ([], [torch.full((1, *shape), float("nan"))],
                            [torch.full((1, *shape), float("inf"))], [torch.full((1, *shape), 1e-40)]):
                expect_value_error(lambda: LinearNormalizer().fit_field("x", invalid, mode=mode, last_n_dims=len(shape)))
    zero = torch.zeros(2, 5, 120, 3)
    descriptors = encoder._patch_descriptors(zero)
    assert descriptors.shape == (2, 5, 5, 8) and not descriptors.any()
    assert torch.isfinite(encoder(zero)).all()
    ones = torch.ones_like(zero)
    d = encoder._patch_descriptors(ones)
    torch.testing.assert_close(d[..., :6], torch.ones_like(d[..., :6]))
    torch.testing.assert_close(d[..., 6], torch.full_like(d[..., 6], 3 ** .5 / (3 ** .5 + 1)))
    labels = (torch.tensor(encoder.effective_config["region_ids"]) + 1) % encoder.num_regions
    custom = XHandPatchEncoder(region_ids=labels, activity_scale=2.0).eval()
    restored = hydra.utils.instantiate(custom.effective_config).eval()
    assert "patch_mask" not in custom.state_dict()
    restored.load_state_dict(custom.state_dict(), strict=True)
    torch.testing.assert_close(restored.patch_mask, custom.patch_mask)
    torch.testing.assert_close(restored(z), custom(z))
    for value in (0, -1, float("nan"), float("inf"), 1e-50, 1e40):
        expect_value_error(lambda: XHandPatchEncoder(activity_scale=value))
    for labels in (torch.zeros(5, 120, dtype=torch.long), torch.full((5, 120), 5), torch.zeros(5, 119)):
        expect_value_error(lambda: XHandPatchEncoder(region_ids=labels))
    for mode in ("identity", "limits", "gaussian"):
        expect_value_error(lambda: encoder.validate_normalization({"taxel_tactile_force": mode}))
    for source in ("sim", "real"):
        validate_tactile_source(tactile_smoke_attrs(source), source)
        expect_value_error(lambda: validate_tactile_source(tactile_smoke_attrs(source), "real" if source == "sim" else "sim"))
        expect_value_error(lambda: validate_tactile_source({}, source))
    return {"status": "passed", "checks": "streaming, units, shapes, affine boundaries, patch descriptors, source"}


@contextmanager
def tactile_fixture(cfg):
    from dexmani_policy.common.config import FIELD_FEATURE_SHAPES

    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    config.agent.update(dict(num_points=32, feature_dim=32, state_out_dim=16,
                             diffusion_step_embed_dim=32,
                             down_dims=[32 * 2 ** i for i in range(len(cfg.agent.down_dims))]))
    config.agent.pc_encoder_config.update(dict(channels=48, num_patches=8, group_size=8, depth=1))
    config.agent.force_predictor_config.update(dict(n_layer=1, n_emb=32))
    config.dataset.augmentation_cfg = None
    config.dataset.max_train_episodes = None
    config.dataloader.update(dict(batch_size=2, num_workers=0, pin_memory=False))
    config.workspace.wandb_cfg = None
    config.training.use_compile = False
    config.training.use_bfloat16 = False
    frames = max(24, config.horizon + 2)
    ends = [frames, 2 * frames]
    config.dataset.split_manifest = dict(episode_ends=ends, train_indices=[0], val_indices=[1])
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory(prefix="tactile-roundtrip-") as temp:
        data_path = Path(temp) / "data.zarr"
        config.zarr_path = config.dataset.zarr_path = str(data_path)
        config.workspace.output_dir = str(Path(temp) / "experiment")
        root = zarr.open_group(str(data_path), mode="w")
        root.attrs.update(tactile_smoke_attrs(cfg.tactile_source, cfg.task_name))
        root.create_dataset("meta/episode_ends", data=np.array(ends))
        for key in [*config.dataset.sensor_modalities, "action"]:
            shape = FIELD_FEATURE_SHAPES.get(key, (32, config.agent.pc_dim) if key == "point_cloud" else (19,))
            values = rng.normal(size=(ends[-1], *shape)).astype(np.float32)
            if key == "taxel_tactile_force":
                values *= 100
            elif key == "finger_tactile_force":
                values[0, 1] = -values[0, 0]
                values[0, 2:] = 0  # Zero net force remains a valid observation.
            root.create_dataset(f"data/{key}", data=values, chunks=(7, *shape))
        yield config, root


def validate_tactile_roundtrip(cfg):
    from dexmani_policy.common.config import FIELD_FEATURE_SHAPES
    from dexmani_policy.datasets.common.robot_dataset import RobotDataset
    from dexmani_policy.training.common.resume import resolve_input_recipe

    with tactile_fixture(cfg) as (config, root):
        data_path = Path(config.dataset.zarr_path)
        frames = int(root["meta/episode_ends"][0])
        # Missing aggregate and wrong dense layout fail at the data boundary.
        missing = list(config.dataset.sensor_modalities) + ["absent"]
        expect_value_error(lambda: RobotDataset(str(data_path), sensor_modalities=missing, tactile_source=cfg.tactile_source))
        if "taxel_tactile_force" in config.dataset.sensor_modalities:
            data = root["data/taxel_tactile_force"][:]
            del root["data/taxel_tactile_force"]
            root.create_dataset("data/taxel_tactile_force", data=data[:, :, :-1])
            expect_value_error(lambda: RobotDataset(str(data_path), sensor_modalities=config.dataset.sensor_modalities, tactile_source=cfg.tactile_source))
            del root["data/taxel_tactile_force"]
            root.create_dataset("data/taxel_tactile_force", data=data)
        from dexmani_policy.datasets.aux.adaptac_dataset import AdapTacDataset

        without_aggregate = [key for key in config.dataset.sensor_modalities if key != "finger_tactile_force"]
        expect_value_error(lambda: AdapTacDataset(str(data_path), tactile_source=cfg.tactile_source, sensor_modalities=without_aggregate))
        for source in (None, "???", "unknown", "sim" if cfg.tactile_source == "real" else "real"):
            expect_value_error(lambda: RobotDataset(
                str(data_path), sensor_modalities=config.dataset.sensor_modalities, tactile_source=source,
            ))
        dataset, normalizer, description = build_dataset_and_normalizer(config)
        for key in ("finger_tactile_force", "taxel_tactile_force"):
            mode = config.normalization.get(key)
            if mode in {"rms", "max_abs"}:
                values = torch.from_numpy(root[f"data/{key}"][:frames]).double()
                magnitude = values.square().mean().sqrt() if mode == "rms" else values.abs().max()
                torch.testing.assert_close(normalizer[key].scale, magnitude.reciprocal().float().expand(FIELD_FEATURE_SHAPES[key]))
        batch = default_collate([dataset[0], dataset[len(dataset) - 1]])
        validate_adaptac_batch(batch, dataset, config)
        with closing(PolicyTrainer(config, dataset, normalizer, description)) as trainer:
            model = trainer.model
            assert ("finger_tactile_force" in model.normalizer.fields) == (
                config.normalization.finger_tactile_force != "identity"
            )
            assert model.force_dim == 15 and model.force_fusion.force_proj.in_features == 15 * (config.horizon + 1)
            inputs = dict_apply(batch, lambda x: x.to(trainer.device))
            contract = validate_adaptac_model_contract(model, inputs)
            snapshot = capture_policy_predictions(model, trainer.ema_model, batch)
            trainer.save_checkpoint("contract")
            checkpoint_path = trainer.checkpoint_dir / "latest.pt"
            saved = cast(dict, OmegaConf.to_container(trainer.cfg, resolve=True))
        for override in ("tactile=sim_force", "normalization.finger_tactile_force=max_abs", "agent.force_loss_weight=0"):
            expect_value_error(lambda: resolve_input_recipe(config, overrides=[override], saved_config=saved))
        data_path.rename(data_path.parent / "unavailable.zarr")
        with patch.object(LinearNormalizer, "fit_field", side_effect=AssertionError("restore must not fit")):
            old = load_checkpoint(checkpoint_path)
            old.effective_config["normalization"]["action"] = "rms"
            old.effective_config["agent"]["clip_sample"] = False
            invalid_path = checkpoint_path.parent / "invalid_action.pt"
            save_checkpoint(invalid_path, old)
            expect_value_error(lambda: restore_policy(invalid_path, use_ema=False, device=cfg.training.device))
            restored = validate_policy_restoration(checkpoint_path, snapshot, device=torch.device(cfg.training.device))
            info = inspect_policy(config.workspace.output_dir, checkpoint="latest", weights="raw")
            observation = {k: v[0].numpy() for k, v in batch["obs"].items()}
            with closing(load_policy(info, device=cfg.training.device, seed=42)) as runtime:
                prediction = runtime.predict(observation)
                runtime.reset_episode()
                np.testing.assert_allclose(prediction, runtime.predict(observation))
                assert prediction.shape == (config.horizon, 19)
    return {"status": "passed", "source": cfg.tactile_source, "model": contract, "restoration": restored}


def validate_adaptac_data_contract():
    from dexmani_policy.datasets.aux.adaptac_dataset import AdapTacDataset

    ends = np.array([7, 10, 13, 14])
    force = (
        100 * np.arange(ends[-1])[:, None, None]
        + 10 * np.arange(5)[None, :, None] + np.arange(3)[None, None, :]
    ).astype(np.float64)
    force[1] = force[0]  # A held sample is still a valid observation row.
    force[2] = 0
    force[10:13] += 100000  # Validation outliers must not affect the affine.
    action = np.broadcast_to(np.arange(ends[-1])[:, None], (ends[-1], 19)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="adaptac-data-") as directory:
        root = zarr.open_group(directory, mode="w")
        root.attrs.update(tactile_smoke_attrs("sim"))
        root.create_dataset("meta/episode_ends", data=ends)
        root.create_dataset("data/finger_tactile_force", data=force)
        root.create_dataset("data/joint_state", data=action)
        root.create_dataset("data/action", data=action)
        dataset = AdapTacDataset(
            directory, tactile_source="sim", sensor_modalities=["joint_state", "finger_tactile_force"],
            horizon=4, obs_horizon=1, pad_after=3,
            split_manifest={
                "episode_ends": ends.tolist(), "train_indices": [0, 1, 3], "val_indices": [2],
            },
            augmentation_cfg={"state": {"noise": {"noise_std": 1.0, "prob": 1.0}}},
        )
        for view in (dataset, dataset.get_validation_dataset()):
            queries = []
            for index in range(len(view)):
                query = view.sampler.sample_source_rows(index)[0]
                end = ends[np.searchsorted(ends, query, side="right")]
                queries.append(query)
                batch = view[index]
                expected = force[np.minimum(query + np.arange(1, 5), end - 1)].astype(np.float32)
                torch.testing.assert_close(batch["future_force"], torch.from_numpy(expected))
                assert batch["future_force"].shape == (4, 5, 3)
                assert batch["future_force"].dtype == torch.float32
                torch.testing.assert_close(batch["future_force_valid"], torch.tensor(query + np.arange(1, 5) < end))
                assert batch["future_force_valid"].any()
                torch.testing.assert_close(
                    batch["future_force"].flatten(-2), torch.from_numpy(expected.reshape(4, -1)),
                )
                torch.testing.assert_close(batch["action"][0], torch.from_numpy(action[query]))
            assert not set(queries).intersection(ends - 1)
        # At query 0, the final force row is beyond action[0:4].
        torch.testing.assert_close(dataset[0]["future_force"][-1], torch.from_numpy(force[4].astype(np.float32)))
        spec = {"joint_state": "limits", "action": "limits", "finger_tactile_force": "rms"}
        normalizer = dataset.get_normalizer(spec)
        real_train = force[np.r_[0:10, 13:14]].astype(np.float32)
        observed = np.concatenate(list(dataset.iter_normalization_data("finger_tactile_force")))
        np.testing.assert_array_equal(observed, real_train)
        reference = LinearNormalizer()
        reference.fit_field("force", [real_train], mode="rms", last_n_dims=2)
        field = normalizer["finger_tactile_force"]
        for name in ("scale", "offset"):
            torch.testing.assert_close(getattr(field, name), getattr(reference["force"], name).expand(5, 3))
        assert dataset.get_validation_dataset().get_normalizer(spec) is normalizer
    return {
        "status": "passed",
        "checks": "order, t+1, beyond-action reads, boundaries, padding, singleton exclusion, train-only shared affine",
    }


def require_gradient(module, name):
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), name
    total = sum(g.detach().float().abs().sum().item() for g in grads)
    assert total > 0, name
    return total


def validate_adaptac_model_contract(model, batch):
    device = next(model.parameters()).device
    raw = batch["obs"]["finger_tactile_force"]
    assert not raw[0].sum(-2).any()
    batch_size = len(raw)
    encoder = model.obs_encoder.tactile_encoder
    tokens = encoder(model.preprocess(batch["obs"])[encoder.input_key])
    assert tokens.shape == (batch_size, 5, 64) and model.force_dim == 15
    assert set(model.obs_encoder.consumed_observation_fields) == {
        "joint_state", "point_cloud", encoder.input_key,
    }
    assert set(model.consumed_observation_fields) == set(batch["obs"])
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-6)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    assert model.force_predictor.noise_scheduler is not model.action_decoder.noise_scheduler
    assert model.force_predictor.noise_scheduler.config.prediction_type == "sample"
    assert not model.force_predictor.noise_scheduler.config.clip_sample
    assert model.action_decoder.noise_scheduler.config.prediction_type == "epsilon"
    loss, metrics = model.compute_loss(batch, log_metrics=False)
    assert torch.isfinite(loss) and metrics == {}
    gradient_results = {}
    for use_bf16 in (False, True):
        model.train().zero_grad(set_to_none=True)
        captured = {}

        def force_output(module, args, output):
            output.retain_grad()
            captured["prediction"] = output

        def fusion_input(module, args):
            captured["guide"] = args[2]

        def tactile_input(module, args):
            key = encoder.input_key
            expected = batch["obs"][key]
            if key in model.normalizer.fields:
                expected = model.normalizer[key].normalize(expected)
            torch.testing.assert_close(args[0], expected.flatten(0, 1))

        original = model.force_predictor.compute_loss_and_prediction

        def supervised_prediction(condition, target, valid):
            torch.testing.assert_close(target, model.normalize_force(batch["future_force"]).flatten(-2))
            loss, prediction = original(condition, target, valid)
            assert target.shape == prediction.shape == (batch_size, model.future_steps, model.force_dim)
            assert loss.dtype == torch.float32
            torch.testing.assert_close(loss, torch.nn.functional.mse_loss(prediction[valid], target[valid]))
            return loss, prediction

        with ExitStack() as stack:
            stack.enter_context(patch.object(model, "force_loss_weight", 0.0))
            stack.enter_context(model.force_predictor.model.register_forward_hook(force_output))
            stack.enter_context(model.force_fusion.register_forward_pre_hook(fusion_input))
            stack.enter_context(encoder.register_forward_pre_hook(tactile_input))
            stack.enter_context(patch.object(
                model.force_predictor, "compute_loss_and_prediction", side_effect=supervised_prediction,
            ))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss, metrics = model.compute_loss(batch)
            torch.testing.assert_close(loss.detach(), metrics["loss_action"])
            assert torch.isfinite(loss)
            loss.backward()
        prediction = captured["prediction"]
        torch.testing.assert_close(
            captured["guide"],
            torch.cat((model.normalize_force(raw).flatten(-2), prediction.float()), dim=1).flatten(1),
        )
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        assert prediction.grad.abs().sum() > 0
        assert prediction.grad[~batch["future_force_valid"]].abs().sum() > 0
        interaction = model.obs_encoder.interaction
        modules = {"force_predictor": model.force_predictor, "tactile_attention": interaction.tactile_attn,
                   "visual_attention": interaction.visual_attn, "visual_encoder": model.obs_encoder.pc_encoder,
                   "tactile_encoder": model.obs_encoder.tactile_encoder}
        sums = {name: require_gradient(module, name) for name, module in modules.items()}
        embedding_grad = interaction.finger_embedding.grad
        assert embedding_grad is not None and torch.isfinite(embedding_grad).all()
        assert embedding_grad.abs().sum() > 0
        sums["finger_embedding"] = embedding_grad.float().abs().sum().item()
        gradient_results["bf16" if use_bf16 else "fp32"] = sums

    model.eval()
    calls = []
    with ExitStack() as stack:
        force_sample = stack.enter_context(patch.object(
            model.force_predictor, "predict_action", wraps=model.force_predictor.predict_action,
        ))
        action_sample = stack.enter_context(patch.object(
            model.action_decoder, "predict_action", wraps=model.action_decoder.predict_action,
        ))
        for name, module in (("encode", model.obs_encoder), ("interaction", model.obs_encoder.interaction),
                             ("force", model.force_predictor.model), ("action", model.action_decoder.model)):
            stack.enter_context(module.register_forward_hook(
                lambda module, args, output, name=name: calls.append(name),
            ))
        generator = torch.Generator(device=device).manual_seed(42)
        rng = get_rng_state(device)
        output = model.predict_action(batch["obs"], generator=generator)
        assert force_sample.call_args.kwargs["generator"] is generator
        assert action_sample.call_args.kwargs["generator"] is generator
        assert list(output) == ["future_actions"]
        assert output["future_actions"].shape == (batch_size, model.future_steps, 19)
        force_steps = model.force_predictor.num_inference_steps
        action_steps = model.action_decoder.num_inference_steps
        assert calls == ["interaction", "encode"] + ["force"] * force_steps + ["action"] * action_steps
        generator_after = generator.get_state()
        calls.clear()
        set_rng_state(rng, device)
        generator.manual_seed(42)
        again = model.predict_action(batch["obs"], generator=generator)
        torch.testing.assert_close(output["future_actions"], again["future_actions"])
        assert torch.equal(generator_after, generator.get_state())
    return {
        "status": "passed", "action_only_gradient_l1": gradient_results,
        "optimizer_parameters": len(ids), "tactile_source": model.tactile_source,
        "sampling": f"one encoding/interaction, independent {force_steps}+{action_steps} DDIM, generator reproducibility",
    }


@torch.inference_mode()
def validate_adaptac_deployment(experiment, dataset, device):
    validation = dataset.get_validation_dataset()
    if validation is None:
        return {"status": "not_run", "reason": "no validation windows"}
    info = inspect_policy(experiment, checkpoint="latest")
    selected = np.sort(np.random.default_rng(42).choice(len(validation), min(32, len(validation)), replace=False))
    queries = validation.sampler.source_rows(validation.sampler.indices[selected])[:, 0]
    with closing(load_policy(info, device=str(device), seed=42)) as runtime:
        model = runtime.agent
        samples = [validation[int(index)] for index in selected]
        errors, masks, zero_errors, persistence_errors = [], [], [], []
        generator = torch.Generator(device=device).manual_seed(42)
        for start in range(0, len(samples), 4):
            chunk = samples[start : start + 4]
            obs = {name: torch.stack([item["obs"][name] for item in chunk]).to(device) for name in chunk[0]["obs"]}
            features = model.obs_encoder(model.preprocess(obs))
            predicted = model.force_predictor.predict_action(
                torch.cat(features[:2], dim=-1).unsqueeze(1),
                (len(chunk), model.future_steps, model.force_dim), generator=generator, dtype=torch.float32,
            )
            target = torch.stack([item["future_force"] for item in chunk])
            error = model.unnormalize_force(predicted.reshape(-1, model.future_steps, 5, 3)).cpu() - target
            masks.append(torch.stack([item["future_force_valid"] for item in chunk]))
            zero_errors.append(-target)
            persistence_errors.append(obs["finger_tactile_force"].cpu() - target)
            assert torch.isfinite(error).all()
            errors.append(error)
        error = torch.cat(errors)
        valid = torch.cat(masks)

        def force_metrics(error):
            def window_rmse(stop):
                mask = valid[:, :stop, None, None]
                return ((error[:, :stop].square() * mask).sum() / (15 * mask.sum())).sqrt().item()
            return {
                "full_horizon_rmse": window_rmse(info.future_steps),
                "execution_window_rmse": window_rmse(info.n_action_steps),
                "per_finger_rmse": ((error.square() * valid[..., None, None]).sum((0, 1, 3))
                                    / (3 * valid.sum())).sqrt().tolist(),
            }

        report = {
            "status": "passed", "samples": len(error), "seed": 42,
            "metric_kind": "pure_noise_forecasting", "valid_successors": valid.sum().item(),
            "weights": info.weights,
            "unit": "N, world axes" if model.tactile_source == "sim" else "SDK native unknown physical scale, sensor axes",
            "finger_order": ADAPTAC_FINGER_ORDER, "query_source_rows": queries.tolist(),
            "force_ddim_steps": model.force_predictor.num_inference_steps,
            "forecast": force_metrics(error), "zero": force_metrics(torch.cat(zero_errors)),
            "persistence": force_metrics(torch.cat(persistence_errors)),
        }
        observation = {name: value.numpy() for name, value in samples[0]["obs"].items()}
        for _ in range(5):
            runtime.predict(observation)
        durations = []
        for _ in range(20):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            query_start = time.perf_counter()
            runtime.predict(observation)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(1000 * (time.perf_counter() - query_start))
        report["latency"] = {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "batch_size": 1, "parameter_dtype": str(next(runtime.agent.parameters()).dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "autocast": False, "warmups": 5, "queries": len(durations),
            "mean_ms": float(np.mean(durations)), "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "includes": (
                "host staging, transfer, normalization, encoding, interaction, "
                "force DDIM, FGAF, action DDIM, CPU action output"
            ),
        }
        return report


def validate_adaptac_contracts(cfg):
    return {
        "configuration": validate_tactile_configuration(),
        "normalization_patch": validate_tactile_numerics(),
        "roundtrip": validate_tactile_roundtrip(cfg),
        "data": validate_adaptac_data_contract(),
    }




def validate_tactile_configuration():
    from dexmani_policy.common.config import load_policy_config, require_tactile_source, validate_tactile_config
    from dexmani_policy.env_runner.sim_runner import SimRunner
    from dexmani_policy.evaluation.config import merge_eval_config

    for preset, source in (("sim_force", "sim"), ("real_force", "real"), ("real_taxel", "real"), ("real_patch", "real")):
        config = load_policy_config("adaptac", [f"tactile={preset}"])
        config.workspace.output_dir = "/tmp/smoke_test_output"
        config = cast(dict, OmegaConf.to_container(config, resolve=True))
        validate_tactile_config(config)
        assert config["tactile_source"] == config["agent"]["tactile_source"] == config["dataset"]["tactile_source"] == source
        assert ("taxel_tactile_force" in config["dataset"]["sensor_modalities"]) == (preset in ("real_taxel", "real_patch"))
    for invalid_source in (None, "???", "unknown"):
        for section in (config, config["agent"], config["dataset"]):
            section["tactile_source"] = invalid_source
        expect_value_error(lambda: validate_tactile_config(config))
    for section in (config, config["agent"], config["dataset"]):
        section.pop("tactile_source")
    expect_value_error(lambda: validate_tactile_config(config))
    assert require_tactile_source(["fingertip_xyz"], None) is None
    fields = ["finger_tactile_force"]
    runner = SimRunner("unused", 1, fields, {})
    with patch.object(runner, "make_env", side_effect=AssertionError("source rejection must precede environment construction")):
        for rejected_source in (None, "???", "unknown", "real"):
            agent = SimpleNamespace(policy_name="renamed_policy", consumed_observation_fields=fields, tactile_source=rejected_source)
            expect_value_error(lambda: next(runner.run(agent, seeds=[0], inference_steps=1, action_steps=1)))
            expect_value_error(lambda: merge_eval_config({"dataset": {"sensor_modalities": fields}, "tactile_source": rejected_source}))
        for accepted_fields, accepted_source in ((fields, "sim"), (["fingertip_xyz"], None)):
            agent = SimpleNamespace(consumed_observation_fields=accepted_fields, tactile_source=accepted_source)
            assert list(runner.run(agent, seeds=[], inference_steps=1, action_steps=1)) == []
    assert not hasattr(runner, "run_episode")
    return {"status": "passed", "checks": "presets, missing/null/invalid sources, geometry, sim entry before environment"}
