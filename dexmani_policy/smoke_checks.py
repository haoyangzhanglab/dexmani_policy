"""Shared config, data, training and restoration checks."""

import importlib
import importlib.util
import json
import tempfile
import traceback
from contextlib import closing, contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint import load_checkpoint, policy_state_dict, restore_policy
from dexmani_policy.common.config import (
    POINT_CLOUD_FIELDS, load_experiment_config, load_policy_config, positive_int,
)
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state, set_seed
from dexmani_policy.training.common.config import validate_config
from dexmani_policy.training.common.data import (
    build_dataset_and_normalizer, build_train_loader, close_data_loader,
)
from dexmani_policy.training.trainer import PolicyTrainer


def load_smoke_config(config_name: str, overrides=()):
    cfg = load_policy_config(config_name, overrides)
    # Hydra compose does not create a runtime output directory.
    cfg.workspace.output_dir = "/tmp/smoke_test_output"
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    return cfg


def iter_hydra_targets(node):
    if isinstance(node, dict):
        if "_target_" in node:
            yield node["_target_"]
        for value in node.values():
            yield from iter_hydra_targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_hydra_targets(value)


def validate_target_imports(cfg) -> int:
    """Import training targets; only locate optional simulation modules."""
    targets = list(iter_hydra_targets(OmegaConf.to_container(cfg, resolve=True)))
    for target in targets:
        module_name, attribute = target.rsplit(".", 1)
        if target == cfg.agent._target_ or module_name.startswith(
            ("dexmani_policy.training.", "dexmani_policy.datasets.")
        ):
            getattr(importlib.import_module(module_name), attribute)
        elif importlib.util.find_spec(module_name) is None:
            raise ModuleNotFoundError(f"Hydra target module not found: {target}")
    return len(targets)


def validate_tensor_layout(value, shape, name, dtype=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name}: expected tensor shape {tuple(shape)}, got {getattr(value, 'shape', None)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {value.dtype}")


def iter_batch_fields(batch):
    for key, value in batch.items():
        if isinstance(value, dict):
            for field, tensor in value.items():
                yield f"{key}.{field}", tensor
        else:
            yield key, value


def observation_shape(dataset, field):
    shape = tuple(dataset.replay_buffer[field].shape[1:])
    if field == "rgb" and dataset.rgb_preprocess_size is not None:
        spatial_shape = dataset.rgb_random_crop_size or dataset.rgb_preprocess_size
        shape = (3, *spatial_shape)
    elif field in POINT_CLOUD_FIELDS:
        points = dataset.random_keep_points if field == "point_cloud" else None
        shape = (points or shape[0], dataset.pc_dim or shape[-1])
    return (dataset.obs_horizon, *shape)


def validate_data_batch(batch, dataset, cfg):
    action = batch["action"]
    batch_size = len(action)
    validate_tensor_layout(action, (batch_size, cfg.horizon, cfg.agent.action_dim), "action")

    obs = batch["obs"]
    if obs.keys() != set(dataset.sensor_modalities):
        raise ValueError("Batch observation fields differ from the dataset recipe")
    for key in dataset.sensor_modalities:
        shape = observation_shape(dataset, key)
        source_dtype = dataset.replay_buffer[key].dtype
        if key == "rgb" and dataset.rgb_preprocess_size is not None:
            dtype = torch.uint8 if dataset.rgb_keep_uint8 else torch.float32
        elif np.issubdtype(source_dtype, np.floating):
            dtype = torch.float32
        else:
            dtype = torch.from_numpy(np.empty((), dtype=source_dtype)).dtype
        validate_tensor_layout(obs[key], (batch_size, *shape), f"obs.{key}", dtype)

    context = batch.get("action_context", {})
    if not isinstance(context, dict):
        raise ValueError("Batch action_context must be a dict")
    for key, value in context.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != batch_size:
            raise ValueError(f"action_context.{key}: expected batched tensor")
        if value.dtype != torch.float32:
            raise ValueError(f"action_context.{key}: expected float32")

    for name, value in iter_batch_fields(batch):
        if value.device.type != "cpu":
            raise ValueError(f"{name}: expected CPU data")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{name}: expected finite data")


def validate_data_loader(loader, cfg, *, split, max_batches, batch_check=validate_data_batch):
    iterator = None
    checked = 0
    try:
        iterator = iter(loader)
        for checked, batch in enumerate(islice(iterator, max_batches), start=1):
            batch_check(batch, loader.dataset, cfg)
            summary = ", ".join(
                f"{key}={tuple(value.shape)} {value.dtype}"
                for key, value in iter_batch_fields(batch)
            )
            print(f"[{split} batch {checked}] {summary}")
        if checked == 0:
            raise ValueError(f"{split}: no batches to check")
    finally:
        close_resources(lambda: close_data_loader(loader, iterator))
    print(f"[{split}] windows={len(loader.dataset):,}, checked batches={checked}")


def validate_dataset(cfg, *, max_batches=2, batch_check=validate_data_batch):
    positive_int(max_batches, "max_batches")
    set_seed(cfg.training.seed)
    cfg.dataloader.pin_memory = False
    dataset, _, _ = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset)
    validate_data_loader(loader, cfg, split="train", max_batches=max_batches, batch_check=batch_check)
    validation = dataset.get_validation_dataset()
    if validation is None:
        print("[validation] 无可用验证窗口")
    else:
        loader = DataLoader(validation, batch_size=cfg.dataloader.batch_size, num_workers=0)
        validate_data_loader(loader, cfg, split="validation", max_batches=max_batches, batch_check=batch_check)


def validate_parameter_updates(model, before):
    changed = None
    for name, parameter in model.named_parameters():
        # Compilation inserts a wrapper into parameter names.
        name = name.replace("_orig_mod.", "")
        if name not in before:
            continue
        current = parameter.detach().cpu()
        if not torch.isfinite(current).all():
            raise AssertionError(f"Non-finite learned parameter: {name}")
        if changed is None and not torch.equal(before[name], current):
            changed = name
    if changed is None:
        raise AssertionError("No trainable parameter changed within the smoke budget")
    return changed


def validate_checkpoint_state(cfg, trainer, checkpoint):
    assert checkpoint.global_step == trainer.global_step
    assert checkpoint.epoch == trainer.current_epoch
    assert trainer.scheduler.state_dict() == checkpoint.scheduler_state
    optimizer_state = trainer.optimizer.state_dict()
    assert optimizer_state["param_groups"] == checkpoint.optimizer_state["param_groups"]
    torch.testing.assert_close(
        optimizer_state["state"], checkpoint.optimizer_state["state"],
        rtol=0, atol=0, check_device=False,
    )
    assert (
        checkpoint.effective_config["training"]["loop"]["total_train_steps"]
        == cfg.training.loop.total_train_steps
    )
    if trainer.ema_updater is not None:
        assert checkpoint.ema_updater_step == trainer.ema_updater.optimization_step
    for model, state in (
        (trainer.model, checkpoint.model_state),
        (trainer.ema_model, checkpoint.ema_model_state),
    ):
        if model is not None:
            torch.testing.assert_close(
                policy_state_dict(model), state, rtol=0, atol=0, check_device=False
            )


def capture_policy_predictions(model, ema_model, batch):
    """Capture CPU inputs/predictions while preserving model modes, backbones and RNG."""
    device = next(model.parameters()).device
    obs = dict_apply(
        {key: value[:1] for key, value in batch["obs"].items()},
        lambda x: x.detach().to("cpu", copy=True),
    )
    prediction_kwargs = {}
    if "action_context" in batch:
        prediction_kwargs["action_context"] = dict_apply(
            batch["action_context"], lambda value: value[:1].detach().to("cpu", copy=True),
        )
    inputs = dict_apply(obs, lambda x: x.to(device))
    kwargs = dict_apply(prediction_kwargs, lambda x: x.to(device))
    rng_state = get_rng_state(device)
    predictions = {}
    reports: dict[str, dict[str, float]] = {}
    for use_ema, policy in ((False, model), (True, ema_model)):
        if policy is None:
            continue
        label = "ema" if use_ema else "raw"
        reports[label] = {}
        modes = [(module, module.training) for module in policy.modules()]
        backbones = [
            (parent, name, child)
            for parent in policy.modules() for name, child in parent.named_children()
            if hasattr(child, "_orig_mod")
        ]
        try:
            policy.eval()
            set_rng_state(rng_state, device)
            with torch.inference_mode():
                prediction = policy.predict_action(inputs, **kwargs)["future_actions"]
            assert prediction.shape == (1, policy.future_steps, policy.control_action_dim)
            assert torch.isfinite(prediction).all()
            if backbones:
                # Compare restoration using eager kernels for every compiled child.
                for parent, name, backbone in backbones:
                    setattr(parent, name, backbone._orig_mod)
                set_rng_state(rng_state, device)
                with torch.inference_mode():
                    eager = policy.predict_action(inputs, **kwargs)["future_actions"]
                difference = (prediction - eager).abs().max().item()
                reports[label]["compiled_eager_max_abs_diff"] = difference
                print(f"{label} compiled/eager prediction max abs diff: {difference:.8g}")
                prediction = eager
            predictions[use_ema] = prediction.to("cpu", copy=True)
        finally:
            for parent, name, backbone in backbones:
                setattr(parent, name, backbone)
            for module, training in modes:
                module.training = training
            set_rng_state(rng_state, device)
    return dict(
        obs=obs, prediction_kwargs=prediction_kwargs, rng_state=rng_state,
        predictions=predictions, reports=reports,
    )


def validate_policy_restoration(saved_config, checkpoint_path, snapshot, *, device):
    """Check restored raw/EMA policies without owning or modifying a Trainer."""
    obs = dict_apply(snapshot["obs"], lambda x: x.to(device))
    kwargs = dict_apply(snapshot["prediction_kwargs"], lambda x: x.to(device))
    reports = {name: dict(values) for name, values in snapshot["reports"].items()}
    rng_state = get_rng_state(device)
    try:
        for use_ema, expected in snapshot["predictions"].items():
            restored = restore_policy(saved_config, checkpoint_path, use_ema=use_ema, device=device)
            set_rng_state(snapshot["rng_state"], device)
            with torch.inference_mode():
                actual = restored.predict_action(obs, **kwargs)["future_actions"].cpu()
            torch.testing.assert_close(actual, expected)
            reports["ema" if use_ema else "raw"]["eager_restore_max_abs_diff"] = (
                actual - expected
            ).abs().max().item()
            del restored
    finally:
        set_rng_state(rng_state, device)
    return reports


@contextmanager
def smoke_output(output_dir):
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as temporary:
            yield Path(temporary)
    else:
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        yield directory


def validate_training(
    cfg, *, max_updates=4, resume_updates=2, rollout=False, output_dir=None,
    batch_check=validate_data_batch, deployment_check=None,
):
    """Run the same bounded training/restoration checks for every policy."""
    positive_int(max_updates, "max_updates")
    if resume_updates < 0:
        raise ValueError("resume_updates must be nonnegative")
    device = torch.device(cfg.training.device)
    set_seed(cfg.training.seed)
    with smoke_output(output_dir) as directory:
        report: dict[str, Any] = {"device": str(device), "training": []}
        try:
            checkpoint = None
            source = None
            run_cfg = OmegaConf.create(cfg)
            phases = [("train", max_updates)]
            if resume_updates:
                phases.append(("resume", resume_updates))
            for phase, budget in phases:
                if checkpoint is not None:
                    total_steps = checkpoint.effective_config["training"]["loop"]["total_train_steps"]
                    if checkpoint.global_step == total_steps:
                        report["resume"] = {"status": "not_run", "reason": "training budget exhausted"}
                        break
                    run_cfg = OmegaConf.create(checkpoint.effective_config)
                with open_dict(run_cfg):
                    run_cfg.workspace.output_dir = str(directory / phase)
                    run_cfg.workspace.wandb_cfg = None
                    if source is not None:
                        run_cfg.resume_from = str(source)
                with (patch.object(LinearNormalizer, "fit_field", side_effect=AssertionError("resume must not fit"))
                      if checkpoint is not None else nullcontext()):
                    dataset, normalizer, description = build_dataset_and_normalizer(
                        run_cfg, resume_checkpoint=checkpoint,
                    )
                with closing(PolicyTrainer(
                    run_cfg, dataset, normalizer, description, checkpoint=checkpoint,
                )) as trainer:
                    if checkpoint is not None:
                        validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                    checkpoint = None
                    start_step = trainer.global_step
                    before = {
                        name: parameter.detach().to(device="cpu", copy=True)
                        for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
                    }
                    trainer.compile_models()
                    trainer.train(max_updates=budget)
                    assert trainer.global_step == min(start_step + budget, trainer.total_train_steps)
                    changed = validate_parameter_updates(trainer.model, before)
                    del before
                    source = (trainer.checkpoint_dir / "latest.pt").resolve()
                    checkpoint = load_checkpoint(source, mmap=True)
                    validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
                    batch_check(batch, dataset, trainer.cfg)
                    snapshot = capture_policy_predictions(trainer.model, trainer.ema_model, batch)
                    validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                del trainer
                experiment = source.parent.parent
                restored = validate_policy_restoration(
                    load_experiment_config(experiment), source, snapshot, device=device,
                )
                report["training"].append({
                    "phase": phase, "updates": checkpoint.global_step - start_step,
                    "global_step": checkpoint.global_step, "updated_parameter": changed,
                    "restoration": restored,
                })
                print(f"[{phase}] step={checkpoint.global_step}; updated {changed}", flush=True)
            if deployment_check is not None:
                report["deployment"] = deployment_check(experiment, dataset, device)
            if rollout:
                report["simulation"] = validate_simulation(experiment, device)
        except BaseException:
            report["error"] = traceback.format_exc()
            raise
        finally:
            close_resources(lambda: (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n"))
        if output_dir is not None:
            print(f"Smoke report: {directory / 'report.json'}", flush=True)
        return report


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def validate_simulation(experiment, device):
    from dexmani_policy.evaluation.config import load_eval_config
    from dexmani_policy.evaluation.rollout import evaluate_policy

    request = load_eval_config(
        experiment, "latest", OmegaConf.from_dotlist([
            "eval.episodes=1", "eval.video.enabled=false", f"training.device={device}",
        ]),
    )
    record, path = evaluate_policy(request, output_root=experiment / "eval_dexsim")
    return {"status": "passed", "statistics": record["statistics"],
            "episodes": record["episodes"], "artifacts": str(path)}


def validate_action_recipes():
    import hydra
    from dexmani_policy.agents.policies.base import BasePolicy
    from dexmani_policy.common.config import validate_normalization_config

    names = ('dp', 'dp3', 'maniflow', 'dqrise', 'lamp', 'latent_flow', 'r3d', 'sat', 'afro', 'cordvip', 'adaptac')
    for name in names:
        cfg = load_policy_config(name)
        cfg.policy_name = 'display_name_is_not_a_recipe'
        validate_normalization_config(cfg)
        cls: type[BasePolicy] = hydra.utils.get_class(cfg.agent._target_)
        if cfg.normalization.action != 'identity':
            cfg.normalization.action = 'limits'
            validate_normalization_config(cfg)
        for mode in ('rms', 'max_abs', 'gaussian'):
            cfg.normalization.action = mode
            if 'clip_sample' in cfg.agent:
                cfg.agent.clip_sample = False
            expect_value_error(lambda: validate_normalization_config(cfg))
            expect_value_error(lambda: cls.validate_normalization_recipe(cfg.normalization, action_key=cfg.action_key))
        if 'action_ee' in cls.supported_action_keys:
            cfg.action_key = 'action_ee'
            cfg.normalization.action = 'auto'
            validate_normalization_config(cfg)
            cfg.normalization.action = 'limits'
            expect_value_error(lambda: validate_normalization_config(cfg))
    model = BasePolicy(torch.nn.Identity(), torch.nn.Identity(), 4, 1, 2, 19)
    for mode in ('rms', 'max_abs', 'gaussian', 'identity'):
        expect_value_error(lambda: model.set_normalization_spec({'action': mode}, {'action': 19}, action_key='action'))
    model.set_normalization_spec({'action': 'auto'}, {'action': 19}, action_key='action')
    expect_value_error(lambda: model.set_normalization_spec({'action': 'limits'}, {'action': 19}, action_key='action_ee'))
    # The mathematical normalizer remains independent of policy recipes.
    normalizer = LinearNormalizer()
    values = torch.tensor([[1., -2.], [3., 4.]])
    normalizer.fit_field('action', [values], mode='rms')
    torch.testing.assert_close(normalizer['action'].scale, values.double().square().mean().sqrt().reciprocal().float().expand(2))
    return {'status': 'passed', 'policies': list(names)}


def validate_ema_updates(device):
    import copy
    from dexmani_policy.training.common.optimization import EMA

    class Model(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.first = torch.nn.Linear(3, 2, device=device, dtype=dtype)
            self.frozen = torch.nn.Parameter(torch.arange(3., device=device), requires_grad=False)
            self.last = torch.nn.Linear(2, 1, device=device, dtype=torch.float64)
            self.register_buffer('normalizer', torch.tensor([2., 3.], device=device))
            self.register_buffer('counter', torch.tensor(5, device=device))

    cases = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for fixed_decay in (None, 0., .85):
            raw = Model(dtype)
            ema = EMA(copy.deepcopy(raw), fixed_decay=fixed_decay)
            expected = copy.deepcopy(ema.averaged_model)
            raw.first = cast(torch.nn.Linear, torch.compile(raw.first, backend="eager"))
            ema.averaged_model.first = torch.compile(ema.averaged_model.first, backend='eager')
            for step in range(5):
                with torch.no_grad():
                    for index, parameter in enumerate(raw.parameters()):
                        if parameter.requires_grad:
                            parameter.fill_((step - 2) * .17 + index * .3)
                    raw.counter.add_(1)
                    for source, target in zip(raw.parameters(), expected.parameters()):
                        if source.requires_grad:
                            decay = ema.get_decay(step)
                            target.mul_(decay).add_(source.to(target), alpha=1 - decay)
                ema.step(raw)
                assert ema.optimization_step == step + 1
                torch.testing.assert_close(policy_state_dict(ema.averaged_model), expected.state_dict(), rtol=1e-5, atol=1e-6)
                assert all(p.grad is None and not p.requires_grad for p in ema.averaged_model.parameters())
                assert ema.averaged_model.first.weight.dtype == torch.float32
                torch.testing.assert_close(ema.averaged_model.normalizer, raw.normalizer, rtol=0, atol=0)
                if step == 2:
                    # Check the next update after restoring the existing state/count fields.
                    restored = EMA(Model(dtype), fixed_decay=fixed_decay)
                    restored.averaged_model.load_state_dict(policy_state_dict(ema.averaged_model))
                    restored.optimization_step = ema.optimization_step
                    ema = restored
            cases += 1
    return {'status': 'passed', 'device': str(device), 'cases': cases,
            'checks': 'old formula, warmup/fixed decay, frozen/buffers, mixed precision, compile pairing, resume'}


def validate_fps(device):
    from itertools import product
    from dexmani_policy.agents.obs_encoder.pointcloud.ops import farthest_point_sample, load_pointcloud_ops

    try:
        backend = load_pointcloud_ops()
    except ImportError as error:
        return {'status': 'not_run', 'reason': str(error)}
    def rng_state():
        return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.empty(0))

    count = 0
    for channels, duplicate in product((3, 6), (False, True)):
        cloud = torch.arange(2 * 16 * channels, device=device, dtype=torch.float32).reshape(2, 16, channels) / 13
        if duplicate:
            cloud[:, 4:8, :3] = cloud[:, :1, :3]
        for random, start, shuffle, noise in product((False, True), (False, True), (False, True), (0., .01)):
            options: dict[str, Any] = dict(use_random=random, use_random_start=start, use_shuffle_output=shuffle, random_noise_scale=noise)
            set_seed(123)
            before = rng_state()
            points, indices = farthest_point_sample(cloud, 7, **options)
            after = rng_state()
            assert points.shape == (2, 7, channels) and indices.shape == (2, 7)
            assert indices.dtype == torch.int64 and ((indices >= 0) & (indices < 16)).all()
            torch.testing.assert_close(points, cloud.gather(1, indices[..., None].expand(-1, -1, channels)), rtol=0, atol=0)
            if not random:
                torch.testing.assert_close(after, before, rtol=0, atol=0)
                _, expected = backend.sample_farthest_points(cloud[..., :3], K=7, random_start_point=False)
                torch.testing.assert_close(indices, expected, rtol=0, atol=0)
            set_seed(123)
            again = farthest_point_sample(cloud, 7, **options)
            torch.testing.assert_close((points, indices), again, rtol=0, atol=0)
            torch.testing.assert_close(after, rng_state(), rtol=0, atol=0)
            count += 1
    return {'status': 'passed', 'backend': 'native PyTorch3D', 'device': str(device), 'cases': count}


def validate_dataset_layouts():
    import zarr
    from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS
    from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
    from dexmani_policy.datasets.common.robot_dataset import RobotDataset
    from dexmani_policy.datasets.aux.r3d_dataset import R3DAuxDataset
    from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
    from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
    from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_policy_data

    split = dict(episode_ends=[8, 16, 24], train_indices=[0], val_indices=[1])
    rng = np.random.default_rng(3)
    arrays = {name: rng.normal(size=shape).astype(np.float32) for name, shape in {
        'action': (24, 19), 'action_ee': (24, 21), 'joint_state': (24, 19), 'point_cloud': (24, 5, 6),
    }.items()}
    windows = 0
    with tempfile.TemporaryDirectory(prefix='dataset-contract-') as directory:
        root = zarr.open_group(directory, mode='w')
        root.attrs.update(dt=.0625)
        root.create_dataset('meta/episode_ends', data=np.array(split['episode_ends']))
        for key, values in arrays.items():
            root.create_dataset(f'data/{key}', data=values, chunks=(3, *values.shape[1:]))
        for cls in (RobotDataset, R3DAuxDataset, LatentFlowDataset):
            for pc_dim in (3, 6):
                options = dict(data_semantics=HAND_ACTION_SEMANTICS) if cls is LatentFlowDataset else dict(pad_after=2)
                dataset = cls(directory, sensor_modalities=['joint_state', 'point_cloud'], horizon=4,
                              obs_horizon=2, pad_before=1, split_manifest=split, pc_dim=pc_dim, **options)
                for view in (dataset, dataset.get_validation_dataset()):
                    for index in range(len(view)):
                        rows = view.sampler.sample_source_rows(index)
                        batch = view[index]
                        np.testing.assert_array_equal(batch['obs']['point_cloud'], arrays['point_cloud'][rows[:2], :, :pc_dim])
                        np.testing.assert_array_equal(batch['obs']['joint_state'], arrays['joint_state'][rows[:2]])
                        if cls is LatentFlowDataset:
                            expected = arrays['action'][rows[1:]]
                            np.testing.assert_array_equal(batch['action_context']['reference_state'], arrays['joint_state'][rows[1]])
                        elif cls is R3DAuxDataset:
                            expected = np.concatenate((arrays['action'][rows], arrays['action_ee'][rows, :9]), axis=-1)
                        else:
                            expected = arrays['action'][rows]
                        np.testing.assert_array_equal(batch['action'], expected)
                        windows += 1
                stats = np.concatenate(list(dataset.iter_normalization_data('action')))
                if cls is LatentFlowDataset:
                    reference = np.concatenate([
                        item['action'].numpy() - item['action_context']['reference_state'].numpy()
                        for item in (dataset[index] for index in range(len(dataset)))
                    ])
                elif cls is R3DAuxDataset:
                    reference = np.concatenate((arrays['action'][:8], arrays['action_ee'][:8, :9]), axis=-1)
                else:
                    reference = arrays['action'][:8]
                np.testing.assert_array_equal(stats, reference)
        # Joint auto/limits equivalence and EEF's unscaled rot6d segment.
        for key in ('action', 'action_ee'):
            dataset = RobotDataset(directory, sensor_modalities=['joint_state'], action_key=key, split_manifest=split)
            normalizer = dataset.get_normalizer({'action': 'auto', 'joint_state': 'limits'})
            expected = LinearNormalizer()
            expected.fit_field('action', [arrays[key][:8]], mode='limits')
            scale, offset = expected['action'].scale, expected['action'].offset
            if key == 'action_ee':
                scale[3:9] = 1
                offset[3:9] = 0
            torch.testing.assert_close(normalizer['action'].scale, scale)
            torch.testing.assert_close(normalizer['action'].offset, offset)
        cfg = load_policy_config('dqrise')
        cfg.dataset.zarr_path = directory
        cfg.dataset.split_manifest = split
        reads = {'action': 0, 'joint_state': 0}
        original = ReplayBuffer.read

        def read(buffer, key, rows, *, columns=None):
            reads[key] += rows.stop - rows.start
            return original(buffer, key, rows, columns=columns)

        with patch.object(ReplayBuffer, 'read', read):
            hand = HandChunkDataset(directory, horizon=4, data_semantics=HAND_ACTION_SEMANTICS, split_manifest=split)
        assert reads == {'action': 16, 'joint_state': 16}
        np.testing.assert_array_equal(hand.hand_actions[:16], arrays['action'][:16, 7:19])
        assert not hand.hand_actions.flags.writeable and not hand.hand_states.flags.writeable
        reads = dict.fromkeys(reads, 0)
        with patch.object(ReplayBuffer, 'read', read):
            train, val, normalizer, _ = prepare_policy_data(cfg)
        assert reads == {'action': 16, 'joint_state': 0}
        torch.testing.assert_close(normalizer['hand'].unnormalize(train), torch.from_numpy(arrays['action'][:8, 7:19]))
        torch.testing.assert_close(normalizer['hand'].unnormalize(val), torch.from_numpy(arrays['action'][8:16, 7:19]))
        for bad in (np.nan, np.inf):
            root['data/action'][9, 10] = bad
            with patch.object(LinearNormalizer, 'fit_field', side_effect=AssertionError('validate before fitting')):
                for build in (
                    lambda: HandChunkDataset(directory, horizon=4, data_semantics=HAND_ACTION_SEMANTICS, split_manifest=split),
                    lambda: prepare_policy_data(cfg),
                ):
                    try:
                        build()
                    except ValueError as error:
                        assert 'row=9, channel=(10,)' in str(error), error
                    else:
                        raise AssertionError('Invalid validation frame was accepted')
    return {'status': 'passed', 'windows': windows, 'hand_reads': 'one pass per selected field'}


def validate_common_contracts(device):
    return {
        'action_recipes': validate_action_recipes(),
        'ema': validate_ema_updates(device),
        'fps': validate_fps(device),
        'datasets': validate_dataset_layouts(),
    }
