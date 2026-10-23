"""Dataset, normalization and DataLoader construction shared by training and smoke."""

import copy
import math
from collections.abc import Mapping
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import (
    FINGER_VECTOR_FIELDS,
    normalization_dims_from_config,
    positive_int,
    resolve_normalization_spec,
    validate_depth_recipe,
)
from dexmani_policy.common.normalization import AffineNormalizer, validate_normalizer_state
from dexmani_policy.common.runtime import worker_init_fn
from dexmani_policy.datasets.common.distributed_sampler import ResumableDistributedSampler
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset


def build_normalizer(dataset, spec: dict, action_key: str) -> AffineNormalizer:
    """Fit the full buffer, preserving rot6d coordinates for action_ee auto mode."""
    normalizer = AffineNormalizer()

    for key, mode in spec.items():
        if mode == "identity":
            continue

        mixed_action = key == "action" and mode == "auto" and action_key == "action_ee"
        mode = "limits" if mode == "auto" else mode
        normalizer.fit_field(key, dataset.iter_normalization_data(key), mode=mode)
        if mixed_action:
            field = normalizer[key]
            if field.params_dict["scale"].numel() <= 9:
                raise ValueError("action_ee requires xyz, rot6d and hand coordinates")
            field.params_dict["scale"][3:9] = 1
            field.params_dict["offset"][3:9] = 0
            for name, value in (("min", -1), ("max", 1), ("mean", 0), ("std", 1)):
                field.input_stats[name][3:9] = value

    return normalizer


def build_dataset_and_normalizer(cfg, *, resume_checkpoint=None, verbose=True):
    """Build data and fit fresh statistics; full resume restores the model's normalizer."""
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        raise ValueError("Resume requires the checkpoint loaded by the training entry point")
    spec = resolve_normalization_spec(cfg)
    dataset_cfg = copy.deepcopy(cfg.dataset)
    saved_recipes = (
        resume_checkpoint.effective_config["data_recipe"] if resume_checkpoint is not None else None
    )
    if saved_recipes is not None:
        children = dataset_cfg.get("datasets", [dataset_cfg])
        if len(children) != len(saved_recipes):
            raise ValueError("Saved data_recipe task count does not match dataset")
        for child, recipe in zip(children, saved_recipes):
            if "split_manifest" in recipe:
                with open_dict(child):
                    child.saved_split = recipe["split_manifest"]
    dataset = hydra.utils.instantiate(dataset_cfg)
    children = dataset.datasets if isinstance(dataset, MultiTaskDataset) else [dataset]
    reference = children[0]
    runtime = capture_real_runtime(dataset, cfg)
    with open_dict(cfg):
        cfg.data_identity = capture_data_identity(dataset)
        # MultiTaskDataset already enforces a shared output layout.
        widths = {
            key: int(reference.replay_buffer[key].shape[-1])
            for key, mode in spec.items()
            if key != "action" and mode != "identity"
        }
        for key in FINGER_VECTOR_FIELDS.intersection(reference.sensor_modalities):
            shape = reference.replay_buffer[key].shape
            if shape[1:] != (5, 3):
                raise ValueError(f"{key} requires stored shape (T, 5, 3), got {shape}")
            if key in widths:
                widths[key] = 15
        if "point_cloud" in widths and reference.pc_dim is not None:
            widths["point_cloud"] = reference.pc_dim
        widths["action"] = int(reference.replay_buffer[reference.action_key].shape[-1]) + (
            9 if reference.use_aux_ee else 0
        )
        if widths["action"] != cfg.agent.action_dim:
            raise ValueError(
                "Dataset action width including auxiliary targets must match agent.action_dim"
            )
        cfg.normalization_dims = widths
        cfg.data_recipe = [
            dict(
                child.data_recipe,
                windows=child.sampler.validity_summary,
                observation_rows=len(child.sampler.observation_source_rows),
                action_rows=len(child.sampler.action_source_rows),
            )
            for child in children
        ]
        for index, recipe in enumerate(cfg.data_recipe):
            if saved_recipes is None:
                recipe.normalization = {
                    "population": "full_replay_buffer",
                }
            else:
                recipe.normalization = saved_recipes[index]["normalization"]
        if runtime is not None:
            cfg.real_runtime = runtime
        else:
            cfg.pop("real_runtime", None)
    normalizer = None
    if resume_checkpoint is None:
        normalizer = build_normalizer(dataset, spec, cfg.action_key)
        validate_normalizer_state(normalizer, spec, normalization_dims_from_config(cfg))
    if verbose:
        summaries = []
        for child in children:
            counts = child.sampler.validity_summary
            summary = (
                f"{Path(child.zarr_path).name}: "
                f"train episodes={int(child.train_mask.sum()):,}, windows={len(child):,}"
            )
            rejected = counts["candidate"] - counts["valid"]
            if rejected:
                summary += f", rejected={rejected:,}"
            summaries.append(summary)
        print(f"[Dataset] {' | '.join(summaries)}")
        if resume_checkpoint is None:
            print("[Normalizer] full_replay_buffer (includes held-out buffer episodes)")
    return dataset, normalizer


def capture_real_runtime(dataset, cfg) -> dict | None:
    """Capture numerical deployment inputs from the buffer actually used to train."""
    if isinstance(dataset, MultiTaskDataset):
        return None
    buffer = dataset.replay_buffer
    attrs = buffer.attrs
    if attrs.get("format") != "dexmani.real.canonical":
        if attrs.get("domain") == "real" or str(attrs.get("format", "")).startswith(
            "dexmani.real."
        ):
            raise ValueError("Real training requires format='dexmani.real.canonical'")
        return None
    if attrs.get("task_name") != cfg.task_name:
        raise ValueError("Real canonical task_name differs from the training task")
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Real canonical dt must be finite and positive")
    runtime = {"dt": float(dt)}
    if "point_cloud" in dataset.sensor_modalities:
        cloud = attrs.get("pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real point_cloud requires root pointcloud_config")
        count = positive_int(cloud.get("num_points"), "pointcloud_config.num_points")
        shape = buffer["point_cloud"].shape
        if shape[1] != count:
            raise ValueError("Stored point count disagrees with pointcloud_config")
        # Preserve the stored point count; encoder FPS selects its own subset.
        runtime["pointcloud"] = dict(cloud)
    if "hand_pointcloud" in dataset.sensor_modalities:
        cloud = attrs.get("hand_pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real hand_pointcloud requires root hand_pointcloud_config")
        count = positive_int(cloud.get("num_points"), "hand_pointcloud_config.num_points")
        data = buffer["hand_pointcloud"]
        if data.shape[1:] != (count, 6) or str(data.dtype) != "float32":
            raise ValueError("Stored hand_pointcloud must match the saved XYZRGB template")
        runtime["hand_pointcloud"] = dict(cloud)
    if "depth" in dataset.sensor_modalities:
        data = buffer["depth"]
        runtime["depth"] = validate_depth_recipe(
            {
                "shape": list(data.shape[1:]),
                "dtype": str(data.dtype),
                "unit": "raw_counts",
                "depth_scale_m": attrs["depth_scale_m_per_unit"],
            }
        )
    if "fingertip_xyz" in dataset.sensor_modalities:
        links = attrs.get("fingertip_link_names")
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Real fingertip_xyz requires five distinct non-empty link names")
        runtime["fingertip_link_names"] = list(links)
    return runtime


def capture_data_identity(dataset):
    """Record episode boundaries and selected splits for resume checks."""
    if isinstance(dataset, MultiTaskDataset):
        return {
            "tasks": {
                name: capture_data_identity(child)
                for name, child in zip(dataset.task_names, dataset.datasets)
            }
        }
    return {
        "episode_ids": dataset.replay_buffer.attrs.get("episode_ids"),
        "episode_ends": dataset.replay_buffer.episode_ends.tolist(),
        "train_episodes": dataset.train_mask.nonzero()[0].tolist(),
        "val_episodes": dataset.val_mask.nonzero()[0].tolist(),
    }


def loader_options(cfg):
    options = OmegaConf.to_container(cfg.dataloader, resolve=True)
    if options.get("num_workers", 0) == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    return options


def build_train_loader(cfg, dataset, *, rank=0, world_size=1):
    options = loader_options(cfg)
    sampler = ResumableDistributedSampler(
        dataset,
        batch_size=options["batch_size"],
        num_replicas=world_size,
        rank=rank,
        shuffle=options.pop("shuffle", False),
        seed=cfg.training.seed,
        drop_last=False,
    )
    samples = sampler.num_samples
    batch_size = options["batch_size"]
    accumulation = cfg.training.loop.gradient_accumulation_steps
    batches = (samples + batch_size - 1) // batch_size
    if (
        accumulation > 1
        and not options.get("drop_last", False)
        and samples % batch_size
        and batches % accumulation != 1
    ):
        raise ValueError(
            f"Mixed micro-batch sizes in accumulation group: "
            f"N={samples}, B={batch_size}, A={accumulation}; "
            "use drop_last=True or gradient_accumulation_steps=1"
        )
    generator = torch.Generator().manual_seed(cfg.training.seed + rank)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        generator=generator,
        worker_init_fn=worker_init_fn,
        **options,
    )
    if len(loader) == 0:
        raise ValueError(
            "Training DataLoader is empty; check dataset length, batch_size and drop_last"
        )
    return loader
