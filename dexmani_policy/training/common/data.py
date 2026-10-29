from collections.abc import Mapping
from itertools import islice
from math import isfinite, prod
from typing import Any

import hydra
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from dexmani_policy.common.config import (
    FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, normalization_feature_shapes, positive_int,
)
from dexmani_policy.common.normalizer import LinearNormalizer


def build_dataset_and_normalizer(cfg, *, resume_checkpoint=None, verbose=True):
    """Construct one admitted train/val source; resume only restores saved affine."""
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        raise ValueError("Resume requires the checkpoint loaded by the training entry point")
    spec = cfg.normalization
    dataset = hydra.utils.instantiate(cfg.dataset)
    buffer = dataset.replay_buffer
    dims = {key: int(buffer[key].shape[-1]) for key, mode in spec.items() if key != "action" and mode != "identity"}
    for key in FIELD_FEATURE_SHAPES.keys() & dims.keys():
        dims[key] = prod(buffer[key].shape[1:])
    if dataset.pc_dim is not None:
        for key in POINT_CLOUD_FIELDS.intersection(dims):
            dims[key] = dataset.pc_dim
    dims["action"] = dataset.action_dim
    if dims["action"] != cfg.agent.action_dim:
        raise ValueError("Dataset action width including auxiliary targets must match agent.action_dim")
    description = {
        "normalization_dims": dims, "split_manifest": dataset.split_manifest,
        "action_key": dataset.action_key, "attrs": buffer.attrs,
        "zarr_path": buffer.path,
        "action_start_index": dataset.action_start_index,
        "control_action_dim": buffer[dataset.action_key].shape[-1],
    }
    runtime = _describe_real_runtime(buffer, dataset.sensor_modalities, cfg.task_name)
    if runtime is not None:
        description["real_runtime"] = runtime
    if hasattr(dataset, "hand_data"):
        description["hand_data"] = dataset.hand_data
    if resume_checkpoint is None:
        normalizer = dataset.get_normalizer(spec)
    else:
        normalizer = LinearNormalizer(normalization_feature_shapes({key: dims[key] for key, mode in spec.items() if mode != "identity"}))
        normalizer.load_state_dict({
            key.removeprefix("normalizer."): value
            for key, value in resume_checkpoint.model_state.items() if key.startswith("normalizer.")
        }, strict=True)
        dataset._normalizer = normalizer
    if verbose:
        print(f"[Dataset] {dataset.training_summary()}")
        if resume_checkpoint is None:
            print("[Normalizer] all real train-episode frames; relative actions retain each window/reference")
    return dataset, normalizer, description


def _describe_real_runtime(buffer, sensor_modalities, task_name) -> dict | None:
    """Capture numerical deployment inputs from the buffer actually used to train."""
    attrs = buffer.attrs
    if attrs.get("format") != "dexmani.real.canonical":
        return None
    if attrs.get("task_name") != task_name:
        raise ValueError("Real canonical task_name differs from the training task")
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not isfinite(dt) or dt <= 0:
        raise ValueError("Real canonical dt must be finite and positive")
    runtime: dict[str, Any] = {"dt": float(dt)}
    if "point_cloud" in sensor_modalities:
        cloud = attrs.get("pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real point_cloud requires root pointcloud_config")
        count = positive_int(cloud["num_points"], "pointcloud_config.num_points")
        shape = buffer["point_cloud"].shape
        if shape[1] != count:
            raise ValueError("Stored point count disagrees with pointcloud_config")
        # Preserve the stored point count; encoder FPS selects its own subset.
        runtime["pointcloud"] = dict(cloud)
    if "hand_pointcloud" in sensor_modalities:
        cloud = attrs.get("hand_pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real hand_pointcloud requires root hand_pointcloud_config")
        count = positive_int(cloud["num_points"], "hand_pointcloud_config.num_points")
        data = buffer["hand_pointcloud"]
        if data.shape[1:] != (count, 6) or str(data.dtype) != "float32":
            raise ValueError("Stored hand_pointcloud must match the saved XYZRGB template")
        runtime["hand_pointcloud"] = dict(cloud)
    if "depth" in sensor_modalities:
        data = buffer["depth"]
        scale = attrs["depth_scale_m_per_unit"]
        if len(data.shape) != 3 or any(size <= 0 for size in data.shape[1:]) or str(data.dtype) != "uint16":
            raise ValueError("Stored depth must have shape [T,H,W] and dtype uint16")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not isfinite(scale) or scale <= 0:
            raise ValueError("Depth requires finite positive depth_scale_m_per_unit")
        runtime["depth"] = {
            "shape": list(data.shape[1:]), "dtype": "uint16",
            "unit": "raw_counts", "depth_scale_m": scale,
        }
    if "fingertip_xyz" in sensor_modalities:
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


def close_data_loader(loader, iterator=None):
    # Persistent workers survive early exits until their iterator is shut down.
    iterator = loader._iterator if iterator is None else iterator
    loader._iterator = None
    if iterator is not None and loader.num_workers > 0:
        iterator._shutdown_workers()


class EpochBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last, *, shuffle, seed):
        self.generator = torch.Generator()
        sampler = RandomSampler(dataset, generator=self.generator) if shuffle else SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)
        self.seed = seed
        self.set_epoch(0)

    @property
    def batches_per_epoch(self):
        return super().__len__()

    def set_epoch(self, epoch, start_batch=0):
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return islice(super().__iter__(), self.start_batch, None)

    def __len__(self):
        return self.batches_per_epoch - self.start_batch


def epoch_position(global_step, batches_per_epoch, accumulation):
    updates = (batches_per_epoch + accumulation - 1) // accumulation
    epoch, update = divmod(global_step, updates)
    return epoch, update * accumulation


def build_train_loader(cfg, dataset):
    options = dict(cfg.dataloader)
    if options.get("num_workers", 0) == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    samples = len(dataset)
    batch_sampler = EpochBatchSampler(
        dataset, options.pop("batch_size"), options.pop("drop_last", False),
        shuffle=options.pop("shuffle", False), seed=cfg.training.seed,
    )
    batch_size = batch_sampler.batch_size
    accumulation = cfg.training.loop.gradient_accumulation_steps
    batches = batch_sampler.batches_per_epoch
    if (
        accumulation > 1
        and not batch_sampler.drop_last
        and samples % batch_size
        and batches % accumulation != 1
    ):
        raise ValueError(
            f"Mixed micro-batch sizes in accumulation group: "
            f"N={samples}, B={batch_size}, A={accumulation}; "
            "use drop_last=True or gradient_accumulation_steps=1"
        )
    # Worker creation must not consume the sample permutation's RNG.
    generator = torch.Generator().manual_seed(cfg.training.seed)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, generator=generator, **options)
    if len(loader) == 0:
        raise ValueError(
            "Training DataLoader is empty; check dataset length, batch_size and drop_last"
        )
    return loader
