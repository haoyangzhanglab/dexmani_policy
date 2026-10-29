from itertools import islice
from math import prod

import hydra
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS


def build_dataset(cfg, *, verbose=True):
    """Admit one train/val source before choosing the model's affine initialization."""
    spec = cfg.normalization
    dataset = hydra.utils.instantiate(cfg.dataset)
    buffer = dataset.replay_buffer
    # RobotDataset 已核对 split 与数据身份一致；resume 额外拒绝双方都没有 ID。
    if (
        cfg.get("resume_from") is not None
        and buffer.attrs.get("format") == "dexmani.real.canonical"
        and buffer.attrs.get("source_id") is None
    ):
        raise ValueError(
            "Canonical checkpoint resume requires source_id in both the dataset and saved split; "
            "re-export legacy data and start a new experiment for reliable resume"
        )
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
    deployment = {}
    if buffer.attrs.get("format") == "dexmani.real.canonical":
        deployment["control_dt_s"] = buffer.attrs.get("dt")
        for modality, attr in (
            ("point_cloud", "pointcloud_config"),
            ("hand_pointcloud", "hand_pointcloud_config"),
            ("fingertip_xyz", "fingertip_link_names"),
        ):
            if modality in dataset.sensor_modalities:
                deployment[attr] = buffer.attrs.get(attr)
        if "depth" in dataset.sensor_modalities:
            deployment["depth_config"] = {
                "shape": list(buffer["depth"].shape[1:]),
                "depth_scale_m": buffer.attrs.get("depth_scale_m_per_unit"),
            }
        if "processing_source" in buffer.attrs:
            deployment["processing_source"] = buffer.attrs["processing_source"]
    description["deployment_metadata"] = deployment
    if hasattr(dataset, "hand_data"):
        description["hand_data"] = dataset.hand_data
    if verbose:
        print(f"[Dataset] {dataset.training_summary()}")
    return dataset, description


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
