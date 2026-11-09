import re
from itertools import islice
from math import prod

import hydra
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from dexmani_policy.common.config import (
    FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, TACTILE_FEATURE_SHAPES, validate_normalization_spec,
)
from dexmani_policy.common.normalizer import (
    FieldNormalizer, LinearNormalizer, validate_normalizer_state,
)


def fit_normalizer(
    dataset, spec, normalizer: LinearNormalizer, *, initial_state=None, inherited_fields=(), hand_affine=None,
) -> None:
    """Fit the configured affine fields in place, preserving inherited and internal fields."""
    validate_normalization_spec(spec, spec)
    for key in inherited_fields:
        if key not in normalizer.fields or spec.get(key) != "limits":
            raise RuntimeError(f"Missing affine field for inherited limits feature: {key}")
    inherited = LinearNormalizer({
        key: normalizer[key].scale.shape for key in inherited_fields
    })
    inherited.load_state_dict({} if initial_state is None else initial_state, strict=True)
    normalizer.fields.update(dict(inherited.fields.items()))
    for key, mode in spec.items():
        if mode == "identity":
            continue
        if key not in normalizer.fields:
            raise RuntimeError(f"Missing affine field for limits feature: {key}")
        if key in inherited_fields:
            continue
        expected_shape = normalizer[key].scale.shape
        chunks = dataset.iter_normalization_data(key)
        if key == "action" and hand_affine is not None:
            arm_dim = normalizer[key].scale.numel() - hand_affine["scale"].numel()
            chunks = (values[..., :arm_dim] for values in chunks)
        normalizer.fit_field(key, chunks)
        if key == "action" and dataset.action_key == "action_ee":
            field = normalizer[key]
            if field.scale.numel() <= 9:
                raise ValueError("action_ee requires xyz, rot6d and hand coordinates")
            field.scale[3:9] = 1
            field.offset[3:9] = 0
        if key == "action" and hand_affine is not None:
            field = normalizer[key]
            normalizer.fields[key] = FieldNormalizer(
                torch.cat((field.scale[:arm_dim], hand_affine["scale"])),
                torch.cat((field.offset[:arm_dim], hand_affine["offset"])),
            )
        if normalizer[key].scale.shape != expected_shape:
            raise ValueError(f"Fitted affine shape mismatch for {key}: expected {tuple(expected_shape)}")
    validate_normalizer_state(normalizer)


def build_dataset(cfg):
    """Construct training samples and describe their stored field dimensions."""
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    buffer = dataset.episode_store
    dims = {
        key: int(buffer[key].shape[-1]) for key in dataset.sensor_modalities
    }
    for key in (FIELD_FEATURE_SHAPES.keys() | TACTILE_FEATURE_SHAPES.keys()) & dims.keys():
        dims[key] = prod(buffer[key].shape[1:])
    if dataset.pc_dim is not None:
        for key in POINT_CLOUD_FIELDS.intersection(dims):
            dims[key] = dataset.pc_dim
    dims["action"] = dataset.action_dim
    if dims["action"] != cfg.agent.action_dim:
        raise ValueError("Dataset action width including auxiliary targets must match agent.action_dim")
    description = {
        "field_dims": dims,
        "action_key": dataset.action_key,
        "hand_coordinates": {key: buffer.attrs[key] for key in ("hand_joint_order", "hand_units")
                             if key in buffer.attrs},
        "control_action_dim": buffer[dataset.action_key].shape[-1],
    }
    deployment = {}
    if "dt" in buffer.attrs:
        deployment["control_dt_s"] = buffer.attrs["dt"]
    for modality, attr in (
        ("point_cloud", "pointcloud_config"),
        ("hand_pointcloud", "hand_pointcloud_config"),
        ("fingertip_xyz", "fingertip_link_names"),
    ):
        if modality in dataset.sensor_modalities and attr in buffer.attrs:
            deployment[attr] = buffer.attrs[attr]
    if "object_pointcloud" in dataset.sensor_modalities:
        metadata = buffer.attrs.get("hoi_pcd")
        if not isinstance(metadata, dict):
            raise ValueError("HOI dataset requires hoi_pcd metadata mapping")
        task_name = metadata.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip() or task_name != cfg.task_name:
            raise ValueError("HOI dataset task_name must be nonempty and match training task")
        names = metadata.get("object_names")
        if (
            not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("HOI dataset object_names must be a nonempty list of unique names")
        digest = metadata.get("template_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("HOI dataset template_sha256 must be a 64-digit SHA256 hex string")
        points = buffer["object_pointcloud"]
        if points.shape[1:] != (1024, 3) or points.dtype.name != "float32":
            raise ValueError("HOI dataset requires float32 [T,1024,3]")
        deployment["hoi_pcd"] = {
            "task_name": task_name,
            "object_names": names,
            "template_sha256": digest.lower(),
        }
    if "depth" in dataset.sensor_modalities:
        deployment["depth_config"] = {
            "shape": list(buffer["depth"].shape[1:]),
            "depth_scale_m": buffer.attrs.get("depth_scale_m_per_unit"),
        }
    description["deployment_metadata"] = deployment
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
