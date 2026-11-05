from itertools import islice
from math import prod

import hydra
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from dexmani_policy.common.config import (
    FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, TACTILE_FEATURE_SHAPES,
)
from dexmani_policy.common.normalizer import (
    FieldNormalizer, LinearNormalizer, normalization_feature_shapes, validate_normalizer_state,
)


def fit_normalizer(dataset, spec, *, feature_dims, initial_state=None, hand_affine=None):
    """Assemble the train affine once, inheriting complete fields before fitting the rest."""
    spec = {key: spec[key] for key in feature_dims}
    expected_shapes = normalization_feature_shapes(feature_dims)
    initial_state = {} if initial_state is None else initial_state
    normalizer = LinearNormalizer({
        key: expected_shapes[key] for key in spec
        if any(f"fields.{key}.{part}" in initial_state for part in ("scale", "offset"))
    })
    normalizer.load_state_dict(initial_state, strict=True)
    validate_normalizer_state(normalizer)
    for key, mode in spec.items():
        if key in normalizer.fields:
            continue
        if mode == "identity":
            if key == "action" or key in TACTILE_FEATURE_SHAPES:
                # Action and tactile loss paths read affine buffers directly.
                shape = expected_shapes[key]
                normalizer.fields[key] = FieldNormalizer(torch.ones(shape), torch.zeros(shape))
            else:
                continue
        else:
            chunks = dataset.iter_normalization_data(key)
            if key == "action" and hand_affine is not None and mode in {"limits", "gaussian"}:
                arm_dim = feature_dims[key] - hand_affine["scale"].numel()
                chunks = (values[..., :arm_dim] for values in chunks)
            tactile_limits = key in TACTILE_FEATURE_SHAPES and mode == "limits"
            normalizer.fit_field(key, chunks, mode=mode,
                                 last_n_dims=1 if tactile_limits else len(expected_shapes[key]))
            if tactile_limits:
                field = normalizer[key]
                field.scale = field.scale.expand(expected_shapes[key]).clone()
                field.offset = field.offset.expand(expected_shapes[key]).clone()
        if key == "action" and mode != "identity" and dataset.action_key == "action_ee":
            field = normalizer[key]
            if field.scale.numel() <= 9:
                raise ValueError("action_ee requires xyz, rot6d and hand coordinates")
            field.scale[3:9] = 1
            field.offset[3:9] = 0
        if key == "action" and hand_affine is not None:
            field = normalizer[key]
            arm_dim = feature_dims[key] - hand_affine["scale"].numel()
            # RMS/MaxAbs must include hand values when determining the shared arm scale.
            normalizer.fields[key] = FieldNormalizer(
                torch.cat((field.scale[:arm_dim], hand_affine["scale"])),
                torch.cat((field.offset[:arm_dim], hand_affine["offset"])),
            )
    return normalizer


def build_dataset(cfg):
    """Construct training samples and describe their stored field dimensions."""
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    buffer = dataset.episode_store
    dims = {
        key: int(buffer[key].shape[-1]) for key in dataset.sensor_modalities
    }
    for key in FIELD_FEATURE_SHAPES.keys() & dims.keys():
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
