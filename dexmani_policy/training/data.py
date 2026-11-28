from math import isfinite, prod

import hydra
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import (
    FIXED_OBSERVATION_SHAPES,
    POINT_CLOUD_FIELDS,
)
from dexmani_policy.common.normalizer import FieldNormalizer, LinearNormalizer
from dexmani_policy.common.pytorch_util import rebase_eef_rotvec
from dexmani_policy.datasets.episode_store import check_finite
from dexmani_policy.env_runner.dexjoco_runner import observation_processing


def _episode_layout(store):
    layout = {"episode_ends": store.episode_ends.tolist(), "episode_groups": store.episode_groups}
    seeds = store.meta.get("episode_seeds")
    if seeds is not None:
        layout["episode_seeds"] = seeds[...].tolist()
    ranges = store.attrs.get("source_row_ranges")
    if ranges is not None:
        layout["source_row_ranges"] = ranges
    return layout


def record_training_data(cfg, store, train, val):
    layout = _episode_layout(store)
    train_groups = store.source_groups(train)
    val_groups = store.source_groups(val)
    record = {"train_source_episodes": sorted(train_groups), "val_source_episodes": sorted(val_groups)}
    with open_dict(cfg):
        cfg.zarr_path = cfg.dataset.zarr_path = store.path
        cfg.training_data = record
        cfg.episode_layout = layout


def validate_resume_data(store, train, val, *, saved: dict):
    saved_path = saved["dataset"]["zarr_path"]
    if store.path != saved_path:
        raise ValueError(f"Resume Zarr path mismatch (dataset={store.path!r}, checkpoint={saved_path!r})")
    record = saved["training_data"]
    for split, mask in (("train", train), ("val", val)):
        expected = record[f"{split}_source_episodes"]
        groups = store.source_groups(mask)
        if groups != set(expected):
            raise ValueError(
                f"Resume {split} source groups mismatch (dataset={sorted(groups)}, checkpoint={expected})"
            )
    layout = saved["episode_layout"]
    if _episode_layout(store) != layout:
        raise ValueError(f"Resume episode layout mismatch for {store.path}")


def fit_observation_normalizer(
    dataset, normalizer: LinearNormalizer, modes, *, initial_state=None, inherited_fields=(),
) -> None:
    """Fit observation scales, retaining affine inherited from encoder pretraining."""
    if initial_state is not None:
        inherited = LinearNormalizer()
        for key in inherited_fields:
            inherited.fields[key] = normalizer[key]
        inherited.load_state_dict(initial_state, strict=True)
        inherited.validate()
    for key in dataset.sensor_modalities:
        if key not in normalizer.fields or key in inherited_fields:
            continue
        normalizer.fit_field(key, dataset.iter_normalization_data(key), mode=modes[key])


def iter_validated_actions(dataset):
    store, key = dataset.episode_store, dataset.action_key
    for start, values in store.iter_chunks(key, episode_mask=dataset.train_mask):
        check_finite(store, key, start, values)
        yield values


def fit_action_normalizer(
    dataset, normalizer: LinearNormalizer, *, hand_affine=None, center_rotvec=False,
) -> None:
    if hand_affine is None:
        chunks = dataset.iter_normalization_data("action")
        if center_rotvec:
            chunks = (rebase_eef_rotvec(torch.as_tensor(values)) for values in chunks)
        normalizer.fit_field("action", chunks)
    else:
        # The hand representation must retain the coordinate system of its prior.
        arm_dim = normalizer["action"].scale.numel() - hand_affine["scale"].numel()
        normalizer.fit_field(
            "action", (values[..., :arm_dim] for values in iter_validated_actions(dataset)),
        )
        arm = normalizer["action"]
        normalizer.fields["action"] = FieldNormalizer(
            torch.cat((arm.scale, hand_affine["scale"])),
            torch.cat((arm.offset, hand_affine["offset"])),
        )
    field = normalizer["action"]
    if dataset.action_key == "action_ee":
        field.scale[3:9] = 1
        field.offset[3:9] = 0


def build_dataset(cfg):
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    buffer = dataset.episode_store
    if buffer.attrs["task_name"] != cfg.task_name:
        raise ValueError(
            f"Dataset task mismatch: dataset={buffer.attrs['task_name']}, config={cfg.task_name}"
        )
    action_space = "eef_rotvec" if dataset.action_key == "action_rotvec" else "joint"
    if buffer.attrs["action_space"] != action_space:
        raise ValueError(f"Dataset action_space must be {action_space} for {dataset.action_key}")
    dt = float(buffer.attrs["dt"])
    if not isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dataset dt: {dt}")
    dims = {}
    for key in dataset.sensor_modalities:
        if key in FIXED_OBSERVATION_SHAPES:
            dims[key] = prod(FIXED_OBSERVATION_SHAPES[key])
        elif key in POINT_CLOUD_FIELDS and dataset.pc_dim is not None:
            dims[key] = dataset.pc_dim
        else:
            dims[key] = buffer[key].shape[-1]
    dims["action"] = dataset.action_dim
    if cfg.agent.action_dim is None:
        cfg.agent.action_dim = dims["action"]
    if "state_key" in cfg.agent and cfg.agent.state_dim is None:
        cfg.agent.state_dim = dims[cfg.agent.state_key]
    if dims["action"] != cfg.agent.action_dim:
        raise ValueError("Dataset action width including auxiliary targets must match agent.action_dim")
    print(f"[Dataset] {dataset.training_summary()}")
    return dataset, {"field_dims": dims, "control_dt_s": dt}


def close_data_loader(loader, iterator=None):
    # Persistent workers survive early exits until their iterator is shut down.
    iterator = loader._iterator if iterator is None else iterator
    loader._iterator = None
    if iterator is not None and loader.num_workers > 0:
        iterator._shutdown_workers()


def build_train_loader(cfg, dataset):
    options = dict(cfg.dataloader)
    if options["num_workers"] == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    loader = DataLoader(dataset, **options)
    if len(loader) == 0:
        raise ValueError(
            "Training DataLoader is empty; check dataset length, batch_size and drop_last"
        )
    return loader


def record_data_processing(cfg, dataset, *, dexjoco=False):
    """记录实际读取字段的处理参数。"""
    processing = dataset.episode_store.attrs.get("data_processing")
    if dexjoco:
        metadata = observation_processing(processing, dataset.sensor_modalities)
    else:
        metadata = {key: processing[key] for key in dataset.episode_store.keys() if key in (processing or {})}
    with open_dict(cfg):
        cfg.data_processing = metadata or None
