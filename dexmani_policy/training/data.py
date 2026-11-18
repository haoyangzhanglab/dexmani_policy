from math import isfinite, prod
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import (
    FIXED_OBSERVATION_SHAPES,
    POINT_CLOUD_FIELDS,
)
from dexmani_policy.common.normalizer import FieldNormalizer, LinearNormalizer
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite
from dexmani_policy.datasets.split import split_episodes


def use_policy_split(cfg, policy_config, *, zarr_path=None):
    policy = OmegaConf.load(Path(policy_config).expanduser())
    source = str(Path(policy.zarr_path).expanduser().resolve())
    cfg.task_name = policy.task_name
    cfg.zarr_path = source if zarr_path is None else zarr_path
    cfg.dataset.zarr_path = cfg.zarr_path
    for key in ("seed", "val_episodes", "max_train_episodes"):
        cfg.dataset[key] = policy.dataset[key]
    with open_dict(cfg):
        cfg.policy_zarr_path = source


def record_training_data(cfg, store, train, val=None):
    train_groups = store.source_groups(train)
    val_groups = store.source_groups(val) if val is not None else set()
    record = {"source": store.source_path, "train_source_episodes": sorted(train_groups)}
    if val_groups:
        record["val_source_episodes"] = sorted(val_groups)
    reference_path = cfg.get("policy_zarr_path")
    if reference_path is not None:
        if store.attrs.get("task_name") != cfg.task_name:
            raise ValueError(f"{store.path}: pretraining task must match {cfg.task_name}")
        if store.source_path != reference_path:
            raise ValueError(f"{store.path}: pretraining data must share the original policy Zarr")
        if reference_path != store.path:
            reference = EpisodeStore(reference_path, keys=())
            expected_train, expected_val = split_episodes(
                reference.episode_groups, seed=cfg.dataset.seed,
                max_train_episodes=cfg.dataset.max_train_episodes, val_episodes=cfg.dataset.val_episodes,
            )
            if (train_groups != reference.source_groups(expected_train)
                    or val_groups != reference.source_groups(expected_val)):
                raise ValueError(f"{store.path}: pretraining data must share the policy source and episode split")
    with open_dict(cfg):
        cfg.pop("policy_zarr_path", None)
        cfg.pop("prior_data", None)
        cfg.zarr_path = cfg.dataset.zarr_path = store.path
        cfg.training_data = record


def restore_data_records(cfg, store, train, val=None, *, saved: dict):
    saved_path = saved["dataset"]["zarr_path"]
    if store.path != saved_path:
        raise ValueError(f"Resume Zarr path mismatch (dataset={store.path!r}, checkpoint={saved_path!r})")
    record = saved["training_data"]
    if store.source_path != record["source"]:
        raise ValueError(
            f"Resume data source mismatch (dataset={store.source_path!r}, checkpoint={record['source']!r})"
        )
    splits = [("train", train, record["train_source_episodes"])]
    if val is not None:
        splits.append(("validation", val, record.get("val_source_episodes", [])))
    for split, mask, expected in splits:
        groups = store.source_groups(mask)
        if groups != set(expected):
            raise ValueError(
                f"Resume {split} source groups mismatch (dataset={sorted(groups)}, checkpoint={expected})"
            )
    with open_dict(cfg):
        cfg.zarr_path = cfg.dataset.zarr_path = saved_path
        cfg.training_data = record
        if "prior_data" in saved:
            cfg.prior_data = saved["prior_data"]
        else:
            cfg.pop("prior_data", None)


def record_prior_data(cfg, dataset, asset, record: dict | None = None, *, from_policy=False):
    store = dataset.episode_store
    if record is None:
        record = {"source": None}
    source = record["source"]
    same_source = source is not None and source == store.source_path
    if same_source:
        excess_train = set(record["train_source_episodes"]) - store.source_groups(dataset.train_mask)
        excess_val = set(record.get("val_source_episodes", ())) - store.source_groups(dataset.val_mask)
        if excess_train or excess_val:
            raise ValueError(
                f"{asset}: same-source prior exceeds the policy episode split: "
                f"train={sorted(excess_train)}, validation={sorted(excess_val)}"
            )
    budget_verified = same_source and not from_policy
    if not budget_verified:
        print(f"[Dataset] Prior {asset} has an unverified training budget")
    with open_dict(cfg):
        if "prior_data" not in cfg:
            cfg.prior_data = []
        cfg.prior_data.append({**record, "artifact": str(asset), "budget_verified": budget_verified})


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


def fit_action_normalizer(dataset, normalizer: LinearNormalizer, *, hand_affine=None) -> None:
    if hand_affine is None:
        normalizer.fit_field("action", dataset.iter_normalization_data("action"))
    else:
        # The hand representation must retain the coordinate system of its prior.
        arm_dim = normalizer["action"].scale.numel() - hand_affine["scale"].numel()
        normalizer.fit_field(
            "action", (values[..., :arm_dim] for values in iter_validated_actions(dataset)),
            check_values=False,
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


def record_data_processing(cfg, dataset):
    """记录实际读取字段的处理参数。"""
    processing = dataset.episode_store.attrs.get("data_processing")
    metadata = None if processing is None else {
        key: processing[key] for key in dataset.episode_store.keys() if key in processing
    }
    with open_dict(cfg):
        cfg.data_processing = metadata or None
