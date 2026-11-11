from math import prod

import hydra
import torch
from torch.utils.data import DataLoader

from dexmani_policy.common.config import (
    FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, TACTILE_FEATURE_SHAPES,
)
from dexmani_policy.common.normalizer import (
    FieldNormalizer, LinearNormalizer, validate_normalizer_state,
)


def fit_normalizer(
    dataset, normalizer: LinearNormalizer, fields, *, initial_state=None, inherited_fields=(), hand_affine=None,
) -> None:
    """Fit limits fields, loading inherited affine into the existing modules."""
    if initial_state is not None:
        inherited = LinearNormalizer()
        for key in inherited_fields:
            inherited.fields[key] = normalizer[key]
        inherited.load_state_dict(initial_state, strict=True)
        validate_normalizer_state(inherited)
    for key in tuple(fields):
        if key in inherited_fields:
            continue
        expected_shape = normalizer[key].scale.shape
        chunks = dataset.iter_normalization_data(key)
        if key == "action" and hand_affine is not None:
            arm_dim = normalizer[key].scale.numel() - hand_affine["scale"].numel()
            normalizer.fit_field(key, (values[..., :arm_dim] for values in chunks))
            field = normalizer[key]
            normalizer.fields[key] = FieldNormalizer(
                torch.cat((field.scale, hand_affine["scale"])),
                torch.cat((field.offset, hand_affine["offset"])),
            )
        else:
            normalizer.fit_field(key, chunks)
        if normalizer[key].scale.shape != expected_shape:
            raise ValueError(f"Fitted affine shape mismatch for {key}: expected {tuple(expected_shape)}")
        if key == "action" and dataset.action_key == "action_ee":
            normalizer[key].scale[3:9] = 1
            normalizer[key].offset[3:9] = 0


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
    print(f"[Dataset] {dataset.training_summary()}")
    deployment = {}
    if "dt" in buffer.attrs:
        deployment["control_dt_s"] = buffer.attrs["dt"]
    for modality, attr in (
        ("point_cloud", "pointcloud_config"),
        ("fingertip_xyz", "fingertip_link_names"),
    ):
        if modality in dataset.sensor_modalities and attr in buffer.attrs:
            deployment[attr] = buffer.attrs[attr]
    if "depth" in dataset.sensor_modalities:
        deployment["depth_config"] = {
            "shape": list(buffer["depth"].shape[1:]),
            "depth_scale_m": buffer.attrs.get("depth_scale_m_per_unit"),
        }
    return dataset, {"field_dims": dims, "deployment_metadata": deployment}


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
