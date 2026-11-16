"""实际 Dataset 的处理记录与预训练恢复契约。"""

from copy import deepcopy

import hydra
import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf, open_dict

from dexmani_policy.training.common.config import load_config
from dexmani_policy.training.common.data import build_dataset, record_data_processing


@pytest.fixture
def processing_source(tmp_path):
    path = tmp_path / "source.zarr"
    source = zarr.open_group(str(path), mode="w-")
    source.attrs.update(
        format="dexmani.real.canonical", task_name="pick_apple_messy",
        dt=0.05, action_space="joint",
        data_processing={
            "point_cloud": {"num_points": 8},
            "hand_pointcloud": {"num_points": 8},
            "object_pointcloud": {"template_sha256": "original"},
            "joint_state": {"joint_units": "rad"},
            "rgb": {"image_space": "pinhole"},
        },
    )
    source.create_dataset("meta/episode_ends", data=np.array([40], dtype=np.int64))
    for key, shape in (
        ("point_cloud", (40, 8, 6)), ("hand_pointcloud", (40, 8, 3)),
        ("object_pointcloud", (40, 1024, 3)), ("joint_state", (40, 19)),
        ("contact_map", (40, 1024)), ("action", (40, 19)),
    ):
        source.create_dataset(f"data/{key}", data=np.zeros(shape, dtype=np.float32))
    return path


@pytest.mark.parametrize("recipe", ["pretrain/obs/afro", "pretrain/obs/cordvip", "dp3"])
def test_processing_roundtrip(tmp_path, processing_source, recipe):
    cfg = load_config(recipe, [f"zarr_path={processing_source}"])
    cfg.workspace.output_dir = str(tmp_path / "run")
    if recipe == "dp3":
        dataset, _ = build_dataset(cfg)
    else:
        assert "sensor_modalities" not in cfg.dataset
        dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    record_data_processing(cfg, dataset)
    metadata = dataset.episode_store.attrs["data_processing"]
    assert OmegaConf.to_container(cfg.data_processing) == metadata
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    restored = OmegaConf.load(path)
    dataset = hydra.utils.instantiate(restored.dataset, _convert_="all")
    record_data_processing(restored, dataset, resuming=True)
    original = deepcopy(dataset.episode_store.attrs["data_processing"])
    for name in dataset.sensor_modalities:
        changed = deepcopy(original)
        key = next(iter(changed[name]))
        changed[name][key] = "changed"
        dataset.episode_store.attrs["data_processing"] = changed
        before = OmegaConf.to_container(restored)
        with pytest.raises(ValueError, match=f"mismatch: {name}"):
            record_data_processing(restored, dataset, resuming=True)
        assert OmegaConf.to_container(restored) == before
    changed = deepcopy(original)
    changed["rgb"]["image_space"] = "changed"
    dataset.episode_store.attrs["data_processing"] = changed
    record_data_processing(restored, dataset, resuming=True)
    assert OmegaConf.to_container(restored.data_processing) == changed


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("resuming", [False, True])
def test_processing_missing_record(processing_source, real, resuming):
    source = zarr.open_group(str(processing_source), mode="a")
    del source.attrs["data_processing"]
    if not real:
        del source.attrs["format"]
    cfg = load_config("pretrain/obs/afro", [f"zarr_path={processing_source}"])
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    if resuming:
        with open_dict(cfg):
            cfg.data_processing = {}
    if real:
        with pytest.raises(KeyError, match="data_processing"):
            record_data_processing(cfg, dataset, resuming=resuming)
    else:
        record_data_processing(cfg, dataset, resuming=resuming)
        assert OmegaConf.to_container(cfg.data_processing) == {}
