import shutil

import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf

from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.training.data import record_training_data, restore_data_records


@pytest.mark.parametrize("changed", ["episode_ends", "episode_groups", "episode_seeds", "source_row_ranges"])
def test_pretraining_checks_episode_layout(tmp_path, make_demo, changed):
    source = make_demo(episode_ends=(4, 8, 12))
    raw = zarr.open_group(str(source), mode="a")
    if changed in ("episode_groups", "source_row_ranges"):
        raw.attrs.update(
            format="dexmani.real.canonical", source_episodes=["a", "b", "c"],
            source_row_ranges=[[0, 4], [0, 4], [0, 4]],
        )
    else:
        raw.create_dataset("meta/episode_seeds", data=[32, 33, 34])
    derived = tmp_path / "derived.zarr"
    shutil.copytree(source, derived)
    pretraining = zarr.open_group(str(derived), mode="a")
    pretraining.attrs["source_zarr_path"] = str(source)

    train = np.ones(3, dtype=bool)

    def record_source():
        cfg = OmegaConf.create({"policy_zarr_path": str(source), "task_name": "test", "dataset": {}})
        record_training_data(cfg, EpisodeStore(derived, keys=()), train)
        return cfg

    recorded = record_source()
    saved = OmegaConf.to_container(recorded, resolve=True)
    if changed == "episode_ends":
        pretraining["meta/episode_ends"][:] = [3, 8, 12]
    elif changed == "episode_groups":
        pretraining.attrs["source_episodes"] = ["b", "a", "c"]
    elif changed == "episode_seeds":
        pretraining["meta/episode_seeds"][:] = [33, 32, 34]
    else:
        pretraining.attrs["source_row_ranges"] = [[1, 5], [0, 4], [0, 4]]
    with pytest.raises(ValueError, match="share the policy episode layout"):
        record_source()
    with pytest.raises(ValueError, match="Resume episode layout mismatch"):
        restore_data_records(recorded, EpisodeStore(derived, keys=()), train, saved=saved)
