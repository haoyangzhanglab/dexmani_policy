import numpy as np
import pytest
import torch
import zarr


@pytest.fixture
def demonstrations(tmp_path):
    path = tmp_path / "demo.zarr"
    root = zarr.open_group(str(path), mode="w")
    rng = np.random.default_rng(3)
    arrays = {
        "action": rng.normal(0, 0.1, (48, 19)).astype(np.float32),
        "joint_state": rng.normal(0, 0.1, (48, 19)).astype(np.float32),
        "point_cloud": rng.random((48, 8, 6), dtype=np.float32),
        "rgb": rng.integers(0, 256, (48, 8, 8, 3), dtype=np.uint8),
    }
    for key, values in arrays.items():
        root.create_dataset(f"data/{key}", data=values, chunks=(4, *values.shape[1:]))
    root.create_dataset("meta/episode_ends", data=np.array([12, 24, 36, 48]))
    root.attrs.update(dt=0.04, action_space="joint", obs_alignment="obs[t]_before_action[t]")
    return path


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
