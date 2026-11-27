from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state
from dexmani_policy.training.trainer import build_trainer

from tests.helpers import tiny_policy_config


@pytest.fixture(scope="session", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def make_demo(tmp_path):
    def create(name="demo", *, auxiliary=True, action_dim=19, episode_ends=(12,)):
        path = tmp_path / f"{name}.zarr"
        store = zarr.open_group(str(path), mode="w")
        store.attrs.update(task_name="test", action_space="joint", dt=0.05)
        rng = np.random.default_rng(1)
        n_rows = episode_ends[-1]
        fields = {
            "action": rng.normal(size=(n_rows, action_dim)).astype(np.float32),
            "joint_state": rng.normal(size=(n_rows, 19)).astype(np.float32),
            "point_cloud": rng.normal(size=(n_rows, 8, 6)).astype(np.float32),
        }
        if auxiliary:
            fields["action_ee"] = rng.normal(size=(n_rows, 21)).astype(np.float32)
        for key, value in fields.items():
            store.create_dataset(f"data/{key}", data=value, chunks=(4, *value.shape[1:]))
        store.create_dataset("meta/episode_ends", data=np.array(episode_ends, dtype=np.int64))
        return path
    return create


@pytest.fixture
def small_policy_config(make_demo):
    path = make_demo()

    def create(output, *overrides):
        return tiny_policy_config(path, output, *overrides)
    return create


@pytest.fixture(autouse=True)
def preserve_process_state():
    rng = get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    precision = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        torch.set_float32_matmul_precision(precision)
        set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@pytest.fixture
def small_policy_checkpoint(tmp_path, small_policy_config):
    directory = tmp_path / "policy_run"
    with closing(build_trainer(small_policy_config(directory))) as trainer:
        trainer.train(max_updates=1)
    return directory / "checkpoints/latest.pt"


@pytest.fixture
def segments(tmp_path):
    path = tmp_path / "segments.zarr"
    group = zarr.open_group(str(path), mode="w")
    rows = np.arange(12, dtype=np.float32)
    fields = {
        "action": np.broadcast_to(rows[:, None] + 100, (12, 19)).copy(),
        "joint_state": np.broadcast_to(rows[:, None] + 200, (12, 19)).copy(),
        "point_cloud": np.arange(12 * 2 * 6, dtype=np.float64).reshape(12, 2, 6),
        "finger_tactile_force": np.broadcast_to(rows[:, None, None], (12, 5, 3)).copy(),
        "rgb": np.arange(12 * 2 * 3 * 3, dtype=np.uint8).reshape(12, 2, 3, 3),
    }
    for key, value in fields.items():
        group.create_dataset(f"data/{key}", data=value, chunks=(2, *value.shape[1:]))
    group.create_dataset("meta/episode_ends", data=np.array([1, 3, 8, 12]))
    return path, fields


@pytest.fixture(params=["~/run", "relative/run", "absolute"])
def output_path(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    original = Path.expanduser

    def expand(path):
        if path.parts and path.parts[0] == "~":
            return tmp_path / "mapped_home" / Path(*path.parts[1:])
        return original(path)

    monkeypatch.setattr(Path, "expanduser", expand)
    value = str(tmp_path / "absolute") if request.param == "absolute" else request.param
    yield value, Path(value).expanduser().resolve()
    assert not (tmp_path / "~").exists()
