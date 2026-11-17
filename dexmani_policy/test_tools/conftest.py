from contextlib import closing

import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state
from dexmani_policy.training.config import load_config


@pytest.fixture(scope="session", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def make_demo(tmp_path):
    def create(name="demo", *, auxiliary=True, action_dim=19):
        path = tmp_path / f"{name}.zarr"
        store = zarr.open_group(str(path), mode="w")
        store.attrs.update(task_name="test", action_space="joint", dt=0.05)
        rng = np.random.default_rng(1)
        fields = {
            "action": rng.normal(size=(12, action_dim)).astype(np.float32),
            "joint_state": rng.normal(size=(12, 19)).astype(np.float32),
            "point_cloud": rng.normal(size=(12, 8, 6)).astype(np.float32),
        }
        if auxiliary:
            fields["action_ee"] = rng.normal(size=(12, 21)).astype(np.float32)
        for key, value in fields.items():
            store.create_dataset(f"data/{key}", data=value, chunks=(4, *value.shape[1:]))
        store.create_dataset("meta/episode_ends", data=np.array([12], dtype=np.int64))
        return path
    return create


@pytest.fixture
def small_dp3_config(make_demo):
    path = make_demo()

    def create(output, *overrides):
        return load_config("dp3", [
            f"zarr_path={path}", "task_name=test", f"workspace.output_dir='{output}'",
            "workspace.wandb_cfg=null", "training.device=cpu", "training.use_compile=false",
            "training.loop.total_train_steps=3", "training.lr_warmup_steps=0",
            "training.loop.log_interval_steps=1", "dataloader.num_workers=0",
            "dataloader.batch_size=2", "dataloader.drop_last=false",
            "dataset.augmentation_cfg=null", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
            "agent.down_dims=[16,32]", "agent.num_points=8", *overrides,
        ])
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
def small_dp3_checkpoint(tmp_path, small_dp3_config):
    from dexmani_policy.training.trainer import build_trainer

    directory = tmp_path / "dp3_run"
    with closing(build_trainer(small_dp3_config(directory))) as trainer:
        trainer.train(max_updates=1)
    return directory / "checkpoints/latest.pt"
