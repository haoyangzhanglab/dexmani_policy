"""Opt-in tests of native dependencies. Pass this file explicitly to pytest."""

import shutil

import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.datasets.preprocess.contact_map import generate
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.env_runner.sim_runner import SimRunner


def test_native_cuda_point_ops_and_bfloat16():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("native integration requires CUDA with BF16 support")
    ops = pytest.importorskip("pytorch3d.ops", reason="native integration requires built PyTorch3D")
    points = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [0., 1., 0.]]],
        device="cuda", requires_grad=True,
    )
    sampled, indices = ops.sample_farthest_points(points, K=2)
    neighbors = ops.knn_points(points, points, K=2)
    (sampled.sum() + neighbors.dists.sum()).backward()
    assert indices.tolist() == [[0, 2]] and neighbors.idx[0, 2].tolist() == [2, 1]
    assert points.grad.isfinite().all()
    model = torch.nn.Linear(3, 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(points)
    assert result.dtype == torch.bfloat16 and result.isfinite().all()


def test_sim_recording_to_contact_pretraining(tmp_path):
    pytest.importorskip("dexmani_sim", reason="native SAPIEN integration requires dexmani_sim")
    pytest.importorskip("open3d", reason="contact labels require Open3D")
    import h5py
    from dexmani_sim import OBSERVATION_KEYS
    from dexmani_sim.data_process.hdf5_to_zarr import HDF5ToZarrConverter
    from dexmani_sim.mimic_gen.utils.env_recorder import EnvRecorder

    runner = SimRunner(
        "pick_apple_messy", 2, list(OBSERVATION_KEYS), {}, 0.0625, env_kwargs={"control_mode": "joint"},
    )
    env = runner.make_env(record_video=False)
    try:
        obs, _ = env.reset(seed=32)
        recorder = EnvRecorder(env, output_root=tmp_path)
        recorder.reset(32)
        recorder.record_initial_obs(obs)
        for _ in range(3):
            sensor = env.object_pcd_sensor
            vertices = []
            for entity, points in zip(sensor.entities, sensor.local_point_clouds):
                pose = np.asarray(entity.get_pose().to_transformation_matrix(), dtype=np.float32)
                vertices.append(points[:, :3] @ pose[:3, :3].T + pose[:3, 3])
            np.testing.assert_allclose(recorder.obs_buffer["object_pointcloud"][-1][:, :3], np.concatenate(vertices))
            recorder.record_step(env.robot.get_qpos().copy())
        episode = recorder.save_episode()
    finally:
        env.close()

    reordered = episode.with_name("episode_0033.h5")
    shutil.copyfile(episode, reordered)
    with h5py.File(reordered, "r+") as file:
        file.attrs["seed"] = 33
        world = file["object_pointcloud"]
        world[:] = world[:][:, ::-1]

    source = HDF5ToZarrConverter(episode.parent, tmp_path / "source.zarr").convert()
    labelled = tmp_path / "contact.zarr"
    generate(source, labelled)
    prepared = zarr.open_group(str(labelled), mode="r")
    assert prepared.attrs["source_zarr_path"] == str(source)
    np.testing.assert_array_equal(prepared["meta/episode_seeds"][:], [32, 33])
    assert "action" not in prepared["data"]
    points = prepared["data/object_pointcloud"][:]
    contacts = prepared["data/contact_map"][:]
    np.testing.assert_array_equal(points[3:], points[:3, ::-1])
    np.testing.assert_allclose(contacts[3:], contacts[:3, ::-1], rtol=1e-4, atol=1e-6)

    dataset = CordViPPretrainDataset(labelled, contact_loss_weight=1, obs_horizon=2)
    labels = dataset[0]["contact_map"]
    assert labels.shape == (2, 1024) and labels.isfinite().all() and labels.std() > 0
