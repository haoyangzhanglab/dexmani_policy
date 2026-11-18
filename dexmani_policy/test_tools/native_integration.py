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
    from dexmani_sim.data_process.hdf5_to_zarr import HDF5ToZarrConverter
    from dexmani_sim.mimic_gen.utils.env_recorder import EnvRecorder
    real_geometry = pytest.importorskip(
        "dexmani_real.geometry.hoi", reason="geometry parity check requires the Real producer",
    )
    import h5py

    runner = SimRunner("pick_apple_messy", 2, [], {}, 0.0625, env_kwargs={"control_mode": "joint"})
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

    with h5py.File(episode, "r") as file:
        points = file["object_geometry/points_local"][:]
        offsets = file["object_geometry/offsets"][:]
        expected = real_geometry.geometry_digest(points, offsets)
    source = HDF5ToZarrConverter(episode.parent, tmp_path / "source.zarr").convert()
    labelled = tmp_path / "contact.zarr"
    generate(source, labelled)
    prepared = zarr.open_group(str(labelled), mode="r")
    assert prepared.attrs["data_processing"]["object_pointcloud"]["geometry_sha256"] == expected
    assert prepared.attrs["source_zarr_path"] == str(source)
    np.testing.assert_array_equal(prepared["meta/episode_seeds"][:], [32])
    assert "action" not in prepared["data"]
    dataset = CordViPPretrainDataset(labelled, contact_loss_weight=1, obs_horizon=2)
    labels = dataset[0]["contact_map"]
    assert labels.shape == (2, 1024) and labels.isfinite().all() and labels.std() > 0

    reordered_dir = tmp_path / "reordered"
    reordered_dir.mkdir()
    reordered = reordered_dir / "episode_0033.h5"
    shutil.copyfile(episode, reordered)
    points[:2] = points[:2][::-1]
    with h5py.File(reordered, "r+") as file:
        file.attrs["seed"] = 33
        file["object_geometry/points_local"][:2] = points[:2]
        world = file["object_pointcloud"]
        world[:, :2] = world[:, :2][:, ::-1]
    other = HDF5ToZarrConverter(reordered_dir, tmp_path / "reordered.zarr").convert()
    processed = zarr.open_group(str(other), mode="r").attrs["data_processing"]
    other_digest = processed["object_pointcloud"]["geometry_sha256"]
    assert other_digest == real_geometry.geometry_digest(points, offsets) != expected

    shutil.copyfile(episode, reordered_dir / episode.name)
    mixed = HDF5ToZarrConverter(reordered_dir, tmp_path / "mixed.zarr").convert()
    with pytest.raises(ValueError, match="shared object geometry_sha256"):
        generate(mixed, tmp_path / "invalid.zarr")
    assert len(CordViPPretrainDataset(mixed, contact_loss_weight=0, obs_horizon=2))

    missing_dir = tmp_path / "missing_geometry"
    missing_dir.mkdir()
    missing = missing_dir / episode.name
    shutil.copyfile(episode, missing)
    with h5py.File(missing, "r+") as file:
        del file["object_geometry"]
    with pytest.raises(KeyError, match="object_geometry"):
        HDF5ToZarrConverter(missing_dir, tmp_path / "missing.zarr").convert()
