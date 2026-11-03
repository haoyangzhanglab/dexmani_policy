"""Check offline contact preprocessing; requires the preprocess extra."""
import tempfile
from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.datasets.preprocess.contact_map import compute_contact_map, generate


def check_contact_data(root):
    source = zarr.open_group(str(root / "source.zarr"), mode="w")
    source.create_dataset("meta/episode_ends", data=np.array([8, 16, 24]))
    source.attrs.update(dt=0.02, hand_pointcloud_config={"num_points": 8},
                        source_id="unused", format="unused", task_name="unused")
    rng = np.random.default_rng(7)
    for key, shape in (
        ("point_cloud", (24, 8, 3)), ("hand_pointcloud", (24, 8, 6)),
        ("object_pointcloud", (24, 1024, 6)), ("joint_state", (24, 19)),
        ("action", (24, 19)),
    ):
        array = source.create_dataset(f"data/{key}", data=rng.normal(size=shape))
        array.attrs["unused"] = True
    original = {key: source["data"][key][:] for key in source["data"]}
    # Warm Open3D before starting workers to catch unsafe fork-based execution.
    generate(root / "source.zarr", root / "serial.zarr", num_workers=1)
    generate(root / "source.zarr", root / "parallel.zarr", num_workers=2)
    serial = zarr.open_group(str(root / "serial.zarr"), mode="r")
    parallel = zarr.open_group(str(root / "parallel.zarr"), mode="r")
    for key, values in original.items():
        np.testing.assert_array_equal(source["data"][key][:], values)
    assert dict(serial.attrs) == {"dt": 0.02, "hand_pointcloud_config": {"num_points": 8}}
    assert set(serial["data"]) == {"hand_pointcloud", "object_pointcloud", "joint_state", "action", "contact_map"}
    for key in serial["data"]:
        actual = serial["data"][key][:]
        assert actual.dtype == np.float32 and not dict(serial["data"][key].attrs)
        np.testing.assert_allclose(actual, parallel["data"][key][:], rtol=1e-6, atol=1e-7)
        if key != "contact_map":
            values = source["data"][key][:]
            if "pointcloud" in key:
                assert actual.shape[-1] == 3 and values.shape[-1] == 6
                values = values[..., :3]
            np.testing.assert_array_equal(values.astype(np.float32), actual)
    np.testing.assert_array_equal(source["meta/episode_ends"][:], parallel["meta/episode_ends"][:])
    labels = parallel["data/contact_map"][:]
    assert labels.dtype == np.float32 and np.isfinite(labels).all()
    assert (labels >= 0).all() and (labels <= 1).all()
    policy_data = DemonstrationDataset(
        root / "serial.zarr", sensor_modalities=list(CordViPPretrainDataset.sensor_modalities),
        horizon=16, obs_horizon=3, pad_before=2, pad_after=7, pc_dim=3,
    )
    rows = policy_data.sampler.sample_source_rows(0)
    np.testing.assert_array_equal(policy_data[0]["action"], original["action"][rows].astype(np.float32))
    original_action = source["data/action"][:]
    del source["data/action"]
    generate(root / "source.zarr", root / "without_action.zarr")
    assert "action" not in zarr.open_group(str(root / "without_action.zarr"), mode="r")["data"]
    assert len(CordViPPretrainDataset(root / "without_action.zarr")) == 24
    try:
        DemonstrationDataset(root / "without_action.zarr", sensor_modalities=["joint_state"], obs_horizon=1)
    except KeyError:
        pass
    else:
        raise AssertionError("Policy dataset accepted missing action")
    source.create_dataset("data/action", data=original_action)
    try:
        generate(root / "source.zarr", root / "serial.zarr")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was overwritten")
    source["data/hand_pointcloud"][0, 0, 0] = np.nan
    try:
        generate(root / "source.zarr", root / "invalid.zarr")
    except ValueError as error:
        assert "Nonfinite data" in str(error)
    else:
        raise AssertionError("Nonfinite geometry was accepted")
    assert not (root / "invalid.zarr").exists() and not list(root.glob(".contact-*"))


def check_contact_formula():
    rng = np.random.default_rng(23)
    objects, hand = rng.normal(size=(17, 3)), rng.normal(size=(9, 3))
    normals = rng.normal(size=(17, 3))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    expected = []
    for point, normal in zip(objects, normals):
        difference = hand - point
        distance = np.linalg.norm(difference, axis=-1)
        alignment = np.abs(difference @ normal) / (distance + 1e-8)
        minimum: np.float64 = np.min(distance * np.exp(1 - alignment))
        expected.append(1 - 2 * (1 / (1 + np.exp(-10 * minimum)) - 0.5))
    for block in (1, 7, 128):
        for sign in (1, -1):
            actual = compute_contact_map(objects, sign * normals, hand, point_chunk_size=block)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def main():
    check_contact_formula()
    with tempfile.TemporaryDirectory(prefix="dexmani-contact-checks-") as directory:
        check_contact_data(Path(directory))
    print("PASS contact formula, serial/spawn generation, alignment and failure cleanup")


if __name__ == "__main__":
    main()
