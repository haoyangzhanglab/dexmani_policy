"""Offline CordViP contact labels, adapted from xuanxuanzzzii/cordvip.

Commit: ad6d441abe7b0480cf79b0311db266965c1313b6
Source: train_policy/script/contact_map.py
Changes: chunked NumPy geometry, lazy Open3D import, native Zarr sidecar IO.
"""

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import zarr

from dexmani_policy.common.io_util import close_resources
from dexmani_policy.datasets.common.geometry import contact_source_metadata
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.validation import validate_fields


def estimate_normals(points):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    return np.asarray(cloud.normals)


def compute_contact_map(object_points, object_normals, hand_points, *, gamma=1.0, theta=10.0, point_chunk_size=128):
    objects = np.asarray(object_points, dtype=np.float64)
    normals = np.asarray(object_normals, dtype=np.float64)
    hand = np.asarray(hand_points, dtype=np.float64)
    contacts = np.empty(len(objects), dtype=np.float64)
    for start in range(0, len(objects), point_chunk_size):
        stop = start + point_chunk_size
        difference = hand[None] - objects[start:stop, None]
        distance = np.linalg.norm(difference, axis=-1)
        direction = difference / (distance[..., None] + 1e-8)
        alignment = np.abs(np.einsum("ijk,ik->ij", direction, normals[start:stop]))
        minimum = (distance * np.exp(gamma * (1 - alignment))).min(axis=1)
        contacts[start:stop] = 1 - 2 * (1 / (1 + np.exp(-theta * minimum)) - 0.5)
    return contacts


def generate(zarr_path, output):
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Contact output already exists: {output}")
    buffer = ReplayBuffer(zarr_path, keys=["hand_pointcloud", "object_pointcloud"])
    source = contact_source_metadata(buffer)
    validate_fields(buffer, np.ones(buffer.n_episodes, dtype=bool), {
        "hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".contact-", dir=output.parent))
    try:
        group = zarr.open_group(str(temporary), mode="w")
        group.attrs.update(source=source, source_path=buffer.path,
                           recipe={"gamma": 1.0, "theta": 10.0, "knn": 30})
        group.create_group("meta").create_dataset("episode_ends", data=buffer.episode_ends)
        labels = group.create_group("data").create_dataset(
            "contact_map", shape=(source["rows"], buffer["object_pointcloud"].shape[1]),
            chunks=(16, buffer["object_pointcloud"].shape[1]), dtype="float32",
        )
        for start in range(0, source["rows"], 16):
            stop = min(start + 16, source["rows"])
            hands = buffer.read("hand_pointcloud", slice(start, stop), columns=slice(0, 3))
            objects = buffer.read("object_pointcloud", slice(start, stop), columns=slice(0, 3))
            labels[start:stop] = np.stack([
                compute_contact_map(obj, estimate_normals(obj), hand) for obj, hand in zip(objects, hands)
            ]).astype(np.float32)
            print(f"Contact rows {stop}/{source['rows']}", flush=True)
        if output.exists():
            raise FileExistsError(f"Contact output already exists: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            close_resources(lambda: shutil.rmtree(temporary))
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args.zarr_path, args.output)


if __name__ == "__main__":
    main()
