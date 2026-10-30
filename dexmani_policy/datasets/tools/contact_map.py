"""Build a self-contained CordViP Zarr with offline contact labels."""

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import zarr

from dexmani_policy.common.io_util import close_resources
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite


CHUNK_ROWS = 16


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
        alignment = np.abs(
            np.einsum("ijk,ik->ij", difference, normals[start:stop])
        ) / (distance + 1e-8)
        minimum = (distance * np.exp(gamma * (1 - alignment))).min(axis=1)
        contacts[start:stop] = 1 - 2 * (1 / (1 + np.exp(-theta * minimum)) - 0.5)
    return contacts


def generate(zarr_path, output):
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Contact output already exists: {output}")
    source = zarr.open_group(str(Path(zarr_path).expanduser()), mode="r")
    keys = ["hand_pointcloud", "object_pointcloud", "joint_state"]
    if "action" in source["data"]:
        keys.append("action")
    buffer = EpisodeStore(zarr_path, keys=keys)
    for key in keys[:2]:
        shape = buffer[key].shape
        if len(shape) != 3 or shape[1] < 1 or shape[2] < 3:
            raise ValueError(f"Contact geometry requires {key}[T,N>0,C>=3]")
    if buffer["object_pointcloud"].shape[1] != 1024:
        raise ValueError("CordViP requires 1024 object points")
    if buffer["joint_state"].shape[1:] != (19,):
        raise ValueError("CordViP requires joint_state[T,19]")
    rows = int(buffer.episode_ends[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".contact-", dir=output.parent))
    try:
        group = zarr.open_group(str(temporary), mode="w")
        retained_attrs = (
            "source_id", "episode_ids", "trial_ids", "format", "task_name",
            "dt", "physics_dt", "frame_skip", "obs_alignment", "action_space",
            "action_dim", "action_coordinates", "joint_order", "units",
            "hand_joint_order", "hand_units", "hand_pointcloud_config",
        )
        group.attrs.update({key: buffer.attrs[key] for key in retained_attrs if key in buffer.attrs})
        group.create_group("meta").create_dataset("episode_ends", data=buffer.episode_ends)
        data = group.create_group("data")
        for key in keys:
            array = buffer[key]
            copied = data.create_dataset(
                key, shape=array.shape, chunks=(CHUNK_ROWS, *array.shape[1:]),
                dtype=array.dtype, compressor=array.compressor,
            )
            copied.attrs.update(dict(array.attrs))
        labels = data.create_dataset(
            "contact_map", shape=(rows, buffer["object_pointcloud"].shape[1]),
            chunks=(CHUNK_ROWS, buffer["object_pointcloud"].shape[1]), dtype="float32",
        )
        for start in range(0, rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, rows)
            points = {}
            for key in keys:
                values = buffer[key][start:stop]
                data[key][start:stop] = values
                if key in keys[:2]:
                    # Geometry uses FP32 floating XYZ; copied arrays keep their stored dtype.
                    dtype = np.float32 if np.issubdtype(values.dtype, np.floating) else values.dtype
                    xyz = np.asarray(values[..., :3], dtype=dtype)
                    check_finite(buffer, key, start, xyz, slice(0, 3))
                    points[key] = xyz
            hands, objects = points["hand_pointcloud"], points["object_pointcloud"]
            labels[start:stop] = np.stack([
                compute_contact_map(obj, estimate_normals(obj), hand) for obj, hand in zip(objects, hands)
            ]).astype(np.float32)
            print(f"Contact rows {stop}/{rows}", flush=True)
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
