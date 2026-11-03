"""Build a self-contained CordViP Zarr with offline contact labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
import shutil
import tempfile

import numpy as np
import zarr

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite


CHUNK_ROWS = 16


def estimate_normals(points):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    return np.asarray(cloud.normals)


def compute_contact_map(object_points, object_normals, hand_points, *, point_chunk_size=128):
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
        minimum = (distance * np.exp(1 - alignment)).min(axis=1)
        contacts[start:stop] = 1 - 2 * (1 / (1 + np.exp(-10 * minimum)) - 0.5)
    return contacts


def frame_contact_map(object_points, hand_points):
    return compute_contact_map(object_points, estimate_normals(object_points), hand_points).astype(np.float32)


def generate(zarr_path, output, num_workers=1):
    positive_int(num_workers, "num_workers")
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Contact output already exists: {output}")
    source = zarr.open_group(zarr.DirectoryStore(Path(zarr_path).expanduser()), mode="r")
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
    if "action" in keys and buffer["action"].shape[1:] != (19,):
        raise ValueError("CordViP policy requires action[T,19]")
    rows = int(buffer.episode_ends[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".contact-", dir=output.parent))
    try:
        group = zarr.open_group(zarr.DirectoryStore(temporary), mode="w")
        retained_attrs = ("dt", "hand_pointcloud_config")
        group.attrs.update({key: buffer.attrs[key] for key in retained_attrs if key in buffer.attrs})
        group.create_group("meta").create_dataset("episode_ends", data=buffer.episode_ends)
        data = group.create_group("data")
        for key in keys:
            array = buffer[key]
            shape = (*array.shape[:-1], 3) if key in keys[:2] else array.shape
            data.create_dataset(
                key, shape=shape, chunks=(CHUNK_ROWS, *shape[1:]), dtype="float32",
            )
        labels = data.create_dataset(
            "contact_map", shape=(rows, buffer["object_pointcloud"].shape[1]),
            chunks=(CHUNK_ROWS, buffer["object_pointcloud"].shape[1]), dtype="float32",
        )
        # Open3D's OpenMP runtime cannot safely be inherited by forked workers.
        workers = (
            ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn"))
            if num_workers > 1 else nullcontext()
        )
        with workers as pool:
            map_frames = map if pool is None else pool.map
            for start in range(0, rows, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, rows)
                points = {}
                for key in keys:
                    columns = slice(0, 3) if key in keys[:2] else None
                    values = np.asarray(buffer.read(key, slice(start, stop), columns=columns), dtype=np.float32)
                    data[key][start:stop] = values
                    if key in keys[:2]:
                        check_finite(buffer, key, start, values, columns)
                        points[key] = values
                labels[start:stop] = np.stack(list(map_frames(
                    frame_contact_map, points["object_pointcloud"], points["hand_pointcloud"],
                )))
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
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    generate(args.zarr_path, args.output, args.num_workers)


if __name__ == "__main__":
    main()
