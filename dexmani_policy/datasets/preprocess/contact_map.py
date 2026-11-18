"""Build a self-contained CordViP Zarr with offline contact labels."""

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite


CHUNK_ROWS = 16
CONTACT_POINT_COUNT = 1024


def require_contact_point_count(buffer):
    if buffer["object_pointcloud"].shape[1] != CONTACT_POINT_COUNT:
        raise ValueError(
            f"{buffer.path}: CordViP contact supervision requires {CONTACT_POINT_COUNT} object point slots"
        )


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
        block = objects[start:stop]
        block_normals = normals[start:stop]
        distance_squared = np.zeros((len(block), len(hand)), dtype=np.float64)
        projection = np.zeros_like(distance_squared)
        for axis in range(3):
            difference = hand[None, :, axis] - block[:, None, axis]
            distance_squared += difference ** 2
            projection += difference * block_normals[:, None, axis]
        distance = np.sqrt(distance_squared, out=distance_squared)
        alignment = np.abs(projection, out=projection) / (distance + 1e-8)
        minimum = (distance * np.exp(1 - alignment)).min(axis=1)
        decay = np.exp(-10 * minimum)
        contacts[start:stop] = 2 * decay / (1 + decay)
    return contacts


def frame_contact_map(object_points, hand_points):
    return compute_contact_map(object_points, estimate_normals(object_points), hand_points).astype(np.float32)


def generate(zarr_path, output, num_workers=1):
    positive_int(num_workers, "num_workers")
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Contact output already exists: {output}")
    fields = {"hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3), "joint_state": None}
    buffer = EpisodeStore(zarr_path, keys=fields, pc_dim=3)
    require_contact_point_count(buffer)
    rows = int(buffer.episode_ends[-1])
    attrs = dict(buffer.attrs)
    attrs["source_zarr_path"] = buffer.source_path
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".contact-", dir=output.parent) as directory:
        temporary = Path(directory) / "data.zarr"
        group = zarr.open_group(zarr.DirectoryStore(temporary), mode="w")
        group.attrs.update(attrs)
        zarr.copy(buffer.meta, group)
        data = group.create_group("data")
        for key, columns in fields.items():
            array = buffer[key]
            shape = (*array.shape[:-1], 3) if columns is not None else array.shape
            data.create_dataset(
                key, shape=shape, chunks=(CHUNK_ROWS, *shape[1:]), dtype="float32",
            )
        labels = data.create_dataset(
            "contact_map", shape=(rows, CONTACT_POINT_COUNT),
            chunks=(CHUNK_ROWS, CONTACT_POINT_COUNT), dtype="float32",
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
                for key, columns in fields.items():
                    values = buffer.read(key, slice(start, stop), columns=columns)
                    if columns is not None:
                        check_finite(buffer, key, start, values, columns)
                        points[key] = values
                    data[key][start:stop] = values
                labels[start:stop] = np.stack(list(map_frames(
                    frame_contact_map, points["object_pointcloud"], points["hand_pointcloud"],
                )))
                print(f"Contact rows {stop}/{rows}", flush=True)
        temporary.rename(output)
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
