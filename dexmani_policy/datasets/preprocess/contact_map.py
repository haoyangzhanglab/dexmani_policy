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
CUDA_FRAME_BATCH = 4
CUDA_POINT_TILE = 1024


def contact_zarr_path(source: Path) -> Path:
    return source.parent / "cordvip" / source.name


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


def batched_contact_maps(object_points, object_normals, hand_points, device):
    import torch

    objects, normals, hand = (
        torch.as_tensor(values, dtype=torch.float64, device=device)
        for values in (object_points, object_normals, hand_points)
    )
    contacts = torch.empty(objects.shape[:2], dtype=torch.float32, device=device)
    # Bound the dense pairwise tensors independently of the point counts.
    for frame_start in range(0, len(objects), CUDA_FRAME_BATCH):
        batch = slice(frame_start, frame_start + CUDA_FRAME_BATCH)
        for start in range(0, objects.shape[1], CUDA_POINT_TILE):
            columns = slice(start, start + CUDA_POINT_TILE)
            points = objects[batch, columns]
            nx, ny, nz = normals[batch, columns].unbind(dim=-1)
            minimum = points.new_full(points.shape[:2], torch.inf)
            for hand_tile in hand[batch].split(CUDA_POINT_TILE, dim=1):
                dx, dy, dz = (hand_tile[:, None] - points[:, :, None]).unbind(dim=-1)
                distance = (dx.square() + dy.square() + dz.square()).sqrt()
                projection = dx * nx[..., None] + dy * ny[..., None] + dz * nz[..., None]
                projection.abs_().div_(distance + 1e-8)
                score = (1 - projection).exp_().mul_(distance)
                minimum = torch.minimum(minimum, score.amin(dim=-1))
            decay = (-10 * minimum).exp()
            contacts[batch, columns] = 2 * decay / (1 + decay)
    return contacts.cpu().numpy()


def generate(zarr_path, output=None, num_workers=1, *, device="cpu"):
    import torch

    positive_int(num_workers, "num_workers")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Contact generation supports device=cpu or cuda")
    source = Path(zarr_path).expanduser().absolute()
    output = contact_zarr_path(source) if output is None else Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Contact output already exists: {output}")
    fields = {"hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3), "joint_state": None}
    buffer = EpisodeStore(source, keys=fields, pc_dim=3)
    rows = int(buffer.episode_ends[-1])
    point_count = buffer["object_pointcloud"].shape[1]
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
            "contact_map", shape=(rows, point_count),
            chunks=(CHUNK_ROWS, point_count), dtype="float32",
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
                objects, hand = points["object_pointcloud"], points["hand_pointcloud"]
                if device.type == "cpu":
                    contacts = np.stack(list(map_frames(frame_contact_map, objects, hand)))
                else:
                    normals = np.stack(list(map_frames(estimate_normals, objects)))
                    contacts = batched_contact_maps(objects, normals, hand, device)
                labels[start:stop] = contacts
                print(f"Contact rows {stop}/{rows}", flush=True)
        temporary.rename(output)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output", help="Defaults to cordvip/<input name> alongside the input Zarr")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="Point-pair computation device, e.g. cuda:0")
    args = parser.parse_args()
    generate(args.zarr_path, args.output, args.num_workers, device=args.device)


if __name__ == "__main__":
    main()
