"""Distance percentiles for raw, common Base-frame coordinates in meters."""

import argparse

import numpy as np
import torch

from dexmani_policy.agents.obs_encoder.proprio import _rotation_6d_to_matrix
from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite


def distance_percentiles(path, max_frames=128, reference=(0.0, 0.0, 0.0)):
    positive_int(max_frames, "max_frames")
    reference = torch.as_tensor(reference, dtype=torch.float32)
    if reference.shape != (3,) or not torch.isfinite(reference).all():
        raise ValueError("reference must contain three finite Base-frame coordinates")
    store = EpisodeStore(path, keys=("eef_pose", "fingertip_xyz", "point_cloud"))
    shape = store["point_cloud"].shape
    if len(shape) != 3 or shape[1] == 0 or shape[2] < 3:
        raise ValueError(f"point_cloud requires [T,N>0,C>=3], got {shape}")
    total = int(store.episode_ends[-1])
    rows = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    distances: dict[str, list[np.ndarray]] = {
        key: [] for key in ("wrist_reference", "finger_local", "hand_scene")
    }
    for row in rows:
        values = {}
        for key in store.keys():
            columns = slice(0, 3) if key == "point_cloud" else None
            array = store.read(key, slice(row, row + 1), columns=columns)
            check_finite(store, key, row, array, columns)
            values[key] = torch.from_numpy(array)
        pose, tips, scene = values["eef_pose"], values["fingertip_xyz"], values["point_cloud"]
        rotation = _rotation_6d_to_matrix(pose[..., 3:])
        local = (tips - pose[..., None, :3]) @ rotation
        anchors = torch.cat((pose[..., None, :3], tips), dim=-2)
        distances["wrist_reference"].append((pose[..., :3] - reference).norm(dim=-1).numpy().ravel())
        distances["finger_local"].append(local.norm(dim=-1).numpy().ravel())
        distances["hand_scene"].append(
            (scene[:, None] - anchors[:, :, None]).norm(dim=-1).numpy().ravel()
        )
    return len(rows), {
        key: np.percentile(np.concatenate(values), [50, 90, 95, 99])
        for key, values in distances.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--reference", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()
    count, percentiles = distance_percentiles(args.path, args.max_frames, args.reference)
    print(f"{args.path}: {count} evenly spaced frames; reference={tuple(args.reference)}; meters")
    print("hand_scene includes all six anchors × all stored scene points.")
    print("| Distance | P50 | P90 | P95 | P99 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for key, values in percentiles.items():
        print(f"| {key} | " + " | ".join(f"{value:.6f}" for value in values) + " |")


if __name__ == "__main__":
    main()
