"""Overlay deterministic FPS centers on raw RGB using their source point colors."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import project_to_image
from dexmani_policy.agents.obs_encoder.pointcloud.ops import farthest_point_sample, preprocess_point_cloud
from dexmani_policy.training.common.config import load_policy_config


@torch.no_grad()
def overlay_frame(ax, data, frame, config):
    rgb = np.asarray(data["rgb"][frame])
    cloud = torch.from_numpy(np.asarray(data["point_cloud"][frame]))
    points = preprocess_point_cloud(
        cloud[None], config.num_points, config.fps_random_config, training=False,
    )
    patches = farthest_point_sample(points, config.pc_encoder_config.num_patches, use_random=False)
    centers = patches[..., :3]
    # Gather source RGB with the sampled XYZ, including when several points share XYZ.
    colors = patches[0, :, 3:6].numpy()
    intrinsic = torch.from_numpy(np.asarray(data["camera_intrinsic"][frame])).reshape(1, 3, 3)
    extrinsic = torch.from_numpy(np.asarray(data["camera_extrinsic"][frame])).reshape(1, 3, 4)
    projection = project_to_image(
        centers, intrinsic, extrinsic, torch.eye(3)[None], rgb.shape[:2],
    )
    grid = projection.grid_xy[0]
    height, width = rgb.shape[:2]
    uv = ((grid + 1) * grid.new_tensor((width, height)) / 2 - 0.5).numpy()
    valid = projection.valid[0].numpy()
    ax.imshow(rgb)
    ax.scatter(*uv[valid].T, c=colors[valid],
               s=22, edgecolors="white", linewidths=0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path,
                        default=sorted(Path("robot_data").glob("*.zarr")))
    parser.add_argument("--output", type=Path, default=Path("outputs/patch_projection"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_policy_config("sat").agent
    for dataset in args.datasets:
        root = zarr.open_group(str(dataset), mode="r")
        ends = np.asarray(root["meta/episode_ends"][:])
        starts = np.r_[0, ends[:-1]]
        episodes = np.unique(np.linspace(0, len(ends) - 1, 3).astype(int))
        fig, axes = plt.subplots(len(episodes), 3, figsize=(12, 3.3 * len(episodes)),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.flat:
            ax.set_axis_off()
        for row, episode in enumerate(episodes):
            frames = np.unique(np.linspace(starts[episode], ends[episode] - 1, 3).astype(int))
            for ax, frame in zip(axes[row], frames):
                overlay_frame(ax, root["data"], int(frame), config)
                ax.set_title(f"episode {episode}, frame {frame}")
        fig.suptitle(f"{dataset.stem} | actual deterministic FPS centers, source point colors")
        fig.savefig(args.output / f"{dataset.stem}_overview.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
