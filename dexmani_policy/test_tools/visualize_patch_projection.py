"""Visualize the actual FPS centers and diagnose calibrated RGB correspondence.

Uses raw dataset RGB/XYZRGB/depth/segmentation and the SAT encoder recipe.
Segmentation footprint agreement is a spatial mixing proxy, not DINO semantics.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import zarr

from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import project_to_image
from dexmani_policy.agents.obs_encoder.pointcloud.ops import farthest_point_sample, preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.rgb.build import build_rgb_encoder
from dexmani_policy.training.common.config import load_policy_config


def grid_to_pixels(grid, height, width):
    return (grid + 1) * grid.new_tensor((width, height)) / 2 - 0.5


@torch.no_grad()
def inspect_frame(data, frame, processor, config, patch_size):
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
    tensor_rgb = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    processed, transform = processor.process_images(tensor_rgb, return_transform=True)
    height, width = processed.shape[-2:]
    assert height % patch_size == 0 and width % patch_size == 0
    raw = project_to_image(centers, intrinsic, extrinsic, torch.eye(3)[None], (rgb.shape[0], rgb.shape[1]))
    resized = project_to_image(centers, intrinsic, extrinsic, transform, (height, width))
    torch.testing.assert_close(raw.grid_xy, resized.grid_xy, atol=2e-6, rtol=0)
    assert torch.equal(raw.valid, resized.valid)
    uv = grid_to_pixels(raw.grid_xy, *rgb.shape[:2])[0].numpy()
    uv_resized = grid_to_pixels(resized.grid_xy, height, width)[0].numpy()
    valid = raw.valid[0].numpy()
    x = np.rint(uv[:, 0]).astype(int).clip(0, rgb.shape[1] - 1)
    y = np.rint(uv[:, 1]).astype(int).clip(0, rgb.shape[0] - 1)
    color_error = np.abs(colors * 255 - rgb[y, x].astype(float)).mean(-1)
    camera_z = (centers[0] * extrinsic[0, 2, :3]).sum(-1) + extrinsic[0, 2, 3]
    depth_mm = np.asarray(data["depth"][frame])[y, x]
    depth_valid = valid & (depth_mm > 0)
    depth_error = np.abs(camera_z.numpy() * 1000 - depth_mm.astype(float))
    depth_error[~depth_valid] = np.nan

    segmentation = torch.from_numpy(np.asarray(data["segmentation"][frame]).astype(np.int64))
    labels = segmentation[y, x]
    one_hot = F.one_hot(segmentation).permute(2, 0, 1)[None].float()
    resized_labels = F.interpolate(one_hot, size=(height, width), mode="nearest-exact")
    # Each token's image footprint is a patch-sized box; bilinear sampling mixes four boxes.
    footprint = F.avg_pool2d(resized_labels, patch_size)
    sampled = F.grid_sample(footprint, resized.grid_xy.unsqueeze(2),
                            mode="bilinear", padding_mode="border", align_corners=False)
    same_label_mass = sampled[0, :, :, 0].T.gather(1, labels[:, None]).squeeze(1).numpy()
    same_label_mass[~valid] = np.nan
    image = (processed * processor.image_std[None, :, None, None]
             + processor.image_mean[None, :, None, None])[0].permute(1, 2, 0).numpy().clip(0, 1)
    return dict(rgb=rgb, image=image, uv=uv, uv_resized=uv_resized, colors=colors,
                valid=valid, depth_error=depth_error, color_error=color_error,
                same_label_mass=same_label_mass, labels=labels.numpy())


def overlay(ax, frame):
    ax.imshow(frame["rgb"])
    valid = frame["valid"]
    ax.scatter(*frame["uv"][valid].T, c=frame["colors"][valid],
               s=22, edgecolors="white", linewidths=0.6)
    ax.set_axis_off()


def detail_figure(frame, title, patch_size, destination):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(frame["rgb"])
    raw_h, raw_w = frame["rgb"].shape[:2]
    axes[0].set_title(f"Raw RGB ({raw_w} x {raw_h})")
    axes[0].set_axis_off()
    overlay(axes[1], frame)
    axes[1].set_title(f"{len(frame['valid'])} FPS centers / recorded point RGB")
    axes[2].imshow(frame["image"])
    valid = frame["valid"]
    scatter = axes[2].scatter(*frame["uv_resized"][valid].T,
                             c=frame["same_label_mass"][valid], cmap="RdYlGn",
                             vmin=0, vmax=1, s=18, edgecolors="black", linewidths=0.3)
    for axis, size in ((axes[2].xaxis, frame["image"].shape[1]),
                       (axes[2].yaxis, frame["image"].shape[0])):
        axis.set_ticks(np.arange(-0.5, size, patch_size))
        axis.set_ticklabels([])
    axes[2].grid(alpha=0.25, color="white", linewidth=0.5)
    height, width = frame["image"].shape[:2]
    axes[2].set_title(f"{width} x {height} / {width // patch_size} x {height // patch_size} grid / label footprint")
    fig.colorbar(scatter, ax=axes[2], shrink=0.75, label="same-label mass (not DINO confidence)")
    fig.suptitle(title)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path,
                        default=sorted(Path("robot_data").glob("*.zarr")))
    parser.add_argument("--output", type=Path, default=Path("outputs/patch_projection"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = load_policy_config("sat")
    config = recipe.agent
    rgb_encoder, processor = build_rgb_encoder(config.rgb_backbone_name, dict(config.rgb_backbone_config))
    patch_size = rgb_encoder.patch_size
    del rgb_encoder
    rows, summary = [], {}
    for dataset in args.datasets:
        root = zarr.open_group(str(dataset), mode="r")
        ends = np.asarray(root["meta/episode_ends"][:])
        starts = np.r_[0, ends[:-1]]
        episodes = np.unique(np.linspace(0, len(ends) - 1, 3).astype(int))
        selection = [(int(ep), int(frame)) for ep in episodes
                     for frame in np.unique(np.linspace(starts[ep], ends[ep] - 1, 3).astype(int))]
        fig, axes = plt.subplots(len(episodes), 3, figsize=(12, 3.3 * len(episodes)),
                                 squeeze=False, constrained_layout=True)
        task_rows = []
        for index, (episode, frame_id) in enumerate(selection):
            frame = inspect_frame(root["data"], frame_id, processor, config, patch_size)
            ax = axes.flat[index]
            overlay(ax, frame)
            ax.set_title(f"episode {episode}, frame {frame_id} | valid {frame['valid'].sum()}/{len(frame['valid'])}")
            for patch in range(len(frame["valid"])):
                row = dict(task=dataset.stem, episode=episode, frame=frame_id, patch=patch,
                           valid=bool(frame["valid"][patch]),
                           u=float(frame["uv"][patch, 0]), v=float(frame["uv"][patch, 1]),
                           depth_error_mm=float(frame["depth_error"][patch]),
                           color_mae_255=float(frame["color_error"][patch]),
                           same_label_mass=float(frame["same_label_mass"][patch]),
                           segmentation_id=int(frame["labels"][patch]))
                task_rows.append(row)
            if index == len(selection) // 2:
                detail_figure(frame, f"{dataset.stem} | episode {episode}, frame {frame_id}",
                              patch_size, args.output / f"{dataset.stem}_detail.png")
        fig.suptitle(f"{dataset.stem} | actual deterministic FPS centers, source point colors")
        fig.savefig(args.output / f"{dataset.stem}_overview.png", dpi=150)
        plt.close(fig)
        valid_rows = [row for row in task_rows if row["valid"]]
        depth = np.array([row["depth_error_mm"] for row in valid_rows])
        depth = depth[np.isfinite(depth)]
        color = np.array([row["color_mae_255"] for row in valid_rows])
        mass = np.array([row["same_label_mass"] for row in valid_rows])
        summary[dataset.stem] = dict(
            frames=len(selection), centers=len(task_rows), valid=len(valid_rows),
            depth_valid=len(depth),
            depth_error_mm_median=float(np.median(depth)) if depth.size else None,
            depth_error_mm_p95=float(np.percentile(depth, 95)) if depth.size else None,
            depth_error_mm_max=float(depth.max()) if depth.size else None,
            color_mae_255_mean=float(color.mean()) if color.size else None,
            color_mae_255_p95=float(np.percentile(color, 95)) if color.size else None,
            same_label_mass_mean=float(mass.mean()) if mass.size else None,
            footprint_other_label_majority=int((mass < 0.5).sum()),
        )
        rows.extend(task_rows)
        print(dataset.stem, json.dumps(summary[dataset.stem]), flush=True)
    with (args.output / "centers.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
