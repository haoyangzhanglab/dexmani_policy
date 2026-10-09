"""只读抽样 Zarr，验证投影与 PointPatch 语义融合；不训练策略。

默认先比较外参方向，不自动选方向。真实 backbone 验证必须显式指定方向。
用法见 docs/point_patch_semantics.md；从仓库根目录以 python -m 运行。
"""

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.obs_encoder.pointcloud.semantic_fusion import (
    project_points_to_images,
    resize_crop_transform,
)


def select_frames(episode_ends, num_frames):
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("episode_ends must be a nonempty strictly increasing vector")
    starts = np.r_[0, ends[:-1]]
    episodes = np.unique(np.linspace(0, len(ends) - 1, min(len(ends), math.ceil(num_frames / 3)), dtype=int))
    frames = []
    for episode in episodes:
        indices = np.unique(np.linspace(starts[episode], ends[episode] - 1, min(3, ends[episode] - starts[episode]), dtype=int))
        frames.extend((int(episode), int(index)) for index in indices)
    return frames[:num_frames]


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "p50": None, "p95": None}
    return {"count": int(values.size), "p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def load_frame(data, frame, depth_scale, device):
    rgb = np.asarray(data["rgb"][frame])
    depth = np.asarray(data["depth"][frame])
    pointcloud = np.asarray(data["point_cloud"][frame])
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("this validator expects single-view uint8 RGB [H,W,3]")
    if depth.shape != rgb.shape[:2] or pointcloud.ndim != 2 or pointcloud.shape[-1] != 6:
        raise ValueError("expected RGB-aligned depth [H,W] and point_cloud [N,6]")
    if not np.isfinite(pointcloud).all() or pointcloud[:, 3:].min() < 0 or pointcloud[:, 3:].max() > 1:
        raise ValueError("point_cloud must be finite XYZ in meters plus RGB in [0,1]")
    intrinsic = np.asarray(data["camera_intrinsic"][frame], dtype=np.float32).reshape(3, 3)
    stored = np.asarray(data["camera_extrinsic"][frame], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    if stored.size == 12:
        extrinsic[:3] = stored.reshape(3, 4)
    elif stored.size == 16:
        extrinsic[:] = stored.reshape(4, 4)
    else:
        raise ValueError("camera_extrinsic must contain 12 or 16 values")
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise ValueError("camera matrices must be finite")
    np.testing.assert_allclose(intrinsic[2], [0, 0, 1], atol=1e-5)
    np.testing.assert_allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-5)
    np.testing.assert_allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-3)
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or np.linalg.det(extrinsic[:3, :3]) < 0:
        raise ValueError("expected positive focal lengths and a proper rigid camera transform")
    return {
        "rgb": torch.from_numpy(rgb).permute(2, 0, 1)[None, None].to(device),
        "pointcloud": torch.from_numpy(pointcloud).float()[None].to(device),
        "depth_m": torch.from_numpy(depth.astype(np.float32) * depth_scale)[None, None].to(device),
        "intrinsics": torch.from_numpy(intrinsic)[None, None].to(device),
        "stored_extrinsic": torch.from_numpy(extrinsic)[None, None].to(device),
    }


def projection_stats(result, sample):
    mask = result["depth_valid"][0, 0]
    uv = result["raw_uv"][0, 0]
    rgb = sample["rgb"][0, 0].permute(1, 2, 0).float() / 255
    height, width = rgb.shape[:2]
    col = (uv[:, 0].clamp(0, width - 1) + .5).floor().long()
    row = (uv[:, 1].clamp(0, height - 1) + .5).floor().long()
    color_error = (rgb[row, col] - sample["pointcloud"][0, :, 3:]).abs().mean(-1)
    return {
        "positive_z_fraction": float((result["camera_depth"] > 0).float().mean()),
        "in_image_fraction": float(result["in_image"].float().mean()),
        "valid_depth_fraction": float(result["depth_valid"].float().mean()),
        "visible_fraction": float((result["view_weight"] > 0).float().mean()),
        # 在所有视野内且有深度的点上统计；不能仅统计已通过容差的点。
        "depth_residual_m": percentiles(result["depth_residual"][0, 0, mask].cpu().numpy()),
        "point_rgb_mae": percentiles(color_error[mask].cpu().numpy()),
    }


def save_overlay(sample, candidates, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = sample["rgb"][0, 0].permute(1, 2, 0).cpu().numpy()
    fig, axes = plt.subplots(1, len(candidates), figsize=(6 * len(candidates), 5), squeeze=False)
    for ax, (name, result) in zip(axes[0], candidates.items()):
        uv = result["raw_uv"][0, 0].cpu().numpy()
        inside = result["in_image"][0, 0].cpu().numpy()
        visible = (result["view_weight"][0, 0] > 0).cpu().numpy()
        ax.imshow(rgb)
        for mask, color, label in ((inside & ~visible, "red", "depth rejected"), (visible, "lime", "visible")):
            ax.scatter(uv[mask, 0], uv[mask, 1], s=5, c=color, alpha=.65, label=label)
        ax.set_title(name)
        ax.set_xlim(-.5, rgb.shape[1] - .5)
        ax.set_ylim(rgb.shape[0] - .5, -.5)
        ax.legend(loc="lower right", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def timed(call, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, (time.perf_counter() - start) * 1000


def create_encoders(args, device):
    from dexmani_policy.agents.obs_encoder.pointcloud.point_patch import PointPatchEncoder
    from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor

    if args.backbone == "dino":
        from dexmani_policy.agents.obs_encoder.rgb.dino import DINO
        vision_class, default_name = DINO, "facebook/dinov2-base"
    else:
        from dexmani_policy.agents.obs_encoder.rgb.siglip import SigLIP
        vision_class, default_name = SigLIP, "google/siglip-base-patch16-224"
    # 必须加载真实预训练权重；失败直接报告，不用随机视觉模型替代。
    vision = vision_class(model_name=args.model_name or default_name, tune_mode="freeze", out_dim=None)
    vision = vision.float().to(device).eval()
    processor = ImageProcessor.from_preset(args.backbone)
    processor.image_size = (args.image_size, args.image_size)
    if args.image_size % vision.patch_size:
        raise ValueError("image-size must be divisible by the selected backbone patch_size")
    point_encoder = PointPatchEncoder(semantic_channels=vision.out_dim).to(device).eval()
    return vision, processor, point_encoder


def save_patch_overlay(image, processor, output, correspondence, grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = image[0, 0].float().cpu()
    rgb = (rgb * processor.image_std[:, None, None] + processor.image_mean[:, None, None]).clamp(0, 1)
    valid = output["semantic_valid_mask"][0].nonzero().flatten()
    chosen = valid[torch.linspace(0, len(valid) - 1, min(3, len(valid)), device=valid.device).long()]
    fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5), squeeze=False)
    height, width = image.shape[-2:]
    for ax, index in zip(axes[0], chosen.tolist()):
        weight = output["pool_weight"][0, index].reshape(grid).float().cpu().numpy()
        center_idx = output["patch_center_idx"][0, index]
        uv = correspondence["point_uv"][0, 0, center_idx].cpu().numpy()
        ax.imshow(rgb.permute(1, 2, 0).numpy())
        ax.imshow(weight, cmap="magma", interpolation="nearest", alpha=.55,
                  extent=(-.5, width - .5, height - .5, -.5), vmin=0, vmax=max(float(weight.max()), 1e-8))
        ax.scatter(*uv, marker="x", c="cyan", s=60)
        ax.set_title(f"patch {index}; coverage={float(output['semantic_coverage'][0, index]):.2f}")
        ax.set_xlim(-.5, width - .5)
        ax.set_ylim(height - .5, -.5)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def validate_features(sample, extrinsic, encoders, args, device, check_backward, overlay_path=None):
    vision, processor, encoder = encoders
    processed = processor.process_images(sample["rgb"])
    image = processed["image"]
    spatial = processed["spatial"]
    transform = resize_crop_transform(spatial["orig_hw"], spatial["resized_hw"], spatial["crop_top_left"], device)
    grid = (image.shape[-2] // vision.patch_size, image.shape[-1] // vision.patch_size)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    def projection():
        return project_points_to_images(
            sample["pointcloud"][..., :3], sample["depth_m"], sample["intrinsics"], extrinsic,
            transform, depth_atol=args.depth_atol, depth_rtol=args.depth_rtol,
        )

    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        if check_backward:
            for _ in range(args.warmup):
                features = vision(image)["patch_tokens"]
                encoder(sample["pointcloud"], image_tokens=features, **projection(),
                        image_hw=tuple(image.shape[-2:]), patch_grid_size=grid)
        features, vision_ms = timed(lambda: vision(image)["patch_tokens"], device)
        correspondence, projection_ms = timed(projection, device)
        kwargs = dict(image_tokens=features, **correspondence, image_hw=tuple(image.shape[-2:]), patch_grid_size=grid)
        # 正常 forward 不返回诊断张量；随后单独获取诊断，避免把诊断开销计入计时。
        _, encoder_ms = timed(lambda: encoder(sample["pointcloud"], **kwargs), device)
        output = encoder(sample["pointcloud"], return_intermediate=True, **kwargs)
        for key in ("patch_token", "semantic_token", "semantic_gate", "pool_weight"):
            if not torch.isfinite(output[key]).all():
                raise AssertionError(f"nonfinite {key}")
        if not output["semantic_valid_mask"].any():
            raise AssertionError("no patch has valid correspondence; inspect calibration before interpreting fusion")
        torch.testing.assert_close(output["pool_weight"].sum(-1), output["semantic_valid_mask"].float(), atol=2e-5, rtol=2e-5)
        result = {
            "patch_shape": list(output["patch_token"].shape), "image_token_shape": list(features.shape),
            "patch_valid_fraction": float(output["semantic_valid_mask"].float().mean()),
            "coverage": percentiles(output["semantic_coverage"].cpu().numpy()),
            "vision_ms": vision_ms, "projection_ms": projection_ms, "point_encoder_ms": encoder_ms,
        }
        if overlay_path is not None:
            save_patch_overlay(image, processor, output, correspondence, grid, overlay_path)
            result["patch_overlay"] = overlay_path.name
        if check_backward:
            absent_kwargs = {**kwargs, "view_weight": torch.zeros_like(correspondence["view_weight"])}
            absent = encoder(sample["pointcloud"], **absent_kwargs)["patch_token"]
            geometry = encoder(sample["pointcloud"])["patch_token"]
            torch.testing.assert_close(absent, geometry, atol=1e-5, rtol=1e-5)
            result["missing_semantics_identity"] = "PASS"
    if check_backward:
        encoder.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            token = encoder(sample["pointcloud"], **kwargs)["patch_token"]
            (token * torch.randn_like(token)).float().mean().backward()
        grad = encoder.semantic_fusion.proj.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise AssertionError("semantic projection must receive finite nonzero gradients")
        if any(parameter.grad is not None for parameter in vision.parameters()):
            raise AssertionError("frozen vision backbone unexpectedly received gradients")
        encoder.zero_grad(set_to_none=True)
        result["fusion_backward"] = "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("robot_data/pick_apple_messy.zarr"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--depth-scale-m-per-unit", type=float, required=True)
    parser.add_argument("--extrinsic-convention", choices=("compare", "world-to-camera", "camera-to-world"), default="compare")
    parser.add_argument("--backbone", choices=("none", "dino", "siglip"), default="none")
    parser.add_argument("--model-name")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth-atol", type=float, default=.01)
    parser.add_argument("--depth-rtol", type=float, default=.01)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()
    if args.num_frames < 1 or args.image_size < 1 or args.warmup < 0:
        parser.error("num-frames/image-size must be positive and warmup nonnegative")
    if not math.isfinite(args.depth_scale_m_per_unit) or args.depth_scale_m_per_unit <= 0:
        parser.error("depth-scale-m-per-unit must be finite and positive")
    if args.backbone != "none" and args.extrinsic_convention == "compare":
        parser.error("choose an explicit extrinsic convention after inspecting geometry/source evidence")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path("outputs") / ("point_patch_semantics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        "status": "RUNNING", "commit": commit, "tracked_changes": dirty,
        "dataset": str(args.dataset.resolve()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "frames": [], "policy_success": "NOT_VERIFIED",
        "note": "PointPatch is randomly initialized for interface/gradient checks; vision uses pretrained weights. Compute timings exclude I/O, preprocessing, H2D, overlays and weight loading.",
    }
    try:
        import zarr
        if not args.dataset.is_dir():
            raise FileNotFoundError(f"Zarr directory not found: {args.dataset.resolve()}")
        root = zarr.open_group(str(args.dataset), mode="r")
        data = root["data"]
        required = ("rgb", "depth", "point_cloud", "camera_intrinsic", "camera_extrinsic")
        for key in required:
            if key not in data:
                raise KeyError(f"required data/{key} is missing")
        ends = np.asarray(root["meta/episode_ends"][:])
        selection = select_frames(ends, args.num_frames)
        if any(data[key].shape[0] != int(ends[-1]) for key in required):
            raise ValueError("frame counts must match episode_ends[-1]")
        recorded_scale = root.attrs.get("depth_scale_m_per_unit")
        if recorded_scale is not None and not math.isclose(float(recorded_scale), args.depth_scale_m_per_unit, rel_tol=1e-5):
            raise ValueError("requested depth scale conflicts with dataset metadata")
        report["dataset_schema"] = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in required}
        report["frame_count"], report["episode_count"] = int(ends[-1]), len(ends)
        report["segmentation_available"] = "segmentation" in data
        encoders = None
        for episode, frame in selection:
            sample = load_frame(data, frame, args.depth_scale_m_per_unit, device)
            stored = sample["stored_extrinsic"]
            transforms = {"world-to-camera": stored, "camera-to-world": torch.linalg.inv(stored)}
            candidates = {
                name: project_points_to_images(
                    sample["pointcloud"][..., :3], sample["depth_m"], sample["intrinsics"], transform,
                    depth_atol=args.depth_atol, depth_rtol=args.depth_rtol, return_intermediate=True,
                ) for name, transform in transforms.items()
            }
            frame_report = {"episode": episode, "frame": frame, "geometry": {
                name: projection_stats(result, sample) for name, result in candidates.items()
            }}
            report["frames"].append(frame_report)
            if not args.skip_overlays:
                save_overlay(sample, candidates, output_dir / f"projection_{frame:07d}.png")
            if args.backbone != "none":
                if encoders is None:
                    encoders = create_encoders(args, device)
                    report["vision_model"] = encoders[0].model_name
                frame_report["features"] = validate_features(
                    sample, transforms[args.extrinsic_convention], encoders, args, device,
                    check_backward=len(report["frames"]) == 1,
                    overlay_path=None if args.skip_overlays else output_dir / f"patch_weights_{frame:07d}.png",
                )
            print(f"frame={frame} geometry=" + json.dumps(frame_report["geometry"]))
        report["status"] = "GEOMETRY_DIAGNOSTICS" if args.backbone == "none" else "PASS_FEATURE_NUMERICS"
        if args.backbone != "none":
            report["timings_ms"] = {key: percentiles([frame["features"][key] for frame in report["frames"]])
                                    for key in ("vision_ms", "projection_ms", "point_encoder_ms")}
            if device.type == "cuda":
                report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
