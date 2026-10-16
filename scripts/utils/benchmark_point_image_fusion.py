"""Synthetic fusion-only timing; does not benchmark DINO/SigLIP or a policy.

Example: python scripts/utils/benchmark_point_image_fusion.py --device cuda --backward
Reports correspondence build, pool/content fusion with reused geometry, and
end-to-end module times. No claim that sparse execution is faster than BMM.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

_PATH = Path(__file__).resolve().parents[2] / "dexmani_policy/agents/obs_encoder/interaction/point_image_fusion.py"
_SPEC = importlib.util.spec_from_file_location("_fusion_benchmark", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--patches", type=int, default=192)
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--grid", type=int, nargs=2, default=(16, 16))
    parser.add_argument("--image-channels", type=int, default=384)
    parser.add_argument("--token-channels", type=int, default=192)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=.2)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.views, args.points, args.patches, args.members,
           *args.grid, args.image_channels, args.token_channels, args.threads) < 1 or args.min_run_time <= 0:
        parser.error("dimensions, threads and min-run-time must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if device.type == "cuda" and args.bf16 and not torch.cuda.is_bf16_supported():
        parser.error("CUDA BF16 is unsupported")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    b, v, n, m, k = args.batch, args.views, args.points, args.patches, args.members
    gh, gw = args.grid
    local = torch.randn(b, m, args.token_channels, device=device, requires_grad=args.backward)
    image = torch.randn(b, v, gh * gw, args.image_channels, device=device, requires_grad=args.backward)
    # Random synthetic projections; real patch locality may differ substantially.
    uv = torch.rand(b, v, n, 2, device=device) * torch.tensor([gw * 14., gh * 14.], device=device)
    idx = torch.randint(n, (b, m, k), device=device)
    confidence = torch.rand(b, v, n, device=device)
    hw = (gh * 14, gw * 14)
    geom_args = (idx, uv, confidence, hw, (gh, gw))
    geometry = _MODULE.build_patch_image_weights(*geom_args)
    results = {}

    def measure(name, fn):
        for _ in range(3):
            fn()
        measurement = Timer(stmt="fn()", globals={"fn": fn}, num_threads=args.threads).blocked_autorange(min_run_time=args.min_run_time)
        results[name] = {"median_ms": measurement.median * 1000, "iqr_ms": measurement.iqr * 1000}

    measure("build_correspondence", lambda: _MODULE.build_patch_image_weights(*geom_args))
    for alignment in ("pool", "content"):
        model = _MODULE.PointImageFusion(args.image_channels, args.token_channels, alignment=alignment).to(device)
        if alignment == "content":
            with torch.no_grad():
                model.content_scale.fill_(.5)
        for reuse in (True, False):
            def run():
                model.zero_grad(set_to_none=True)
                local.grad = None
                image.grad = None
                with torch.set_grad_enabled(args.backward), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    if reuse:
                        output = model.fuse_tokens(local, image, geometry)
                    else:
                        output = model(local, idx, image, uv, confidence, hw, (gh, gw))
                    if args.backward:
                        output["fused_token"].square().mean().backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            name = f"{alignment}_{'reused' if reuse else 'full'}"
            measure(name, run)
            if device.type == "cuda":
                model.zero_grad(set_to_none=True)
                local.grad = None
                image.grad = None
                torch.cuda.synchronize(device)
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                run()
                results[name]["additional_peak_allocated_mib"] = (torch.cuda.max_memory_allocated(device) - baseline) / 2**20
    report = {"torch": torch.__version__, "device": str(device), "cuda_available": torch.cuda.is_available(),
              "settings": vars(args), "timings": results,
              "correspondence_mib": geometry["pool_weight"].numel() * geometry["pool_weight"].element_size() / 2**20,
              "mean_corresponding_tokens": (geometry["pool_weight"] > 0).sum(-1).float().mean().item(),
              "note": "Synthetic random tokens/projections; no pretrained model, training quality or deployment speed validation."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
