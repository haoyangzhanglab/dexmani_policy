"""Isolated local geometry and gather measurements; no policies or datasets."""

import argparse
import statistics
import time

import torch
from torch import nn

from dexmani_policy.agents.modules import build_mlp
from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.pointcloud.point_transformer import LocalGeometryEmbedding


class PointNetReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_mlp = build_mlp([6, 64, 64], activate_last=True)
        self.projection = nn.Linear(64, 192)

    def forward(self, xyz, attributes):
        points = torch.cat((xyz.float() / 0.10, attributes.float()), -1)
        features = self.point_mlp(points.to(self.point_mlp[0].weight.dtype))
        return self.projection(features.amax(2))


def median_ms(step, args, prepare=None):
    for _ in range(args.warmup):
        if prepare is not None:
            prepare()
        step()
    times = []
    for _ in range(args.iterations):
        if prepare is not None:
            prepare()
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def geometry_case(batch, hidden, args):
    torch.manual_seed(args.seed)
    xyz = torch.randn(batch, 128, 32, 3, device=args.device) * 0.10
    attributes = torch.rand_like(xyz)
    torch.manual_seed(args.seed)
    model = (
        PointNetReference() if hidden is None else LocalGeometryEmbedding(6, 192, 0.10, hidden)
    ).to(args.device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hidden_dim = 64 if hidden is None else hidden
    hidden_macs = (1 if hidden is None else 4) * hidden_dim**2
    macs = 128 * 32 * (6 * hidden_dim + hidden_macs) + 128 * hidden_dim * 192

    def forward():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            return model(xyz, attributes)

    def forward_backward():
        forward().float().square().mean().backward()

    model.eval()
    with torch.inference_mode():
        forward_ms = median_ms(forward, args)
    model.train()
    # Match training's gradient allocation without timing gradient clearing.
    backward_ms = median_ms(
        forward_backward, args, prepare=lambda: model.zero_grad(set_to_none=True),
    )

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = forward()
    torch.cuda.synchronize()
    inference_peak = torch.cuda.max_memory_allocated()
    del output
    torch.cuda.reset_peak_memory_stats()
    output = forward()
    torch.cuda.synchronize()
    activation_peak = torch.cuda.max_memory_allocated() - base
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    training_peak = torch.cuda.max_memory_allocated()
    model_input = sum(p.numel() * p.element_size() for p in model.parameters())
    model_input += xyz.numel() * xyz.element_size() + attributes.numel() * attributes.element_size()
    mib = 1024**2
    name = "PointNet64" if hidden is None else f"Fusion{hidden}"
    print(
        f"| {name} | {batch} | {parameters} | {macs / 1e6:.3f} | "
        f"{forward_ms:.3f} | {backward_ms:.3f} | "
        f"{batch * 128 * 32 / forward_ms / 1000:.2f} | "
        f"{batch * 128 * 32 / backward_ms / 1000:.2f} | "
        f"{model_input / mib:.2f} | {base / mib:.2f} | {inference_peak / mib:.2f} | "
        f"{activation_peak / mib:.2f} | {training_peak / mib:.2f} | "
        f"{(training_peak - base) / mib:.2f} |", flush=True,
    )


def gather_case(batch, args):
    torch.manual_seed(args.seed)
    points = torch.randn(batch, 1024, 6, device=args.device)
    neighbors = torch.randint(1024, (batch, 128, 32), device=args.device)
    centers = points[:, :128, :3].contiguous()
    xyz = points[..., :3].contiguous()
    attributes = points[..., 3:]

    def old_gather():
        return index_points(xyz, neighbors), index_points(attributes, neighbors)

    def new_gather():
        return index_points(points, neighbors)

    def old_group():
        coords = points[..., :3].float().contiguous()
        relative = index_points(coords, neighbors) - centers.unsqueeze(2)
        attrs = index_points(points[..., 3:6], neighbors)
        return relative, attrs.float()

    def new_group():
        prepared = points[..., :6].float().contiguous()
        # Include the XYZ staging copy used by sampling, but exclude sampling itself.
        _xyz = prepared[..., :3].contiguous()
        grouped = index_points(prepared, neighbors)
        return grouped[..., :3] - centers.unsqueeze(2), grouped[..., 3:]

    with torch.inference_mode():
        torch.testing.assert_close(old_group(), new_group())
        for name, old, new in (
            ("Gather", old_gather, new_gather),
            ("Group preparation", old_group, new_group),
        ):
            old_ms, new_ms = median_ms(old, args), median_ms(new, args)
            print(f"| {name} | {batch} | {old_ms:.4f} | {new_ms:.4f} | {old_ms / new_ms:.2f} |", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.iterations < 20 or args.warmup < 1 or min(args.batch_sizes) < 1:
        parser.error("require iterations >= 20, warmup >= 1, positive batches")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    print(
        f"GPU: {torch.cuda.get_device_name()}; PyTorch: {torch.__version__}; "
        f"CUDA: {torch.version.cuda}; precision: {args.precision}; matmul: highest; "
        f"warmup: {args.warmup}; iterations: {args.iterations}; seed: {args.seed}", flush=True,
    )
    print(
        "Linear MACs are per cloud. Latencies are synchronized wall-clock medians. "
        "F+B includes scalar loss, excludes zero_grad and optimizer. "
        "Memory is allocated MiB, not reserved; Base also includes persistent backend allocations. "
        "Training delta includes gradients and temporaries. "
        "Train-forward delta observes activations/temporaries before backward. "
        "Throughput counts grouped points including repeated neighbors.", flush=True,
    )
    print(
        "| Model | B | Params | MACs M/cloud | F ms | F+B ms | F Mpoints/s | F+B Mpoints/s | "
        "Model+inputs MiB | Base MiB | Infer peak MiB | Train-forward delta MiB | "
        "F+B peak MiB | F+B delta MiB |"
    )
    print("|" + "---|" * 14)
    for batch in args.batch_sizes:
        for hidden in [None, 32, 64]:
            try:
                geometry_case(batch, hidden, args)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM: B={batch}, hidden={hidden}, precision={args.precision}", flush=True)
            torch.cuda.empty_cache()
    print("\nGather uses FP32 inputs and fixed synthetic indices; sampling is excluded.")
    print("| Operation | B | Double ms | Single ms | Speedup |\n|---|---|---|---|---|")
    for batch in args.batch_sizes:
        try:
            gather_case(batch, args)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM: gather B={batch}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
