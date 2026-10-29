"""Compare hand-scene module cost; synthetic inputs do not measure policy quality."""

import argparse
import json
import subprocess
import types
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from dexmani_policy.agents.obs_encoder.interaction.hand_scene_encoder import HandSceneEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Trusted local Git ref containing the previous encoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    classes = []
    if args.baseline_ref:
        root = Path(__file__).resolve().parents[2]
        path = "dexmani_policy/agents/obs_encoder/interaction/hand_scene_encoder.py"
        source = subprocess.check_output(
            ["git", "show", f"{args.baseline_ref}:{path}"], cwd=root, text=True,
        )
        baseline = types.ModuleType("hand_scene_baseline")
        baseline.__package__ = "dexmani_policy.agents.obs_encoder.interaction"
        exec(compile(source, f"{args.baseline_ref}:{path}", "exec"), baseline.__dict__)
        classes.append(("baseline", baseline.HandSceneEncoder))
    classes.append(("current", HandSceneEncoder))

    results = []
    for batch in args.batch_sizes:
        torch.manual_seed(47)
        points = torch.randn(batch, 1024, 3, device=device) * .08
        patches = {
            "tokens": torch.randn(batch, 96, 192, device=device, requires_grad=True),
            "xyz": points[:, :96],
            "mask": torch.ones(batch, 96, dtype=torch.bool, device=device),
        }
        anchors = torch.randn(batch, 6, 3, device=device) * .03
        hand = torch.randn(batch, 6, 192, device=device, requires_grad=True)
        rotation = torch.eye(3, device=device).expand(batch, -1, -1)
        for label, cls in classes:
            torch.manual_seed(53)
            model = cls().to(device)

            def forward():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    return model(points, patches, anchors, hand, rotation_world_from_eef=rotation)

            def inference():
                with torch.inference_mode():
                    forward()

            def training():
                model.zero_grad(set_to_none=True)
                hand.grad = patches["tokens"].grad = None
                forward()["relation_update"].float().square().mean().backward()

            result = {"model": label, "batch": batch, "parameters": sum(p.numel() for p in model.parameters())}
            for name, operation in (("forward", inference), ("forward_backward", training)):
                model.train(name == "forward_backward")
                for _ in range(5):
                    operation()
                measurement = Timer(
                    stmt="operation()", globals={"operation": operation}, num_threads=args.threads,
                ).blocked_autorange(min_run_time=args.min_run_time)
                result[name + "_median_ms"] = measurement.median * 1000
                result[name + "_iqr_ms"] = measurement.iqr * 1000
            results.append(result)
    print(json.dumps({
        "torch": torch.__version__, "device": str(device), "threads": args.threads,
        "autocast": "bf16" if args.bf16 else "fp32", "points": 1024, "patches": 96,
        "token_channels": 192, "surface_k": 32, "return_intermediate": False,
        "baseline_ref": args.baseline_ref, "min_run_time": args.min_run_time, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
