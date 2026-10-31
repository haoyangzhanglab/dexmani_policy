"""Analyze an explicitly selected VQ checkpoint and its actual train/held-out rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.codebook import validate_hand_coordinates
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import LinearNormalizer, validate_normalizer_state
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.training.pretrain.hand_prior.codebook import deployment_diagnostics, export_codebook
from dexmani_policy.training.pretrain.hand_prior.vqvae import load_hand_array


@torch.no_grad()
def analyze(checkpoint_path, output_dir, *, device="cpu", zarr_path=None, chunk_size=4096):
    import matplotlib.pyplot as plt

    checkpoint_path, output = Path(checkpoint_path), Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint["split_metadata"]
    buffer = EpisodeStore(zarr_path or source["dataset_config"]["zarr_path"], keys=["action"])
    validate_hand_coordinates(buffer.attrs)
    train, val, _ = split_episodes(buffer, manifest=source["split_manifest"])
    normalizer = LinearNormalizer({"hand": (12,)})
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"], strict=True)
    validate_normalizer_state(normalizer)
    splits = {
        name: normalizer["hand"].normalize(load_hand_array(buffer, mask)).numpy()
        for name, mask in (("train", train), ("held_out", val))
    }
    output.mkdir(parents=True, exist_ok=False)
    table, metadata = export_codebook(checkpoint_path, output / "codebook.npz", device=device)
    report = deployment_diagnostics(
        table, splits, metadata=metadata, hand_scale=normalizer["hand"].scale,
        chunk_size=chunk_size,
    )
    model = HandVQVAE.from_checkpoint(checkpoint, device=device).eval()
    usage = torch.zeros(model.num_groups, model.codebook_size, dtype=torch.long, device=device)
    for rows in np.array_split(splits["train"], max(1, (len(splits["train"]) + chunk_size - 1) // chunk_size)):
        _, _, indices, _ = model(torch.as_tensor(rows, device=device))
        usage.scatter_add_(1, indices.T, torch.ones_like(indices.T))
    report["train_quantizer_group_counts"] = usage.cpu().tolist()
    with atomic_path(output / "codebook_diagnostics.json") as temporary:
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for group, counts in enumerate(usage.cpu()):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(model.codebook_size), counts.numpy())
        ax.set(xlabel="Code index", ylabel="Count", title=f"Group {group} selected-checkpoint usage")
        fig.tight_layout()
        fig.savefig(output / f"code_usage_g{group}.png")
        plt.close(fig)
    metrics_path = checkpoint_path.parent / "metrics.jsonl"
    if metrics_path.exists():
        history = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([row["epoch"] for row in history], [row["train_total"] for row in history])
        ax.set(xlabel="Epoch", ylabel="Training objective", title="VQ-VAE training objective")
        fig.tight_layout()
        fig.savefig(output / "loss_curve.png")
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--zarr-path")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    analyze(args.checkpoint, args.output_dir, device=args.device,
            zarr_path=args.zarr_path, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
