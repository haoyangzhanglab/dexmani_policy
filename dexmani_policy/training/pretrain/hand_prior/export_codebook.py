"""Export PCA-ordered hand states from a DQ-RISE VQ-VAE checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.training.pretrain.hand_prior.codebook import build_runtime_codebook


def export_codebook(checkpoint_path, output_path, *, device="cpu", overwrite=False):
    output = Path(output_path)
    if output.suffix != ".npz":
        raise ValueError("Codebook path must use the .npz suffix")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Codebook already exists: {output}; use --overwrite explicitly")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    codebook = build_runtime_codebook(checkpoint, checkpoint_path, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output, overwrite=overwrite, suffix=".tmp.npz") as temporary:
        codebook.save(temporary)
    print(f"Exported {codebook.num_codes} hand states to {output}")
    print("Layer weights:", codebook.layer_weights.tolist())
    print(json.dumps(codebook.artifact_metadata["export_diagnostics"], indent=2))
    return codebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_codebook(args.checkpoint, args.output, device=args.device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
