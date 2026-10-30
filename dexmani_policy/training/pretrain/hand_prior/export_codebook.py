"""Export PCA-ordered hand states from a DQ-RISE VQ-VAE checkpoint."""

import argparse
import json

import torch

from dexmani_policy.training.pretrain.hand_prior.codebook import export_codebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    codebook, metadata = export_codebook(args.checkpoint, args.output, device=args.device, overwrite=args.overwrite)
    print(f"Exported {codebook.num_codes} hand states to {args.output}")
    print("Layer weights:", codebook.layer_weights.tolist())
    print(json.dumps(metadata["export_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
