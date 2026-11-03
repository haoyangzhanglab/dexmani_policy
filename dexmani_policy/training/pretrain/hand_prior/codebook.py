"""Export PCA-ordered hand states from a DQ-RISE VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.io_util import atomic_path


@torch.no_grad()
def build_hand_prototypes(model: HandVQVAE) -> torch.Tensor:
    """Decode and clamp the runtime prototype table."""
    embeddings = model.codebooks
    weights = model.quantizer.layer_weights.softmax(0)
    combinations = torch.cartesian_prod(
        *[
            torch.arange(model.codebook_size, device=embeddings.device)
            for _ in range(model.num_groups)
        ]
    ).reshape(-1, model.num_groups)
    latent = embeddings.new_zeros(len(combinations), model.latent_dim)
    for group, weight in enumerate(weights):
        latent = latent + weight * embeddings[group, combinations[:, group]]
    decoded = model.decoder(latent).float()
    if not torch.isfinite(decoded).all():
        raise ValueError("VQ decoder produced non-finite hand states")
    return decoded.clamp(-1, 1)


@torch.no_grad()
def export_codebook(checkpoint_path, output_path, *, device="cpu", overwrite=False) -> int:
    """Decode and order a complete VQ checkpoint, then publish its runtime table."""
    output = Path(output_path)
    if output.suffix != ".npz":
        raise ValueError("Codebook path must use the .npz suffix")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Codebook already exists: {output}; use --overwrite explicitly")

    from sklearn.decomposition import PCA

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = HandVQVAE.from_checkpoint(checkpoint, device=device).eval()
    poses = build_hand_prototypes(model).cpu().numpy()
    if len(poses) > 1:
        projection = PCA(n_components=1).fit_transform(poses)[:, 0]
        poses = poses[np.argsort(projection)]
    state = checkpoint["normalizer_state_dict"]
    scale = state["fields.hand.scale"].detach().float().cpu()
    offset = state["fields.hand.offset"].detach().float().cpu()
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output) as temporary:
        payload = {
            "sorted_hand_poses": poses,
            "hand_normalizer_scale": scale.numpy(),
            "hand_normalizer_offset": offset.numpy(),
        }
        # An open file prevents NumPy from silently appending a second suffix.
        with open(temporary, "wb") as file:
            np.savez(file, **payload)
    return len(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = export_codebook(args.checkpoint, args.output, device=args.device, overwrite=args.overwrite)
    print(f"Exported {count} hand states to {args.output}")


if __name__ == "__main__":
    main()
