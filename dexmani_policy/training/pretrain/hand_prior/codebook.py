"""Export PCA-ordered hand states from a DQ-RISE VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
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
    decoded = model.decode(latent).float()
    if not torch.isfinite(decoded).all():
        raise ValueError("VQ decoder produced non-finite hand states")
    return decoded.clamp(-1, 1)


@torch.no_grad()
def export_codebook(checkpoint_path, output_path, *, device="cpu", overwrite=False) -> tuple[HandStateCodebook, dict]:
    """Decode and order the selected VQ checkpoint, then publish its runtime table."""
    output = Path(output_path)
    if output.suffix != ".npz":
        raise ValueError("Codebook path must use the .npz suffix")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Codebook already exists: {output}; use --overwrite explicitly")

    from sklearn.decomposition import PCA

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    split = checkpoint["split_metadata"]
    model = HandVQVAE.from_checkpoint(checkpoint, device=device).eval()
    codebook = HandStateCodebook(model.hand_dim, model.num_groups, model.codebook_size)
    poses = build_hand_prototypes(model).cpu()
    pca = PCA(n_components=1)
    projection = pca.fit_transform(poses.numpy())[:, 0]
    permutation = torch.from_numpy(np.argsort(projection))
    state = checkpoint["normalizer_state_dict"]
    scale = state["fields.hand.scale"].detach().float().cpu()
    offset = state["fields.hand.offset"].detach().float().cpu()
    codebook.set_prototypes(poses[permutation])
    metadata = {
        "model_config": checkpoint["model_config"],
        "split_manifest": split["split_manifest"],
        "ordering": "decoded_prototype_pca1_ascending",
        "selection": {
            "selection_step": int(checkpoint["epoch"]),
            "unit": "epoch",
            "split": split["selection_split"],
            "metric": split["selection_metric"],
            "value": checkpoint["metrics"][split["selection_metric"]],
        },
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output, overwrite=overwrite) as temporary:
        payload = {
            "sorted_hand_poses": codebook.sorted_hand_poses.detach().cpu().numpy(),
            "hand_normalizer_scale": scale.numpy(),
            "hand_normalizer_offset": offset.numpy(),
            "pose_space": np.asarray("normalized"),
            "hand_dim": np.asarray(codebook.hand_dim),
            "num_groups": np.asarray(codebook.num_groups),
            "codebook_size": np.asarray(codebook.codebook_size),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        }
        # An open file prevents NumPy from silently appending a second suffix.
        with open(temporary, "wb") as file:
            np.savez(file, **payload)
    return codebook, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    codebook, _ = export_codebook(args.checkpoint, args.output, device=args.device, overwrite=args.overwrite)
    print(f"Exported {codebook.num_codes} hand states to {args.output}")


if __name__ == "__main__":
    main()
