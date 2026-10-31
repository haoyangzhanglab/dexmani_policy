from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.artifacts import save_codebook
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
        "hand_data": split["hand_data"],
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
        save_codebook(
            temporary, codebook, metadata, hand_affine={"scale": scale, "offset": offset},
        )
    return codebook, metadata
