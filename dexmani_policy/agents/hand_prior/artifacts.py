import json
from pathlib import Path

import numpy as np
import torch

from .codebook import HandStateCodebook


def save_codebook(
    path: str | Path, codebook: HandStateCodebook, metadata: dict, *,
    pca_permutation, layer_weights, hand_affine,
) -> None:
    payload = {
        "sorted_hand_poses": codebook.sorted_hand_poses.detach().cpu().numpy(),
        "pca_permutation": pca_permutation.detach().cpu().numpy(),
        "layer_weights": layer_weights.detach().cpu().numpy(),
        "hand_normalizer_scale": hand_affine["scale"].detach().cpu().numpy(),
        "hand_normalizer_offset": hand_affine["offset"].detach().cpu().numpy(),
    }
    payload.update(
        pose_space=np.asarray("normalized"),
        hand_dim=np.asarray(codebook.hand_dim),
        num_groups=np.asarray(codebook.num_groups),
        codebook_size=np.asarray(codebook.codebook_size),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    # An open file prevents NumPy from silently appending a second suffix.
    with open(path, "wb") as file:
        np.savez(file, **payload)


def load_codebook(path: str | Path, codebook: HandStateCodebook) -> tuple[dict, dict]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        if data["pose_space"].item() != "normalized":
            raise ValueError("Hand codebook requires normalized poses; regenerate the codebook")
        dimensions = tuple(data[name].item() for name in ("hand_dim", "num_groups", "codebook_size"))
        if dimensions != (codebook.hand_dim, codebook.num_groups, codebook.codebook_size):
            raise ValueError(f"Hand codebook dimensions do not match Policy: {dimensions}")
        poses = torch.from_numpy(data["sorted_hand_poses"])
        affine = {
            key: torch.from_numpy(data[f"hand_normalizer_{key}"])
            for key in ("scale", "offset")
        }
        metadata = json.loads(data["metadata_json"].item())
    for key, values in affine.items():
        if values.shape != (codebook.hand_dim,) or values.is_complex() or values.dtype == torch.bool:
            raise ValueError(f"Hand normalizer {key} requires {codebook.hand_dim} real values")
        affine[key] = values.float()
    if any(not torch.isfinite(value).all() for value in affine.values()) or (affine["scale"] == 0).any():
        raise ValueError("Hand normalizer must be finite and invertible in runtime dtype")
    codebook.set_prototypes(poses)
    return metadata, affine
