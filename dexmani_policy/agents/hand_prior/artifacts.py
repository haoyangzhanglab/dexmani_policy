import json
from pathlib import Path

import numpy as np
import torch

from .codebook import HandStateCodebook


def save_codebook(path: str | Path, codebook: HandStateCodebook, metadata: dict) -> None:
    payload = {name: getattr(codebook, name).detach().cpu().numpy() for name in codebook.BUFFER_NAMES}
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


def load_codebook(path: str | Path, codebook: HandStateCodebook) -> dict:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        if data["pose_space"].item() != "normalized":
            raise ValueError("Hand codebook requires normalized poses; regenerate the codebook")
        dimensions = tuple(data[name].item() for name in ("hand_dim", "num_groups", "codebook_size"))
        if dimensions != (codebook.hand_dim, codebook.num_groups, codebook.codebook_size):
            raise ValueError(f"Hand codebook dimensions do not match Policy: {dimensions}")
        values = {name: torch.from_numpy(data[name]) for name in codebook.BUFFER_NAMES}
        metadata = json.loads(data["metadata_json"].item())
    codebook.set_buffers(values)
    return metadata
