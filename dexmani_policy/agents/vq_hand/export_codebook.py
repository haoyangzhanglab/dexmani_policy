"""Export a PCA-ordered runtime codebook from a trained VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dexmani_policy.agents.vq_hand import CodebookManager, VQVAEHand
from dexmani_policy.utils.atomic import atomic_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_hand_normalizer(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = checkpoint["normalizer_state_dict"]
    return (
        state["params_dict.hand.scale"].detach().cpu(),
        state["params_dict.hand.offset"].detach().cpu(),
    )


def extract_codebook(
    checkpoint_path: str,
    output_path: str,
    *,
    device: str = "cuda",
    overwrite: bool = False,
) -> CodebookManager:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Codebook already exists: {output}; use --overwrite explicitly")
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from dexmani_policy.agents.normalization import (
        ALLOWED_NORMALIZATION_MODES, validate_dq_normalization,
    )

    metadata = checkpoint["split_metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("VQ checkpoint split_metadata must be a mapping")
    spec = metadata.get("normalization_spec")
    # Standalone VQ declares action only; policy specs also include observations.
    if not isinstance(spec, dict) or any(
        not isinstance(key, str) or not key or not isinstance(mode, str)
        or mode not in ALLOWED_NORMALIZATION_MODES
        or (mode == "auto" and key != "action")
        for key, mode in spec.items()
    ):
        raise ValueError("VQ normalization_spec must map fields to valid modes")
    validate_dq_normalization(spec)
    vqvae = VQVAEHand.from_checkpoint(checkpoint, map_location="cpu")
    vqvae = vqvae.to(device).eval()

    manager = CodebookManager.extract_from_vqvae(vqvae)
    normalizer = _extract_hand_normalizer(checkpoint)
    manager.set_hand_normalizer(*normalizer)
    manager.artifact_metadata.update(
        {
            "source_checkpoint": str(Path(checkpoint_path).resolve()),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_epoch": int(checkpoint["epoch"]),
            "normalizer_sha256": hashlib.sha256(
                b"".join(t.contiguous().numpy().tobytes() for t in normalizer)
            ).hexdigest(),
            "checkpoint_metrics": checkpoint["metrics"],
            "split_metadata": checkpoint["split_metadata"],
        }
    )

    poses = manager.reindex_by_pca(vqvae)
    if output.suffix != ".npz":
        raise ValueError("Codebook path must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output, overwrite=overwrite, suffix=".tmp.npz") as temporary:
        manager.save(temporary)

    diagnostics = manager.last_export_diagnostics
    print(f"Extracted {len(poses)} prototypes with shape {poses.shape}")
    print("Layer weights:", manager.layer_weights.tolist())
    print("Decoder/export diagnostics:")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"Checkpoint SHA256: {manager.artifact_metadata['source_checkpoint_sha256']}")
    print(f"Saved: {output_path}")
    return manager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_codebook(
        args.checkpoint,
        args.output,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
