"""Extract a PCA-ordered runtime codebook from a trained VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dexmani_policy.agents.vq_hand import CodebookManager, VQVAEHand


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_hand_normalizer(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    state = checkpoint.get("normalizer_state_dict")
    if state is not None:
        scale_key = "params_dict.hand.scale"
        offset_key = "params_dict.hand.offset"
        if scale_key in state and offset_key in state:
            return state[scale_key].detach().cpu(), state[offset_key].detach().cpu()

    params = checkpoint.get("normalizer_params")
    try:
        hand = params["hand"]
        return hand["scale"].detach().cpu(), hand["offset"].detach().cpu()
    except (TypeError, KeyError, AttributeError):
        return None


def extract_codebook(
    checkpoint_path: str,
    output_path: str,
    *,
    device: str = "cuda",
    include_per_group: bool = False,
) -> CodebookManager:
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vqvae = VQVAEHand.from_checkpoint(checkpoint, map_location="cpu")
    vqvae = vqvae.to(device).eval()

    manager = CodebookManager.extract_from_vqvae(vqvae)
    normalizer = _extract_hand_normalizer(checkpoint)
    if normalizer is None:
        raise ValueError(
            "Checkpoint does not contain a recoverable hand normalizer. "
            "Re-train or convert the checkpoint before exporting a codebook."
        )
    manager.set_hand_normalizer(*normalizer)
    manager.artifact_metadata.update(
        {
            "source_checkpoint": str(Path(checkpoint_path).resolve()),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_metrics": checkpoint.get("metrics", {}),
            "split_metadata": checkpoint.get("split_metadata", {}),
        }
    )

    poses = manager.reindex_by_pca(vqvae)
    if include_per_group:
        manager.build_per_group_codebooks(vqvae)
    manager.save(output_path)

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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_per_group", action="store_true")
    args = parser.parse_args()
    extract_codebook(
        args.checkpoint,
        args.output,
        device=args.device,
        include_per_group=args.include_per_group,
    )


if __name__ == "__main__":
    main()
