"""Export a PCA-ordered runtime codebook from a trained VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from dexmani_policy.agents.vq_hand import CodebookManager, VQVAEHand
from dexmani_policy.utils.atomic import atomic_path


def load_checkpoint_snapshot(path: str | Path):
    """Hash and deserialize one open-file snapshot, bounded even for large checkpoints."""
    digest = hashlib.sha256()
    with open(path, "rb") as file, tempfile.TemporaryFile() as snapshot:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
            snapshot.write(chunk)
        snapshot.seek(0)
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
    return checkpoint, digest.hexdigest()


def _extract_hand_normalizer(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = checkpoint["normalizer_state_dict"]
    return (
        state["params_dict.hand.scale"].detach().cpu(),
        state["params_dict.hand.offset"].detach().cpu(),
    )


def build_runtime_codebook(checkpoint, source_hash, checkpoint_path, *, device):
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
            "source_checkpoint_sha256": source_hash,
            "source_epoch": int(checkpoint["epoch"]),
            "normalizer_sha256": hashlib.sha256(
                b"".join(t.contiguous().numpy().tobytes() for t in normalizer)
            ).hexdigest(),
            "checkpoint_metrics": checkpoint["metrics"],
            "split_metadata": checkpoint["split_metadata"],
        }
    )

    manager.reindex_by_pca(vqvae)
    from dexmani_policy.agents.vq_hand.codebook_manager import compact_artifact_metadata
    manager.artifact_metadata = compact_artifact_metadata(manager.artifact_metadata)
    return manager


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
    checkpoint, source_hash = load_checkpoint_snapshot(checkpoint_path)
    manager = build_runtime_codebook(checkpoint, source_hash, checkpoint_path, device=device)
    poses = manager.sorted_hand_poses
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


@torch.no_grad()
def deployment_diagnostics(manager, splits, *, chunk_size=4096):
    """Measure the actual nearest-prototype/quantization chain, never VQ encoding.

    Inputs are policy-normalized hand rows. Nearest neighbors remain in the
    manager's affine-raw space; physical errors use its saved action affine.
    """
    import numpy as np
    from dexmani_policy.utils.validation import positive_int
    positive_int(chunk_size, 'chunk_size')
    device = manager.sorted_hand_poses.device
    width = manager.hand_dim
    chunk_size = min(chunk_size, max(1, (16 * 1024 * 1024) // (manager.num_codes * width * 4)))
    report = {'distance_space': 'CodebookManager affine_raw squared Euclidean',
              'source_checkpoint_sha256': manager.artifact_metadata.get('source_checkpoint_sha256'),
              'source_epoch': manager.artifact_metadata.get('source_epoch'), 'splits': {}}
    scale = manager.hand_normalizer_scale.cpu().double() if manager.has_hand_normalizer else None
    for name, rows in splits.items():
        count = len(rows)
        if count == 0:
            report['splits'][name] = {'count': 0, 'status': 'NOT VERIFIED: no held-out rows'}
            continue
        spaces = ['normalized'] + (['physical'] if scale is not None else [])
        sums = {space: [torch.zeros(width, dtype=torch.float64), torch.zeros(width, dtype=torch.float64)]
                for space in spaces}
        per_sample = {space: np.empty(count, dtype=np.float32) for space in spaces}
        utilization = torch.zeros(manager.num_codes, dtype=torch.int64)
        for start in range(0, count, chunk_size):
            values = torch.as_tensor(rows[start:start + chunk_size], dtype=torch.float32, device=device)
            if values.ndim != 2 or values.shape[-1] != width or not torch.isfinite(values).all():
                raise ValueError('Codebook diagnostics require finite normalized (N, hand_dim) rows')
            index = manager.hand_pose_to_continuous_index(values)
            restored, discrete = manager.continuous_index_to_hand_pose(index)
            restored = restored.reshape_as(values)
            utilization += torch.bincount(discrete.flatten().cpu(), minlength=manager.num_codes)
            error = (restored - values).cpu().double()
            for space in spaces:
                delta = error if space == 'normalized' else error / scale
                sums[space][0] += delta.abs().sum(0)
                sums[space][1] += delta.square().sum(0)
                per_sample[space][start:start + len(values)] = delta.square().mean(-1).sqrt().numpy()
        result = {'count': count, 'prototype_counts': utilization.tolist(),
                  'used_prototypes': int((utilization > 0).sum())}
        for space in spaces:
            absolute, squared = sums[space]
            result[space] = {'mae': float(absolute.sum() / (count * width)),
                             'rmse': float((squared.sum() / (count * width)).sqrt()),
                             'joint_mae': (absolute / count).tolist(),
                             'joint_rmse': (squared / count).sqrt().tolist(),
                             'sample_rmse_quantiles': dict(zip(('p50', 'p90', 'p95', 'p99', 'max'),
                                np.quantile(per_sample[space], [.5, .9, .95, .99, 1]).tolist()))}
        if scale is None:
            result['physical'] = {'status': 'NOT VERIFIED: missing affine normalizer'}
        report['splits'][name] = result
    adjacent = torch.diff(manager.sorted_hand_poses, dim=0).norm(dim=-1)
    report['pca_adjacent_affine_raw_distance'] = {'mean': float(adjacent.mean()) if len(adjacent) else 0.,
                                                'max': float(adjacent.max()) if len(adjacent) else 0.}
    return report


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
