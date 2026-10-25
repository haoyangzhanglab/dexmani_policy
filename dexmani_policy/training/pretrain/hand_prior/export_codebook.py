"""Export PCA-ordered hand states from a DQ-RISE VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, validate_dq_normalization
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.runtime import atomic_path


@torch.no_grad()
def build_hand_prototypes(model: HandVQVAE) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the runtime table; retain unclamped poses for export diagnostics."""
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
    return decoded.clamp(-1, 1), decoded


@torch.no_grad()
def build_runtime_codebook(checkpoint, checkpoint_path, *, device):
    split = checkpoint["split_metadata"]
    validate_dq_normalization(split["normalization_spec"])
    model = HandVQVAE.from_checkpoint(checkpoint, device=device).eval()
    codebook = HandStateCodebook(model.hand_dim, model.num_groups, model.codebook_size)
    poses, decoded = build_hand_prototypes(model)
    poses, decoded = poses.cpu(), decoded.cpu()
    weights = model.quantizer.layer_weights.softmax(0)
    pca = PCA(n_components=1)
    projection = pca.fit_transform(poses.numpy())[:, 0]
    permutation = torch.from_numpy(np.argsort(projection))
    state = checkpoint["normalizer_state_dict"]
    scale = state["params_dict.hand.scale"].detach().float().cpu()
    offset = state["params_dict.hand.offset"].detach().float().cpu()
    codebook.set_buffers(
        {
            "sorted_hand_poses": poses[permutation],
            "pca_permutation": permutation,
            "layer_weights": weights.cpu(),
            "hand_normalizer_scale": scale,
            "hand_normalizer_offset": offset,
        }
    )
    variance_ratio = float(np.nan_to_num(pca.explained_variance_ratio_[0], nan=0.0))
    diagnostics = {
        "decoder_min_before_clamp": float(decoded.min()),
        "decoder_max_before_clamp": float(decoded.max()),
        "outside_fraction": float((decoded.abs() > 1).double().mean()),
        "max_violation": float((decoded.abs() - 1).clamp_min(0).max()),
        "pca_explained_variance_ratio": variance_ratio,
    }
    codebook.artifact_metadata = {
        "model_config": checkpoint["model_config"],
        "hand_data": split["hand_data"],
        "split_manifest": split["split_manifest"],
        "normalization_spec": split["normalization_spec"],
        "ordering": "decoded_prototype_pca1_ascending",
        "selection": {
            "selection_step": int(checkpoint["epoch"]),
            "unit": "epoch",
            "split": split["selection_split"],
            "metric": split["selection_metric"],
            "value": checkpoint["metrics"][split["selection_metric"]],
        },
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "export_diagnostics": diagnostics,
    }
    if variance_ratio < 0.3:
        warnings.warn(
            f"Hand prototypes have weak PC1 ordering: variance ratio={variance_ratio:.4f}",
            RuntimeWarning,
        )
    if diagnostics["outside_fraction"]:
        warnings.warn(f"Decoded hand states were clamped to [-1, 1]: {diagnostics}", RuntimeWarning)
    return codebook


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


@torch.no_grad()
def deployment_diagnostics(codebook, splits, *, chunk_size=4096):
    """Measure actual lookup errors in normalized and physical hand coordinates."""
    positive_int(chunk_size, "chunk_size")
    device, width = codebook.sorted_hand_poses.device, codebook.hand_dim
    chunk_size = min(chunk_size, max(1, (16 * 1024 * 1024) // (codebook.num_codes * width * 4)))
    report = {
        "distance_space": "HandStateCodebook normalized squared Euclidean",
        "source_checkpoint": codebook.artifact_metadata["source_checkpoint"],
        "selection": codebook.artifact_metadata["selection"],
        "splits": {},
    }
    scale = codebook.hand_normalizer_scale.cpu().double()
    spaces = ("normalized", "physical")
    for name, rows in splits.items():
        count = len(rows)
        if not count:
            report["splits"][name] = {"count": 0, "status": "NOT VERIFIED: no held-out rows"}
            continue
        sums = {space: torch.zeros(2, width, dtype=torch.float64) for space in spaces}
        per_sample = {space: np.empty(count, dtype=np.float32) for space in spaces}
        utilization = torch.zeros(codebook.num_codes, dtype=torch.int64)
        for start in range(0, count, chunk_size):
            values = torch.as_tensor(
                rows[start : start + chunk_size], dtype=torch.float32, device=device
            )
            if values.ndim != 2:
                raise ValueError("Codebook diagnostics require (N, hand_dim) rows")
            continuous, _ = codebook.encode(values)
            restored, indices = codebook.decode(continuous.squeeze(-1))
            utilization += torch.bincount(indices.cpu(), minlength=codebook.num_codes)
            error = (restored - values).cpu().double()
            for space in spaces:
                delta = error if space == "normalized" else error / scale
                sums[space][0] += delta.abs().sum(0)
                sums[space][1] += delta.square().sum(0)
                per_sample[space][start : start + len(values)] = (
                    delta.square().mean(-1).sqrt().numpy()
                )
        result = {
            "count": count,
            "lookup_mse": float(sums["normalized"][1].sum() / (count * width)),
            "prototype_counts": utilization.tolist(),
            "used_prototypes": int((utilization > 0).sum()),
        }
        for space in spaces:
            absolute, squared = sums[space]
            result[space] = {
                "mae": float(absolute.sum() / (count * width)),
                "rmse": float((squared.sum() / (count * width)).sqrt()),
                "joint_mae": (absolute / count).tolist(),
                "joint_rmse": (squared / count).sqrt().tolist(),
                "sample_rmse_quantiles": dict(
                    zip(
                        ("p50", "p90", "p95", "p99", "max"),
                        np.quantile(per_sample[space], [0.5, 0.9, 0.95, 0.99, 1]).tolist(),
                    )
                ),
            }
        report["splits"][name] = result
    adjacent = torch.diff(codebook.sorted_hand_poses, dim=0).norm(dim=-1)
    report["pca_adjacent_normalized_distance"] = {
        "mean": float(adjacent.mean()) if len(adjacent) else 0.0,
        "max": float(adjacent.max()) if len(adjacent) else 0.0,
    }
    return report


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
