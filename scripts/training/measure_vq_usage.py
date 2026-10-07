"""Measure dataset-level VQ diagnostics for the exact runtime codebook."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.vq_hand import CodebookManager, VQVAEHand
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.training.resume import validate_data_identity
from scripts.training.train_vq_hand import build_policy_dataset, policy_hand_rows


def _args_dict(checkpoint: dict) -> dict:
    args = checkpoint["args"]
    if type(args) is not dict:
        raise ValueError("VQ checkpoint args must be a mapping")
    return args


def _normalizer_from_checkpoint(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = checkpoint["normalizer_state_dict"]
    return (
        state["params_dict.hand.scale"].float(),
        state["params_dict.hand.offset"].float(),
    )


def _normalize(
    data: np.ndarray, scale: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
    return tensor * scale.cpu() + offset.cpu()


def measure(
    checkpoint_path: str,
    zarr_path: str,
    *,
    codebook_path: str | None = None,
    action_key: str | None = None,
    tcp_dim: int | None = None,
    sample_size: int = 5000,
    seed: int = 0,
    split: str = "train",
    chunk_size: int = 4096,
) -> dict:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = _args_dict(checkpoint)
    action_key = action_key or args["action_key"]
    tcp_dim = int(tcp_dim if tcp_dim is not None else args["tcp_dim"])

    model = VQVAEHand.from_checkpoint(checkpoint, map_location="cpu").eval()
    metadata = checkpoint["split_metadata"]
    if "policy_config" in metadata:
        register_resolvers()
        cfg = OmegaConf.create(metadata["policy_config"])
        # The resolved data recipe is retained even if top-level interpolations change.
        cfg.dataset = OmegaConf.create(copy.deepcopy(metadata["resolved_dataset"]))
        if Path(zarr_path).resolve() != Path(cfg.dataset.zarr_path).resolve():
            raise ValueError("Policy-aligned usage requires the saved dataset path")
        if action_key != cfg.action_key or tcp_dim != int(cfg.agent.tcp_dim):
            raise ValueError("Usage action layout must match the saved Policy")
        from dexmani_policy.datasets.base_dataset import restored_time_filter_kwargs
        for key, value in restored_time_filter_kwargs(metadata.get("data_recipe")).items():
            cfg.dataset[key] = value
        saved_split = metadata.get("data_recipe", {}).get("split_manifest")
        if saved_split is not None or cfg.dataset.get("split_manifest") is not None:
            required = {"content", "sha256", "actual_train_ids", "train_mask", "val_mask"}
            if not isinstance(saved_split, dict) or not required <= saved_split.keys():
                raise ValueError("Policy-aligned usage requires complete saved split_manifest evidence")
            cfg.dataset.saved_split = copy.deepcopy(saved_split)
        dataset = build_policy_dataset(cfg)
        validate_data_identity({"revision": metadata.get("data_revision")},
                               {"revision": dataset.data_revision})
        validation = dataset.get_validation_dataset()
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if "episode_ends" in metadata and dataset.replay_buffer.episode_ends.tolist() != metadata["episode_ends"]:
            raise ValueError("Dataset episode boundaries changed since VQ preparation")
        for name, subset, mask in (("train", dataset, dataset.train_mask),
                                   ("val", validation, dataset.val_mask)):
            rows = subset.sampler.action_source_rows.tolist() if subset is not None else []
            if rows != metadata[name + "_source_rows"]:
                raise ValueError("Dataset qualified source rows changed since VQ preparation")
            ids = metadata.get(name + "_episode_ids")
            if ids is not None and np.flatnonzero(mask).tolist() != ids:
                raise ValueError("Dataset episode IDs changed since VQ preparation")
        dataset = validation if split == "validation" else dataset
        hand = policy_hand_rows(dataset, tcp_dim)
    else:
        buffer = ReplayBuffer.open(zarr_path, keys=[action_key])
        actions = buffer.read(action_key, slice(None))
        hand = actions[:, tcp_dim:]
    if not len(hand):
        raise ValueError("No qualified VQ usage samples in the selected split")
    if hand.shape[1] != model.hand_dim:
        raise ValueError(
            f"Data hand_dim={hand.shape[1]} does not match checkpoint {model.hand_dim}"
        )

    scale, offset = _normalizer_from_checkpoint(checkpoint)
    hand_norm = _normalize(hand, scale, offset)

    if codebook_path:
        manager = CodebookManager(
            hand_dim=model.hand_dim,
            num_groups=model.num_groups,
            codebook_size=model.codebook_size,
        )
        manager.load(codebook_path)
        if manager.has_hand_normalizer:
            torch.testing.assert_close(manager.hand_normalizer_scale, scale)
            torch.testing.assert_close(manager.hand_normalizer_offset, offset)
    else:
        manager = CodebookManager.extract_from_vqvae(model)
        manager.set_hand_normalizer(scale, offset)
        manager.reindex_by_pca(model)

    # Preserve squared-difference arithmetic and first-index argmin ties.
    count = manager.num_codes
    nearest_counts = torch.zeros(count, dtype=torch.long)
    tuple_counts = torch.zeros(count, dtype=torch.long)
    nearest_l2 = torch.empty(len(hand_norm))  # exact quantiles require O(N) scalars
    prototypes_norm = manager._from_raw(manager.sorted_hand_poses.cpu())
    multipliers = torch.tensor(
        [model.codebook_size**p for p in reversed(range(model.num_groups))]
    )
    with torch.no_grad():
        for start in range(0, len(hand_norm), chunk_size):
            batch = hand_norm[start:start + chunk_size]
            distances = (batch[:, None, :] - prototypes_norm[None, :, :]).square().sum(-1)
            values, _ = distances.min(-1)
            continuous = manager.hand_pose_to_continuous_index(batch)
            runtime_ids = torch.floor(
                ((continuous.squeeze(-1) + 1) * 0.5 * (count - 1)).clamp(0, count - 1) + 0.5
            ).long()
            nearest_counts += torch.bincount(runtime_ids, minlength=count)
            nearest_l2[start:start + len(batch)] = values.sqrt()
            ids = (model.encode_to_index(batch).long() * multipliers).sum(-1)
            tuple_counts += torch.bincount(ids, minlength=count)
    nearest_prob = nearest_counts.float() / nearest_counts.sum().clamp_min(1)

    generator = torch.Generator().manual_seed(seed)
    subset_size = min(sample_size, len(hand_norm))
    subset = hand_norm[
        torch.randperm(len(hand_norm), generator=generator)[:subset_size]
    ]
    with torch.no_grad():
        enc, vq, _, mse = model(subset)

    probability_nonzero = nearest_prob[nearest_prob > 0]
    entropy = -(probability_nonzero * probability_nonzero.log()).sum()
    normalized_entropy = entropy / np.log(max(count, 2))

    return {
        "num_codes": count,
        "nn_prototype_used": int((nearest_counts > 0).sum()),
        "nn_prototype_used_1pct": int((nearest_prob > 0.01).sum()),
        "nn_normalized_entropy": float(normalized_entropy),
        "nn_counts": nearest_counts.tolist(),
        "encoder_tuple_used": int((tuple_counts > 0).sum()),
        "encoder_tuple_counts": tuple_counts.tolist(),
        "recon_weighted_l1": float(enc),
        "commitment_mse": float(vq),
        "recon_mse": float(mse),
        "nn_l2_mean": float(nearest_l2.mean()),
        "nn_l2_p95": float(torch.quantile(nearest_l2, 0.95)),
        "nn_l2_p99": float(torch.quantile(nearest_l2, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument(
        "--codebook",
        default=None,
        help="Exact .npz used by the policy. Strongly recommended.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Policy-aligned checkpoints use saved qualified rows; standalone checkpoints use all rows",
    )
    parser.add_argument("--action_key", default=None)
    parser.add_argument("--tcp_dim", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    result = measure(
        args.checkpoint,
        args.zarr,
        codebook_path=args.codebook,
        action_key=args.action_key,
        tcp_dim=args.tcp_dim,
        split=args.split,
        chunk_size=args.chunk_size,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
