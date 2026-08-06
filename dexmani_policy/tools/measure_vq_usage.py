"""Measure dataset-level VQ diagnostics for the exact runtime codebook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexmani_policy.agents.vq_hand import CodebookManager, VQVAEHand
from dexmani_policy.datasets.replay_buffer import ReplayBuffer


def _args_dict(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return vars(args) if hasattr(args, "__dict__") else dict(args)


def _normalizer_from_checkpoint(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = checkpoint.get("normalizer_state_dict")
    if state is not None:
        return (
            state["params_dict.hand.scale"].float(),
            state["params_dict.hand.offset"].float(),
        )
    params = checkpoint.get("normalizer_params")
    if params is not None:
        return params["hand"]["scale"].float(), params["hand"]["offset"].float()
    raise ValueError("Checkpoint has no hand normalizer")


def _normalize(data: np.ndarray, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
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
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = _args_dict(checkpoint)
    action_key = action_key or args.get("action_key", "action_ee")
    tcp_dim = int(tcp_dim if tcp_dim is not None else args.get("tcp_dim", 9))

    model = VQVAEHand.from_checkpoint(checkpoint, map_location="cpu").eval()
    buffer = ReplayBuffer.copy_from_path(zarr_path, keys=[action_key])
    actions = np.asarray(buffer[action_key])
    hand = actions[:, tcp_dim:]
    if hand.shape[1] != model.hand_dim:
        raise ValueError(f"Data hand_dim={hand.shape[1]} does not match checkpoint {model.hand_dim}")

    scale, offset = _normalizer_from_checkpoint(checkpoint)
    hand_norm = _normalize(hand, scale, offset)

    manager = CodebookManager(
        hand_dim=model.hand_dim,
        num_groups=model.num_groups,
        codebook_size=model.codebook_size,
    )
    if codebook_path:
        manager.load(codebook_path)
        if manager.has_hand_normalizer:
            torch.testing.assert_close(manager.hand_normalizer_scale, scale)
            torch.testing.assert_close(manager.hand_normalizer_offset, offset)
    else:
        manager = CodebookManager.extract_from_vqvae(model)
        manager.set_hand_normalizer(scale, offset)
        manager.reindex_by_pca(model)

    # Nearest decoded-prototype usage: this is the label distribution actually
    # consumed by DQ-RISE policy training.
    continuous = manager.hand_pose_to_continuous_index(hand_norm)
    count = manager.num_codes
    nearest_ids = torch.floor(
        ((continuous.squeeze(-1) + 1.0) * 0.5 * (count - 1)).clamp(0, count - 1) + 0.5
    ).long()
    nearest_counts = torch.bincount(nearest_ids, minlength=count)
    nearest_prob = nearest_counts.float() / nearest_counts.sum().clamp_min(1)

    # Encoder tuple usage is a different diagnostic and is reported separately.
    tuple_indices = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(hand_norm), batch_size):
            tuple_indices.append(model.encode_to_index(hand_norm[start : start + batch_size]))
    tuple_indices = torch.cat(tuple_indices, dim=0)
    multipliers = torch.tensor(
        [model.codebook_size**power for power in reversed(range(model.num_groups))],
        dtype=torch.long,
    )
    tuple_ids = (tuple_indices.long() * multipliers).sum(dim=-1)
    tuple_counts = torch.bincount(tuple_ids, minlength=count)

    generator = torch.Generator().manual_seed(seed)
    subset_size = min(sample_size, len(hand_norm))
    subset = hand_norm[torch.randperm(len(hand_norm), generator=generator)[:subset_size]]
    with torch.no_grad():
        enc, vq, _, mse = model(subset)

    prototypes_norm = manager._from_raw(manager.sorted_hand_poses.cpu())
    diff = hand_norm[:, None, :] - prototypes_norm[None, :, :]
    nearest_l2 = diff.square().sum(-1).min(-1).values.sqrt()

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
    parser.add_argument("--action_key", default=None)
    parser.add_argument("--tcp_dim", type=int, default=None)
    args = parser.parse_args()
    result = measure(
        args.checkpoint,
        args.zarr,
        codebook_path=args.codebook,
        action_key=args.action_key,
        tcp_dim=args.tcp_dim,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
