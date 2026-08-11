"""Pre-compute GaLR hand latents from trajectory data — OPFA Stage 2.

Runs a frozen GaLR encoder over each frame of a Zarr trajectory dataset,
producing per-episode ``(T, 1024)`` obs and action latent tensors.

Pipeline::

    Zarr trajectory
      → extract hand states  (joint_state[:, 7:] = 12-d hand joints)
      → extract hand actions  (action[:, 7:]      = 12-d hand target joints)
      → HandFKGenerator(joint_angles) → KPConv data dict
      → frozen GaLR encoder.encode(data_dict) → 1024-d latent
      → save as ``{output_path}`` (.pt, per-episode lists)

Speed: ~15-20 fps on GPU (vs ~2 fps on CPU).  An angle→latent cache eliminates
redundant FK+GaLR work for repeated joint configurations (common in trajectories),
pushing throughput to 30-50+ fps on typical data.

Usage (CLI)::

    python -m dexmani_policy.agents.opfa.preprocess \\
        --zarr_path robot_data/pour.zarr \\
        --galr_ckpt data/pretrained/galr/best.pt \\
        --output_path robot_data/pour_opfa_latents.pt

Programmatic::

    from dexmani_policy.agents.opfa.preprocess import preprocess_dataset
    preprocess_dataset(
        zarr_path="robot_data/pour.zarr",
        galr_ckpt="data/pretrained/galr/best.pt",
        output_path="robot_data/pour_opfa_latents.pt",
        device="cuda:0",
        max_episodes=50,
    )

Output:
  - ``{output_path}``: ``.pt`` file with keys:
    - ``obs_latents``: ``list[Tensor(T_ep, 1024)]`` — per-episode state latents.
    - ``action_latents``: ``list[Tensor(T_ep, 1024)]`` — per-episode action latents.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from dexmani_policy.agents.opfa.galr_autoencoder import GaLRAutoencoder, LatentCache, load_galr_encoder
from dexmani_policy.agents.opfa.hand_fk import HandFKGenerator

# =============================================================================
# Helpers
# =============================================================================


def extract_hand_joints(
    joint_state: np.ndarray,
    arm_dim: int = 7,
) -> np.ndarray:
    """Extract 12-d hand joint angles from full joint state.

    Args:
        joint_state: ``(T, 19)`` or ``(T, 39)`` (FAAS) — arm + hand.
        arm_dim: number of leading arm dimensions (7 for joint, 9 for EE).

    Returns:
        ``(T, 12)`` hand-only joint angles in VAE/DexMani order.
    """
    return joint_state[:, arm_dim:arm_dim + 12]


def encode_one_frame(
    hand_joints: np.ndarray,
    fk_gen: HandFKGenerator,
    galr: GaLRAutoencoder,
    device: str,
    cache: LatentCache | None = None,
) -> torch.Tensor:
    """Encode a single frame's hand joint angles → 1024-d latent.

    Args:
        hand_joints: ``(12,)`` numpy array of joint angles (VAE order).
        fk_gen: pre-initialised HandFKGenerator (on ``device``).
        galr: frozen GaLR autoencoder (on ``device``).
        device: torch device.
        cache: optional ``LatentCache`` for angle→latent lookup.

    Returns:
        ``(1024,)`` latent vector on CPU (mean-pooled from L2-normalized
        superpoint features).
    """
    # Cache lookup
    if cache is not None:
        hit = cache.get(hand_joints)
        if hit is not None:
            return hit

    angles_t = torch.from_numpy(hand_joints).float().to(device)
    with torch.no_grad():
        data_dict = fk_gen(angles_t, hand_type="xhand")
        latent = galr.encode(data_dict)  # (1024,)
    result = latent.cpu()

    if cache is not None:
        cache.put(hand_joints, result)

    return result


# =============================================================================
# Main pipeline
# =============================================================================


def preprocess_dataset(
    zarr_path: str,
    galr_ckpt: str,
    output_path: str | None = None,
    device: str = "cuda:0",
    max_episodes: int | None = None,
    log_interval: int = 50,
    cache_size: int = 10000,
) -> str:
    """Pre-compute GaLR hand latents for all episodes in a Zarr dataset.

    Args:
        zarr_path: Path to input Zarr trajectory file (e.g. ``robot_data/pour.zarr``).
        galr_ckpt: Path to trained GaLR checkpoint.
        output_path: Output ``.pt`` file path.  Default: ``{zarr_stem}_opfa_latents.pt``.
        device: Torch device (default ``"cuda:0"``).  GPU FK gives ~10× speedup.
        max_episodes: Limit to first N episodes (for quick testing).
        log_interval: Print progress every N episodes.
        cache_size: Max entries in angle→latent LRU cache (0 = disabled).

    Returns:
        Path to the saved ``.pt`` file.
    """
    # Resolve output path
    if output_path is None:
        stem = Path(zarr_path).stem
        output_path = str(Path(zarr_path).parent / f"{stem}_opfa_latents.pt")

    print("=" * 60)
    print("OPFA Stage 2: Pre-computing GaLR hand latents")
    print("=" * 60)
    print(f"  Zarr:      {zarr_path}")
    print(f"  GaLR ckpt: {galr_ckpt}")
    print(f"  Output:    {output_path}")
    print(f"  Device:    {device}")

    # ── Load Zarr ──────────────────────────────────────────────────
    print("\n[1/4] Loading Zarr dataset ...")
    t0 = time.time()

    root = zarr.open(zarr_path, mode="r")

    # Discover data keys
    data_group = root["data"]
    if "action" not in data_group:
        raise KeyError(f"No 'action' key found in {zarr_path}/data/")
    if "joint_state" not in data_group:
        raise KeyError(f"No 'joint_state' key found in {zarr_path}/data/")

    actions = data_group["action"]  # (total_frames, 19) or (total_frames, 21)
    joint_states = data_group["joint_state"]  # (total_frames, 19) or (total_frames, 39)

    # Read episode boundaries
    meta = root["meta"]
    episode_ends = meta["episode_ends"]  # (num_episodes,)
    num_episodes = len(episode_ends)
    num_frames = actions.shape[0]

    # Determine arm_dim from action dimension
    action_dim = actions.shape[1]
    joint_dim = joint_states.shape[1]

    if action_dim == 19:
        arm_dim = 7
    elif action_dim == 21:
        arm_dim = 9
    else:
        raise ValueError(f"Unexpected action dim {action_dim} — expected 19 or 21")
    # OPFA does NOT support FAAS — hand latents are computed from native joint
    # angles, not FAAS-encoded values.
    if joint_dim >= 39 or action_dim >= 39:
        raise NotImplementedError(
            f"OPFA preprocessing does not support FAAS data "
            f"(joint_dim={joint_dim}, action_dim={action_dim}). "
            f"FAAS hand encodings are not compatible with GaLR FK."
        )

    # EE mode (action_dim=21): joint_state is always 19-d (7 arm + 12 hand),
    # so extract_hand_joints must use a fixed arm_dim=7 for joint_state,
    # regardless of the action's arm_dim=9.
    js_arm_dim = 7  # joint_state arm dimension is always 7

    print(f"  Episodes:      {num_episodes}")
    print(f"  Total frames:  {num_frames}")
    print(f"  Action dim:    {action_dim} (arm_dim={arm_dim})")
    print(f"  Joint dim:     {joint_states.shape[1]}")

    if max_episodes is not None and max_episodes < num_episodes:
        num_episodes = max_episodes
        print(f"  ── Limited to first {num_episodes} episodes ──")

    print(f"  Zarr loaded in {time.time() - t0:.1f}s")

    # ── Load GaLR encoder + FK generator ───────────────────────────
    print("\n[2/4] Loading GaLR encoder + FK generator ...")
    t0 = time.time()

    galr = load_galr_encoder(galr_ckpt, device)
    fk_gen = HandFKGenerator().to(device)
    fk_gen.eval()
    for p in fk_gen.parameters():
        p.requires_grad_(False)

    print(f"  GaLR params:  {sum(p.numel() for p in galr.parameters()):,}")
    print(f"  FK device:    {device}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Latent cache for repeated joint configurations
    latent_cache = LatentCache(max_size=cache_size) if cache_size > 0 else None
    if latent_cache is not None:
        print(f"  Latent cache: enabled (max_size={cache_size})")

    # ── Encode episodes ────────────────────────────────────────────
    print(f"\n[3/4] Encoding {num_episodes} episodes ...")
    t0 = time.time()

    obs_latents: list[torch.Tensor] = []
    action_latents: list[torch.Tensor] = []

    for ep_idx in range(num_episodes):
        ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        ep_end = int(episode_ends[ep_idx])
        ep_len = ep_end - ep_start

        # Read episode data as numpy (single slice → efficient)
        js_ep = joint_states[ep_start:ep_end]  # (T, joint_dim)
        act_ep = actions[ep_start:ep_end]      # (T, action_dim)

        # Extract 12-d hand joint angles.
        # IMPORTANT: joint_state always has arm_dim=7, even in EE mode
        # (action_dim=21 means 9-d EE + 12-d hand, but joint_state is 7+12=19).
        hand_js = extract_hand_joints(js_ep, arm_dim=js_arm_dim)   # (T, 12)
        hand_act = extract_hand_joints(act_ep, arm_dim=arm_dim)    # (T, 12)

        # Encode frame-by-frame
        ep_obs_lat = torch.empty(ep_len, 1024)
        ep_act_lat = torch.empty(ep_len, 1024)

        for t in range(ep_len):
            ep_obs_lat[t] = encode_one_frame(hand_js[t], fk_gen, galr, device, latent_cache)
            ep_act_lat[t] = encode_one_frame(hand_act[t], fk_gen, galr, device, latent_cache)

        obs_latents.append(ep_obs_lat)
        action_latents.append(ep_act_lat)

        if (ep_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            fps = sum(len(ol) for ol in obs_latents) / elapsed
            print(f"  [{ep_idx + 1:4d}/{num_episodes}] "
                  f"last_ep_len={ep_len:4d}  "
                  f"elapsed={elapsed:.0f}s  "
                  f"rate={fps:.1f} fps")

    total_frames_processed = sum(len(ol) for ol in obs_latents)
    elapsed = time.time() - t0
    print(f"  Done: {total_frames_processed} frames in {elapsed:.0f}s "
          f"({total_frames_processed / elapsed:.1f} fps avg)")

    # ── Save ───────────────────────────────────────────────────────
    print(f"\n[4/4] Saving to {output_path} ...")
    t0 = time.time()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {"obs_latents": obs_latents, "action_latents": action_latents},
        output_path,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved {num_episodes} episodes ({total_frames_processed} frames)")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n✓ OPFA Stage 2 complete: {output_path}")
    print("  Ready for Stage 3: bash scripts/training/train.sh opfa")
    if latent_cache is not None:
        s = latent_cache.stats
        print(f"  Cache: {s['hits']} hits / {s['misses']} misses "
              f"({s['hit_rate']:.1%} hit rate, {s['size']} entries)")

    return output_path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for latent precomputation (OPFA Stage 2)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compute GaLR hand latents from trajectory data (OPFA Stage 2)",
    )
    parser.add_argument(
        "--zarr_path", type=str, required=True,
        help="Path to input Zarr trajectory file (e.g. robot_data/pour.zarr).",
    )
    parser.add_argument(
        "--galr_ckpt", type=str, required=True,
        help="Path to trained GaLR checkpoint (from Stage 1).",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output .pt file path.  Default: {zarr_stem}_opfa_latents.pt",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device (default: cuda:0). GPU FK gives ~10× speedup.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Limit to first N episodes (for quick testing).",
    )
    parser.add_argument(
        "--log_interval", type=int, default=50,
        help="Print progress every N episodes (default: 50).",
    )
    args = parser.parse_args()

    preprocess_dataset(
        zarr_path=args.zarr_path,
        galr_ckpt=args.galr_ckpt,
        output_path=args.output_path,
        device=args.device,
        max_episodes=args.max_episodes,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
