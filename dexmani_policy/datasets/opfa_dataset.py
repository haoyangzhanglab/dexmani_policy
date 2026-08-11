"""OPFADataset — loads pre-computed GaLR hand latents alongside standard PCDataset data.

Each sample contains:

  - ``obs["point_cloud"]``: scene PC (standard PCDataset)
  - ``obs["joint_state"]``: native arm+hand joint state (standard PCDataset)
  - ``obs["hand_latent"]``: pre-computed 1024-d GaLR state latent per timestep
  - ``action``: native 19-d action (standard PCDataset)
  - ``action_latent``: pre-computed 1024-d GaLR action latent per timestep

The latents are pre-computed offline via ``dexmani_policy.agents.opfa.preprocess`` and
stored as a ``.pt`` file with keys ``"obs_latents"`` and ``"action_latents"``
— each a list of ``(T_ep, 1024)`` tensors, one per episode.

Normalizer
----------

The normalizer is fitted on **1031-d latent-space actions** ``[arm_raw(7), action_latent_scaled(1024)]``,
matching the official OPFA approach.  This ensures per-dimension ``(scale, offset)``
normalisation that automatically balances the arm (7-d, radian-scale) and hand-latent
(1024-d, L2-normalised) components.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from dexmani_policy.common.normalizer import SingleFieldLinearNormalizer
from dexmani_policy.datasets.pc_dataset import PCDataset


class OPFADataset(PCDataset):
    """PCDataset with pre-computed GaLR hand latents.

    Args:
        latent_path: Path to ``.pt`` file containing:
          - ``obs_latents``: ``List[Tensor(T, 1024)]`` per-episode state latents.
          - ``action_latents``: ``List[Tensor(T, 1024)]`` per-episode action latents.
        **kwargs: forwarded to ``PCDataset``.
    """

    def __init__(self, latent_path: str, **kwargs):
        super().__init__(**kwargs)
        latent_data = torch.load(latent_path, map_location="cpu", weights_only=True)
        self._obs_latents: list[torch.Tensor] = latent_data["obs_latents"]
        self._action_latents: list[torch.Tensor] = latent_data["action_latents"]

        n_latent_eps = len(self._obs_latents)
        if n_latent_eps != len(self._action_latents):
            raise ValueError(
                f"Latent count mismatch: obs={n_latent_eps} vs action={len(self._action_latents)}"
            )

        # Filter sampler to only include episodes that have pre-computed latents.
        # The latent file may cover fewer episodes than the full Zarr dataset
        # (e.g. when using --max_episodes for quick testing).
        ep_ends = self.sampler.replay_buffer.episode_ends
        buffer_starts = self.sampler.indices[:, 0]
        ep_indices = np.searchsorted(ep_ends, buffer_starts, side="right")

        if n_latent_eps < len(ep_ends):
            valid_mask = ep_indices < n_latent_eps
            n_before = len(self.sampler.indices)
            self.sampler.indices = self.sampler.indices[valid_mask]
            print(
                f"OPFADataset: filtered samples {n_before} → {len(self.sampler.indices)} "
                f"(latents cover {n_latent_eps}/{len(ep_ends)} episodes)"
            )

        if len(self.sampler.indices) == 0:
            raise ValueError(
                f"No samples overlap between latent file ({n_latent_eps} episodes) "
                f"and dataset ({len(ep_ends)} episodes). "
                f"Re-run 'python -m dexmani_policy.agents.opfa.preprocess' "
                f"without --max_episodes, or on the full dataset."
            )

        # Build sample index → episode index mapping (after potential filter)
        self._sample_ep_idx = self._build_sample_ep_mapping()

    # -----------------------------------------------------------------
    # Sample → episode mapping
    # -----------------------------------------------------------------

    def _build_sample_ep_mapping(self) -> np.ndarray:
        """Map each flat sample index to its episode index."""
        ep_ends = self.sampler.replay_buffer.episode_ends
        indices = self.sampler.indices  # (n_samples, 4)
        mapping = np.zeros(len(indices), dtype=np.int64)
        for i in range(len(indices)):
            buffer_start = int(indices[i, 0])
            ep_idx = np.searchsorted(ep_ends, buffer_start, side="right")
            mapping[i] = ep_idx
        return mapping

    # -----------------------------------------------------------------
    # Normalizer — fit on 1031-d latent actions (matching official OPFA)
    # -----------------------------------------------------------------

    def get_normalizer(self, mode="limits"):
        """Fit normalizer on 1031-d ``[arm_raw(7), action_latent_scaled(1024)]``.

        The official OPFA fits its normaliser directly on the concatenated
        latent-space action so that every dimension gets an independent
        ``(scale, offset)`` — automatically balancing arm (radian-scale,
        few dims) and hand-latent (L2-normalised, many dims) in the
        denoising loss.
        """
        # 1. Base (PCDataset) normalizer: joint_state + point_cloud + action(19-d)
        normalizer = super().get_normalizer(mode=mode)

        # 2. Build 1031-d data matching the target format in compute_loss:
        #    [arm_raw(7), action_latent * sqrt(1024)]
        joint_state, action_19 = self._get_faas_normalizer_data()
        ep_ends = self.replay_buffer.episode_ends
        n_eps = min(len(self._action_latents), len(ep_ends))
        scale = math.sqrt(1024)

        parts = []
        for ep_idx in range(n_eps):
            start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
            end = int(ep_ends[ep_idx])
            ep_len = end - start
            ep_arm_raw = torch.from_numpy(action_19[start:end, :7]).float()
            ep_lat = self._action_latents[ep_idx][:ep_len] * scale
            parts.append(torch.cat([ep_arm_raw, ep_lat], dim=-1))

        target_1031 = torch.cat(parts, dim=0)  # (N, 1031)

        # 3. Replace the 19-d action normalizer with the 1031-d version
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit_params(
            target_1031, last_n_dims=1, mode=mode,
        )

        # 4. Replace point_cloud normalizer with identity (matching official OPFA).
        # Official IsaacDataset.get_normalizer() sets point_cloud to identity
        # (no normalization), passing raw XYZ coordinates to PointNet.
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()

        # 5. Fit normalizer for hand_latent (obs).
        # Without this, L2-normalised latents (~0.03 per dim) pass through
        # the LinearNormalizer unchanged, starving the hand_mlp of gradient
        # signal during early training.  Official OPFA normalises agent_pos
        # (the equivalent field) to [-1, 1].
        hl_parts = []
        for ep_idx in range(n_eps):
            start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
            end = int(ep_ends[ep_idx])
            ep_len = end - start
            hl_parts.append(self._obs_latents[ep_idx][:ep_len])
        hand_latent_data = torch.cat(hl_parts, dim=0)  # (total_frames, 1024)
        normalizer["hand_latent"] = SingleFieldLinearNormalizer.create_fit_params(
            hand_latent_data, last_n_dims=1, mode=mode,
        )

        return normalizer

    # -----------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Resolve episode and frame range
        ep_idx = int(self._sample_ep_idx[idx])
        buffer_start, buffer_end, sample_start, sample_end = [
            int(x) for x in self.sampler.indices[idx]
        ]

        ep_ends = self.sampler.replay_buffer.episode_ends
        ep_start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
        local_start = buffer_start - ep_start
        local_end = buffer_end - ep_start

        # Slice latents from the episode
        obs_lat_raw = self._obs_latents[ep_idx][local_start:local_end]
        act_lat_raw = self._action_latents[ep_idx][local_start:local_end]

        seq_len = self.sampler.sequence_length
        n_real = local_end - local_start

        # Always pad to sequence_length (some samples have fewer real frames
        # due to boundary clipping — mirror the nearest frame like the replay buffer).
        if n_real < seq_len:
            obs_lat = torch.zeros(seq_len, 1024, dtype=obs_lat_raw.dtype)
            act_lat = torch.zeros(seq_len, 1024, dtype=act_lat_raw.dtype)
            obs_lat[sample_start:sample_end] = obs_lat_raw
            act_lat[sample_start:sample_end] = act_lat_raw
            if sample_start > 0:
                obs_lat[:sample_start] = obs_lat_raw[0]
                act_lat[:sample_start] = act_lat_raw[0]
            if sample_end < seq_len:
                obs_lat[sample_end:] = obs_lat_raw[-1]
                act_lat[sample_end:] = act_lat_raw[-1]
        else:
            obs_lat = obs_lat_raw
            act_lat = act_lat_raw

        data["obs"]["hand_latent"] = obs_lat.float()
        data["action_latent"] = act_lat.float()

        return data

    def get_validation_dataset(self):
        """Return validation split sharing latent data (shallow copy)."""
        val = super().get_validation_dataset()
        if val is None:
            return None
        # Share latent arrays (read-only, no mutation)
        val._obs_latents = self._obs_latents
        val._action_latents = self._action_latents
        # Filter validation sampler too (same latent coverage constraint as train)
        n_latent_eps = len(self._obs_latents)
        ep_ends = self.sampler.replay_buffer.episode_ends
        if n_latent_eps < len(ep_ends):
            buffer_starts = val.sampler.indices[:, 0]
            ep_indices = np.searchsorted(ep_ends, buffer_starts, side="right")
            valid_mask = ep_indices < n_latent_eps
            val.sampler.indices = val.sampler.indices[valid_mask]
        if len(val.sampler.indices) == 0:
            # No validation episodes in latent coverage — return None
            return None
        # Validation dataset uses its own sampler, must rebuild episode mapping
        val._sample_ep_idx = val._build_sample_ep_mapping()
        return val
