"""Full future hand chunks, independent of policy observations and visual validity."""

import math

import numpy as np
from omegaconf import OmegaConf

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset


def describe_hand_chunks(dataset, data_semantics, source_id):
    """Record stored semantics separately from explicit configuration declarations."""
    semantics = (
        OmegaConf.to_container(data_semantics, resolve=True)
        if OmegaConf.is_config(data_semantics) else dict(data_semantics)
    )
    if semantics != HAND_ACTION_SEMANTICS:
        raise ValueError("Hand chunks require the declared arm7/XHand12 absolute-command semantics")
    buffer = dataset.replay_buffer
    for key in ("action", "joint_state"):
        if buffer[key].shape[1:] != (19,):
            raise ValueError(f"{key} must have shape [T,19]")
    attrs = buffer.attrs
    evidence = {}
    for key, expected in semantics.items():
        if key in attrs and attrs[key] != expected:
            raise ValueError(f"Stored {key} conflicts with declared hand action semantics")
        evidence[key] = "zarr" if key in attrs else "config_declaration"
    if attrs.get("hand_joint_order", semantics["joint_order"][1:]) != semantics["joint_order"][1:]:
        raise ValueError("Stored hand_joint_order differs from XHand order")
    if attrs.get("hand_units", "radians") != "radians":
        raise ValueError("Stored hand_units must be radians")
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Hand chunks require finite positive recorded dt")
    stored_source = attrs.get("source_id")
    for value in (source_id, stored_source):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("source_id must be a nonempty string")
    if source_id is not None and stored_source is not None and source_id != stored_source:
        raise ValueError("Declared source_id differs from Zarr source_id")
    source_id = source_id or stored_source or dataset.zarr_path
    episode_ids = attrs.get("episode_ids")
    if episode_ids is not None and (
        not isinstance(episode_ids, list) or len(episode_ids) != buffer.n_episodes
        or any(not isinstance(value, str) or not value for value in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError("episode_ids must be distinct nonempty strings matching episodes")
    return {
        "contract": dict(semantics, horizon=dataset.horizon, dt=float(dt)),
        "provenance": {
            "source_id": source_id,
            "zarr_path": dataset.zarr_path,
            "semantics_evidence": evidence,
            "episode_ids": episode_ids,
            "episode_ends": buffer.episode_ends.tolist(),
            "train_episodes": np.flatnonzero(dataset.train_mask).tolist(),
            "val_episodes": np.flatnonzero(dataset.val_mask).tolist(),
        },
    }


class HandChunkDataset(DemonstrationDataset):
    def __init__(
        self, zarr_path, *, data_semantics, horizon=32, seed=0, val_ratio=0.1,
        max_train_episodes=None, split_manifest=None, saved_split=None, source_id=None,
    ):
        if positive_int(horizon, "horizon") < 3:
            raise ValueError("HandVAE horizon must be at least 3 for second differences")
        super().__init__(
            zarr_path, sensor_modalities=["joint_state"], horizon=horizon, obs_horizon=1,
            seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes,
            split_manifest=split_manifest, saved_split=saved_split,
        )
        self.hand_data = describe_hand_chunks(self, data_semantics, source_id)

    def finite_rows(self, key, columns=None):
        return super().finite_rows(key, columns=slice(7, 19))

    def raw_hand_batch(self, indices):
        rows = self.sampler.source_rows(self.sampler.indices[indices])
        actions = self.replay_buffer.read("action", rows.reshape(-1), columns=slice(7, 19))
        q0 = self.replay_buffer.read("joint_state", rows[:, 0], columns=slice(7, 19))
        return actions.reshape(-1, self.horizon, 12), q0

    def __getitem__(self, idx):
        actions, q0 = self.raw_hand_batch(np.array([idx]))
        return {"action": actions[0], "reference_state": q0[0]}
