"""Hand action coordinates, codec compatibility and pretraining splits."""

import math

HAND_ACTION_SEMANTICS = {
    "action_space": "joint",
    "action_coordinates": "absolute_joint_command",
    "joint_order": ["arm:7", "thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
    "units": "radians",
    "obs_alignment": "obs[t]_before_action[t]",
    "reference": "measured_query_state",
    "window": "query_future_full_horizon",
}


def validate_codec_data(codec_data, policy_data):
    """Allow independent corpora while preserving action semantics and shared holdouts."""
    left, right = codec_data["hand_data"]["contract"], policy_data["hand_data"]["contract"]
    if left.keys() != right.keys() or any(left[key] != right[key] for key in left if key != "dt"):
        raise ValueError("Codec/policy hand action contract mismatch")
    if not math.isclose(left["dt"], right["dt"], rel_tol=1e-6, abs_tol=0):
        raise ValueError("Codec/policy dt mismatch; no automatic resampling")

    source, target = codec_data["split_manifest"], policy_data["split_manifest"]
    source_identity = source["source_id"] or codec_data["zarr_path"]
    target_identity = target["source_id"] or policy_data["zarr_path"]
    if source_identity != target_identity:
        return

    source_ids, target_ids = source["episode_ids"], target["episode_ids"]
    if source_ids is None or target_ids is None:
        if source["episode_ends"] != target["episode_ends"] or source_ids != target_ids:
            raise ValueError("Same-source subsets require stable episode_ids")
        source_ids = target_ids = range(len(source["episode_ends"]))

    source_val = {source_ids[i] for i in source["val_indices"]}
    target_val = {target_ids[i] for i in target["val_indices"]}
    shared = set(source_ids).intersection(target_ids)
    # Shared episodes must keep the same holdout membership.
    if (source_val & shared) != (target_val & shared):
        raise ValueError("Same-source codec/policy split mismatch or validation leakage")


def validate_pretraining_split(manifest, pretrained_split):
    if manifest != pretrained_split:
        raise ValueError("Pretraining/policy split mismatch")


def validate_hand_coordinates(attrs):
    order = HAND_ACTION_SEMANTICS["joint_order"][1:]
    if (
        attrs.get("hand_joint_order", order) != order
        or attrs.get("hand_units", "radians") != "radians"
    ):
        raise ValueError("Hand data must use XHand thumb/index/middle/ring/pinky order in radians")
