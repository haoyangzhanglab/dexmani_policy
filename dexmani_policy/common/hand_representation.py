"""Physical hand data semantics shared by datasets, policies and pretraining."""

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


def validate_hand_chunk_compatibility(codec_data, policy_data):
    """Fresh reuse checks coordinates and split leakage, not corpus size or eligibility."""
    left, right = codec_data["contract"], policy_data["contract"]
    if left.keys() != right.keys() or any(left[key] != right[key] for key in left if key != "dt"):
        raise ValueError("Codec/policy hand action contract mismatch")
    if not math.isclose(left["dt"], right["dt"], rel_tol=1e-6, abs_tol=0):
        raise ValueError("Codec/policy dt mismatch; no automatic resampling")
    source, target = codec_data["provenance"], policy_data["provenance"]
    if source["source_id"] != target["source_id"]:
        return
    source_ids, target_ids = source["episode_ids"], target["episode_ids"]
    if source_ids is None or target_ids is None:
        if source["episode_ends"] != target["episode_ends"] or source_ids != target_ids:
            raise ValueError("Same-source subsets require stable episode_ids")
        source_ids = target_ids = list(range(len(source["episode_ends"])))
    source_train = {source_ids[i] for i in source["train_episodes"]}
    source_val = {source_ids[i] for i in source["val_episodes"]}
    target_train = {target_ids[i] for i in target["train_episodes"]}
    target_val = {target_ids[i] for i in target["val_episodes"]}
    shared = set(source_ids) & set(target_ids)
    if source_train & target_val or source_val & target_train or (source_val & shared) != (target_val & shared):
        raise ValueError("Same-source codec/policy split mismatch or validation leakage")


def validate_pretraining_data(current, manifest, pretrained, pretrained_split):
    if manifest != pretrained_split:
        raise ValueError("Pretraining/policy split manifest mismatch")
    changed = [key for key in current if pretrained[key] != current[key]]
    if changed:
        raise ValueError(f"Hand pretraining data mismatch: {changed}")
