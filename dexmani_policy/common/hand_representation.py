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
    """Allow independent corpora while preserving action semantics and shared holdouts."""
    left, right = codec_data["contract"], policy_data["contract"]
    if left.keys() != right.keys() or any(left[key] != right[key] for key in left if key != "dt"):
        raise ValueError("Codec/policy hand action contract mismatch")
    if not math.isclose(left["dt"], right["dt"], rel_tol=1e-6, abs_tol=0):
        raise ValueError("Codec/policy dt mismatch; no automatic resampling")

    source, target = codec_data["provenance"], policy_data["provenance"]
    source_identity = source["source_id"] or source["zarr_path"]
    target_identity = target["source_id"] or target["zarr_path"]
    if source_identity != target_identity:
        return

    source_ids, target_ids = source["episode_ids"], target["episode_ids"]
    if source_ids is None or target_ids is None:
        if source["episode_ends"] != target["episode_ends"] or source_ids != target_ids:
            raise ValueError("Same-source subsets require stable episode_ids")
        source_ids = target_ids = range(len(source["episode_ends"]))

    source_val = {source_ids[i] for i in source["val_episodes"]}
    target_val = {target_ids[i] for i in target["val_episodes"]}
    shared = set(source_ids).intersection(target_ids)
    # Matching validation membership also prevents leakage across disjoint train/val splits.
    if (source_val & shared) != (target_val & shared):
        raise ValueError("Same-source codec/policy split mismatch or validation leakage")


def validate_pretraining_data(current, manifest, pretrained, pretrained_split):
    # Paths and diagnostic counters are not identity or action semantics.
    for key in ("source_id", "episode_ids", "episode_ends", "trial_ids", "train_indices", "val_indices"):
        if manifest.get(key) != pretrained_split.get(key):
            raise ValueError(f"Pretraining/policy split mismatch: {key}")
    changed = [key for key, value in current.items() if key not in pretrained or pretrained[key] != value]
    if changed:
        raise ValueError(f"Hand pretraining data mismatch: {changed}")
    # Old artifacts recorded unique qualified rows. Full coverage proves that their
    # population matches the new episode population; otherwise require a new artifact.
    ends = manifest["episode_ends"]
    lengths = [end - start for start, end in zip([0] + ends[:-1], ends)]
    for split in ("train", "val"):
        key = f"{split}_action_rows"
        expected = sum(lengths[i] for i in manifest[f"{split}_indices"])
        if key in pretrained and pretrained[key] != expected:
            raise ValueError("Legacy hand artifact used window-filtered rows; rebuild with the episode population")
