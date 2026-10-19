"""Content provenance for the three hand representations; no runtime file access."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def implementation_version():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "dexmani_policy"], cwd=root, text=True,
        ).strip())
        return dict(revision=revision, implementation_dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(revision=None, implementation_dirty=None)


def metadata_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def source_contract(dataset):
    """Capture qualified source content once per training dataset instance."""
    if hasattr(dataset, "hand_source_contract"):
        return dataset.hand_source_contract
    buffer = dataset.replay_buffer
    attrs = buffer.attrs
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Hand representation data requires finite positive dt")
    order = ["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"]
    if attrs.get("hand_joint_order", order) != order or attrs.get("hand_units", "radians") != "radians":
        raise ValueError("Hand source must use XHand thumb/index/middle/ring/pinky order in radians")
    val = dataset.get_validation_dataset()
    splits = {}
    for name, child in (("train", dataset), ("val", val)):
        rows = np.empty(0, dtype=np.int64) if child is None else child.sampler.action_source_rows
        windows = np.empty((0, 4), dtype=np.int64) if child is None else child.sampler.indices
        obs_rows = np.empty(0, dtype=np.int64) if child is None else child.sampler.observation_source_rows
        leading = int(windows[:, 2].sum())
        trailing = int((dataset.horizon - windows[:, 3]).sum())
        splits[name] = dict(source_rows_sha256=array_sha256(rows), source_rows_count=len(rows),
                            windows_sha256=array_sha256(windows), windows_count=len(windows),
                            observation_rows_sha256=array_sha256(obs_rows),
                            leading_padding_frames=leading, trailing_padding_frames=trailing,
                            window_padding_fraction=(leading + trailing) / max(1, len(windows) * dataset.horizon))
    arrays = {}
    # Action/state content identifies the representation; observation validity fixes eligibility.
    for key in sorted({dataset.action_key, "joint_state"}):
        if key not in buffer:
            continue
        arr = buffer[key]
        digest = hashlib.sha256(str((arr.shape, np.dtype(arr.dtype).str)).encode())
        step = buffer.scan_chunk_rows(key)
        for start in range(0, arr.shape[0], step):
            digest.update(np.ascontiguousarray(arr[start:start + step]).tobytes())
        arrays[key] = digest.hexdigest()
    contract = dict(
        schema="hand_source.v1", action_key=dataset.action_key, hand_joint_order=order,
        hand_units="radians", data_revision=dataset.data_revision,
        task_name=attrs.get("task_name"), episode_ids=attrs.get("episode_ids"),
        episode_ends_sha256=array_sha256(np.asarray(buffer.episode_ends, dtype=np.int64)),
        train_mask=dataset.train_mask.tolist(), val_mask=dataset.val_mask.tolist(),
        arrays=arrays, splits=splits, dt=float(dt), obs_alignment=attrs.get("obs_alignment"),
        window=dict(horizon=dataset.horizon, obs_horizon=dataset.obs_horizon,
                    pad_before=dataset.pad_before, pad_after=dataset.pad_after),
        data_recipe=dataset.data_recipe,
    )
    if hasattr(dataset, "lamp_source_metadata"):
        contract["history"] = dataset.lamp_source_metadata
        contract["obs_alignment"] = dataset.lamp_source_metadata["obs_alignment"]
    if hasattr(dataset, "codec_source_metadata"):
        contract["reference"] = dataset.codec_source_metadata
        contract["obs_alignment"] = dataset.codec_source_metadata["obs_alignment"]
        contract["statistics_population"] = "train_windows_including_repeated_rows_and_padding"
    # Detach from mutable dataset recipes and normalize scalar/container types.
    dataset.hand_source_contract = json.loads(json.dumps(contract, allow_nan=False))
    dataset.hand_implementation_version = implementation_version()
    return dataset.hand_source_contract


def state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_source_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Hand artifact requires source metadata; regenerate the artifact")
    contract = metadata.get("source_contract")
    if (not isinstance(contract, dict) or contract.get("schema") != "hand_source.v1"
            or metadata.get("source_contract_sha256") != metadata_sha256(contract)):
        raise ValueError("Missing/invalid hand source contract; regenerate the artifact with current code")
    return contract


def artifact_record(kind, schema, sha256, metadata, module):
    expected_schema = {"lamp": "lamp_prior.v2", "temporal": "temporal_hand_codec.v1",
                       "dqrise": "hand_codebook.v3"}[kind]
    if (schema != expected_schema or not isinstance(sha256, str) or len(sha256) != 64
            or any(c not in "0123456789abcdef" for c in sha256)):
        raise ValueError("Invalid hand artifact schema/SHA; regenerate or use historical code")
    contract = validate_source_metadata(metadata)
    selection = metadata.get("selection")
    if (not isinstance(selection, dict)
            or not {"selection_step", "split", "metric", "value"} <= selection.keys()):
        raise ValueError("Missing artifact selection metadata; regenerate the artifact")
    if (type(selection["selection_step"]) is not int or selection["selection_step"] < 0
            or not isinstance(selection["value"], (int, float))
            or not math.isfinite(selection["value"])):
        raise ValueError("Invalid artifact selection step/metric value")
    if kind in {"lamp", "temporal"}:
        beta, target = selection.get("beta"), selection.get("target_beta")
        if (not isinstance(beta, (int, float)) or not isinstance(target, (int, float))
                or not math.isfinite(beta) or not 0 < target == beta
                or selection.get("selection_step", 0) < selection.get("eligible_step", 1)):
            raise ValueError("Representation selection precedes the complete positive KL schedule")
        expected_split = "validation" if any(contract["val_mask"]) else "final_no_holdout"
        if (selection["split"] != expected_split
                or not {"reconstruction", "kl", "physical_joint_mae"} <= selection.keys()):
            raise ValueError("Selection split/metrics do not match the source contract")
    return dict(kind=kind, schema=schema, sha256=sha256, source_contract=contract,
                source_contract_sha256=metadata_sha256(contract), selection=selection,
                representation_state_sha256=state_sha256(module),
                details={key: metadata[key] for key in (
                    "relative_statistics", "loss_reduction", "source_summary", "model_config",
                    "normalization_spec", "prototype_count", "ordering", "layer_weights",
                    "implementation",
                ) if key in metadata})


def validate_artifact_record(record, kind, module):
    if not isinstance(record, dict) or record.get("kind") != kind:
        raise ValueError("Checkpoint requires action_space_artifact; use historical code or retrain")
    validated = artifact_record(kind, record.get("schema"), record.get("sha256"), record, module)
    if record.get("representation_state_sha256") != validated["representation_state_sha256"]:
        raise ValueError("Embedded representation differs from action_space_artifact")
    details = record.get("details", {})
    if not isinstance(details.get("normalization_spec"), dict):
        raise ValueError("Artifact metadata requires the normalization recipe")
    if hasattr(module, "model_config") and details.get("model_config") != module.model_config:
        raise ValueError("Embedded representation architecture differs from artifact metadata")
    if kind == "dqrise":
        if (details.get("prototype_count") != module.num_codes
                or details.get("ordering") != "decoded_prototype_pca1_ascending"
                or details.get("layer_weights") != module.layer_weights.detach().cpu().tolist()):
            raise ValueError("Codebook metadata differs from embedded prototypes/weights")


def validate_selection_schedule(steps, beta, eligible_step):
    if not math.isfinite(beta) or beta <= 0 or steps < eligible_step:
        raise ValueError(f"Policy artifacts require positive beta and steps >= {eligible_step} "
                         "to complete KL warmup/ramp; shorten the schedule explicitly for smoke")


def policy_representation(model):
    from dexmani_policy.agents.core.lamp_policy import LAMPPolicy
    from dexmani_policy.agents.core.temporal_latent_flow_policy import TemporalLatentFlowPolicy
    from dexmani_policy.agents.core.dqrise_policy import DQRISEPolicy

    if isinstance(model, LAMPPolicy):
        return "lamp", model.prior
    if isinstance(model, TemporalLatentFlowPolicy):
        return "temporal", model.codec
    if isinstance(model, DQRISEPolicy):
        return "dqrise", model.hand_codebook
    return None


def restore_artifact_metadata(model, config):
    representation = policy_representation(model)
    if representation is None:
        return
    record = config.get("action_space_artifact")
    from omegaconf import OmegaConf

    if OmegaConf.is_config(record):
        record = OmegaConf.to_container(record, resolve=True)
    kind, module = representation
    validate_artifact_record(record, kind, module)
    from dexmani_policy.utils.config import resolve_normalization_spec

    if record["details"]["normalization_spec"] != resolve_normalization_spec(config):
        raise ValueError("Saved artifact/policy normalization recipes disagree")
    model.action_space_artifact = record
